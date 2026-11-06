"""Detector + classifier inference: detector D1 finds animals -> crop each box -> classifier C1 names it.

Evaluate on a split (10-class mAP, comparable with D2):
    python -m src.pipeline --det runs/D1_agnostic/best.pt --cls runs/C1_finetune_aug/best.pt --split test
Run on one image and save a picture with boxes (demo):
    python -m src.pipeline --det runs/D1_agnostic/best.pt --cls runs/C1_finetune_aug/best.pt \\
        --image some_photo.jpg --out-image results/figures/demo.png
"""
import argparse

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from src.data.common import CLASSES, expand_box
from src.data.corruptions import KINDS, apply_corruption
from src.data.transforms import build_crop_transform
from src.det_eval import coco_eval, load_gt, multi_animal_stats, per_class_ap50
from src.engine import get_device
from src.models.classifier import build_classifier
from src.models.detector import build_detector
from src.utils import REPO_ROOT, append_results, load_json, resolve, save_json


def load_models(det_ckpt, cls_ckpt, device):
    dk = torch.load(resolve(det_ckpt), map_location=device, weights_only=False)
    dcfg = dk["config"]
    assert dcfg["agnostic"], "the pipeline expects the class-agnostic detector (D1)"
    det = build_detector(2, dcfg["min_size"], dcfg["max_size"], dcfg["trainable_layers"], pretrained_backbone=False)
    det.load_state_dict(dk["model"])
    ck = torch.load(resolve(cls_ckpt), map_location=device, weights_only=False)
    cls = build_classifier(pretrained=False)
    cls.load_state_dict(ck["model"])
    return det.to(device).eval(), cls.to(device).eval(), dcfg, ck["config"]


@torch.no_grad()
def detect_and_classify(img: Image.Image, det, cls, eval_tf, device, score_thr=0.05, margin=0.10):
    """Returns a list of {bbox (xywh), category_id (1-10), score} for one PIL image."""
    out = det([TF.to_tensor(img).to(device)])[0]
    b = out["boxes"]
    # boxes under 1 px wide or tall cannot be cropped (and cannot match an animal at IoU 0.5)
    keep = (out["scores"] >= score_thr) & (b[:, 2] - b[:, 0] >= 1) & (b[:, 3] - b[:, 1] >= 1)
    boxes, s_det = out["boxes"][keep].tolist(), out["scores"][keep].tolist()
    if not boxes:
        return []
    W, H = img.size
    crops = torch.stack([eval_tf(img.crop(expand_box(x1, y1, x2 - x1, y2 - y1, W, H, margin)))
                         for x1, y1, x2, y2 in boxes]).to(device)
    conf, label = cls(crops).softmax(dim=1).max(dim=1)
    return [{"bbox": [x1, y1, x2 - x1, y2 - y1], "category_id": int(k) + 1, "score": float(sd * c)}
            for (x1, y1, x2, y2), sd, c, k in zip(boxes, s_det, conf.tolist(), label.tolist())]


def draw(img, dets, path, min_score=0.5):
    img = img.copy()
    d = ImageDraw.Draw(img)
    for r in dets:
        if r["score"] < min_score:
            continue
        x, y, w, h = r["bbox"]
        d.rectangle([x, y, x + w, y + h], outline=(255, 64, 64), width=3)
        d.text((x + 3, y + 3), f"{CLASSES[r['category_id'] - 1]} {r['score']:.2f}", fill=(255, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", required=True)
    ap.add_argument("--cls", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--corruption", default="none", choices=KINDS)
    ap.add_argument("--severity", type=int, default=1, choices=[1, 2])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--image", default=None, help="run on a single image instead of a split")
    ap.add_argument("--out-image", default="results/figures/demo.png")
    args = ap.parse_args()

    device = get_device()
    det, cls, dcfg, ccfg = load_models(args.det, args.cls, device)
    eval_tf = build_crop_transform("A", train=False)

    if args.image:
        img = apply_corruption(Image.open(args.image).convert("RGB"), args.corruption, args.severity)
        dets = detect_and_classify(img, det, cls, eval_tf, device, args.score_thr)
        draw(img, dets, resolve(args.out_image))
        for r in sorted(dets, key=lambda r: -r["score"]):
            if r["score"] >= 0.5:
                print(f"{CLASSES[r['category_id'] - 1]:>9}  score {r['score']:.2f}  box {[round(v) for v in r['bbox']]}")
        print(f"saved {args.out_image}")
        return

    data = resolve(args.data_root or dcfg["data_root"])
    gt_json = data / "annotations" / f"instances_{args.split}.json"
    images = load_json(gt_json)["images"]
    results = []
    for k, info in enumerate(images):
        img = Image.open(data / "images" / info["file_name"]).convert("RGB")
        img = apply_corruption(img, args.corruption, args.severity, seed=info["id"])
        for r in detect_and_classify(img, det, cls, eval_tf, device, args.score_thr):
            results.append({"image_id": info["id"], **r})
        if k % 500 == 0:
            print(f"{k}/{len(images)} images")

    gt = load_gt(gt_json)
    stats, E = coco_eval(gt, results, verbose=True)
    if E is not None:
        stats.update(per_class_ap50(E))
    stats.update(multi_animal_stats(gt, results))
    eval_set = args.split if args.corruption == "none" else f"{args.split}_{args.corruption}{args.severity}"
    run_id = f"{dcfg['run_name']}+{ccfg['run_name']}"
    save_json(results, REPO_ROOT / "results" / "predictions" / f"pipeline_{run_id}_{eval_set}.json")
    append_results(run_id, "two_stage", f"{dcfg['_config_file']}+{ccfg['_config_file']}", eval_set, stats)
    print(f"{run_id} on {eval_set}: mAP50 {stats['AP50']:.4f}  mAP {stats['AP']:.4f}")


if __name__ == "__main__":
    main()
