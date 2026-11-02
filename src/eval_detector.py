"""Evaluate a trained detector (D1 class-agnostic or D2 10-class) on full images.

    python -m src.eval_detector --ckpt runs/D1_agnostic/best.pt --split test
    python -m src.eval_detector --ckpt runs/D2_10class/best.pt --split test --corruption blur --severity 1
"""
import argparse

import torch

from src.data.corruptions import KINDS
from src.data.datasets import DetectionDataset
from src.det_eval import coco_eval, load_gt, multi_animal_stats, per_class_ap50, run_detector
from src.engine import get_device
from src.models.detector import build_detector
from src.utils import REPO_ROOT, append_results, resolve, save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--corruption", default="none", choices=KINDS)
    ap.add_argument("--severity", type=int, default=1, choices=[1, 2])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(resolve(args.ckpt), map_location=device, weights_only=False)
    cfg = ck["config"]
    agnostic = cfg["agnostic"]
    model = build_detector(2 if agnostic else 11, cfg["min_size"], cfg["max_size"], cfg["trainable_layers"],
                           pretrained_backbone=False).to(device)
    model.load_state_dict(ck["model"])

    data = resolve(args.data_root or cfg["data_root"])
    gt_json = data / "annotations" / f"instances_{args.split}.json"
    ds = DetectionDataset(gt_json, data / "images", agnostic=agnostic, train=False,
                          corruption=args.corruption, severity=args.severity)
    results = run_detector(model, ds, device, args.batch_size, args.num_workers)

    gt = load_gt(gt_json, agnostic=agnostic)
    stats, E = coco_eval(gt, results, verbose=True)
    if E is not None:
        stats.update(per_class_ap50(E))
    stats.update(multi_animal_stats(load_gt(gt_json), results))  # multi-animal stats use real classes for "species"

    eval_set = args.split if args.corruption == "none" else f"{args.split}_{args.corruption}{args.severity}"
    if agnostic:
        eval_set += "_agnostic"
    tag = f"{cfg['run_name']}_{eval_set}"
    save_json(results, REPO_ROOT / "results" / "predictions" / f"{tag}.json")
    append_results(cfg["run_name"], "fasterrcnn_r18_fpn", cfg["_config_file"], eval_set, stats)
    print(f"{tag}: AP50 {stats['AP50']:.4f}  AP {stats['AP']:.4f}  AP_small {stats['AP_small']:.4f}")


if __name__ == "__main__":
    main()
