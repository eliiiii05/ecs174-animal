"""Evaluate a trained classifier on GT crops (optionally corrupted).

    python -m src.eval_classifier --ckpt runs/C1_finetune_aug/best.pt --split test
    python -m src.eval_classifier --ckpt runs/C1_finetune_aug/best.pt --corruption lowlight --severity 2

"""
import argparse
import csv

import torch
from torch.utils.data import DataLoader

from src.data.common import CLASSES
from src.data.corruptions import KINDS
from src.data.datasets import CropDataset
from src.data.transforms import build_crop_transform
from src.engine import get_device, predict_classifier
from src.metrics import accuracy_by_bucket, classification_metrics
from src.models.classifier import build_classifier
from src.utils import REPO_ROOT, append_results, resolve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--corruption", default="none", choices=KINDS)
    ap.add_argument("--severity", type=int, default=1, choices=[1, 2])
    ap.add_argument("--data-root", default=None, help="default: the data_root stored in the checkpoint")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(resolve(args.ckpt), map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_classifier(pretrained=False).to(device)
    model.load_state_dict(ck["model"])

    data = resolve(args.data_root or cfg["data_root"])
    ds = CropDataset(data / "crops" / f"{args.split}.csv", data, build_crop_transform("A", train=False),
                     corruption=args.corruption, severity=args.severity)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    y_true, y_pred, probs, idx = predict_classifier(model, loader, device)

    metrics = classification_metrics(y_true, y_pred)
    buckets = [ds.rows[i]["size_bucket"] for i in idx]
    metrics.update(accuracy_by_bucket(y_true, y_pred, buckets))

    eval_set = args.split if args.corruption == "none" else f"{args.split}_{args.corruption}{args.severity}"
    tag = f"{cfg['run_name']}_{eval_set}"
    pred_path = REPO_ROOT / "results" / "predictions" / f"{tag}.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ann_id", "image_id", "path", "label", "pred", "confidence", "size_bucket"])
        for k, i in enumerate(idx):
            r = ds.rows[i]
            w.writerow([r["ann_id"], r["image_id"], r["path"], y_true[k], y_pred[k],
                        round(float(probs[k].max()), 4), r["size_bucket"]])

    append_results(cfg["run_name"], "resnet18", cfg["_config_file"], eval_set, metrics)
    print(f"{tag}: accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}")
    print("  per-class F1: " + ", ".join(f"{c} {metrics['f1_' + c]:.3f}" for c in CLASSES))
    print(f"  predictions -> {pred_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
