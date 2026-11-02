"""CLIP zero-shot baseline: classify GT crops using only the 10 class names (no training).

    python -m src.clip_zeroshot --split test
    python -m src.clip_zeroshot --split test --corruption lowlight --severity 2

Runs fine on free Colab. Appends metrics to results/results.csv.
"""
import argparse
import csv

import torch
from PIL import Image

from src.data.common import CLASSES
from src.data.corruptions import KINDS, apply_corruption
from src.engine import get_device
from src.metrics import accuracy_by_bucket, classification_metrics
from src.utils import REPO_ROOT, append_results, resolve

TEMPLATES = ["a photo of a {}.", "a photo of the {} in the wild.", "a close-up photo of a {}."]
MODEL_NAME = "openai/clip-vit-base-patch32"


@torch.no_grad()
def class_text_features(model, proc, device):
    texts = [t.format(c) for c in CLASSES for t in TEMPLATES]
    tok = proc(text=texts, return_tensors="pt", padding=True).to(device)
    f = model.get_text_features(**tok)
    f = f / f.norm(dim=-1, keepdim=True)
    f = f.view(len(CLASSES), len(TEMPLATES), -1).mean(dim=1)  # prompt ensembling
    return f / f.norm(dim=-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--corruption", default="none", choices=KINDS)
    ap.add_argument("--severity", type=int, default=1, choices=[1, 2])
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    from transformers import CLIPModel, CLIPProcessor
    device = get_device()
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_NAME)
    text_f = class_text_features(model, proc, device)

    data = resolve(args.data_root)
    with open(data / "crops" / f"{args.split}.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    y_true, y_pred = [], []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            imgs = [apply_corruption(Image.open(data / r["path"]).convert("RGB"), args.corruption,
                                     args.severity, seed=start + k) for k, r in enumerate(batch)]
            px = proc(images=imgs, return_tensors="pt").to(device)
            img_f = model.get_image_features(**px)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            y_pred += (img_f @ text_f.T).argmax(dim=-1).tolist()
            y_true += [int(r["label"]) for r in batch]

    metrics = classification_metrics(y_true, y_pred)
    metrics.update(accuracy_by_bucket(y_true, y_pred, [r["size_bucket"] for r in rows]))
    eval_set = args.split if args.corruption == "none" else f"{args.split}_{args.corruption}{args.severity}"
    append_results("clip_zeroshot", "clip-vit-b32", "zero-shot", eval_set, metrics)
    print(f"CLIP zero-shot on {eval_set}: accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
