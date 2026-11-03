"""Step 2: cut every ground-truth box out of its image and cache it for the classifier.

    python -m src.data.make_crops --data data

Each box is expanded by --margin on every side, the crop's short side is resized
to --short-side, and it is saved as <data>/crops/<split>/<ann_id>.jpg.
Train/val skip boxes whose short side is under --min-side pixels (too blurry to
classify); the test split keeps every box so results can be reported by size.
An index CSV per split is written to <data>/crops/<split>.csv.
"""
import argparse
import csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from src.data.common import CLASSES, expand_box, size_bucket
from src.utils import load_json, resolve

CSV_FIELDS = ["path", "label", "class_name", "ann_id", "image_id", "box_w", "box_h", "area", "size_bucket"]


def resize_short_side(img: Image.Image, short: int) -> Image.Image:
    w, h = img.size
    scale = short / min(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)


def process_image(job):
    img_path, anns, out_dir, rel_dir, margin, short, min_side, keep_small = job
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    rows = []
    for a in anns:
        x, y, w, h = a["bbox"]
        if not keep_small and min(w, h) < min_side:
            continue
        crop = img.crop(expand_box(x, y, w, h, W, H, margin))
        crop = resize_short_side(crop, short)
        name = f"{a['id']}.jpg"
        crop.save(out_dir / name, quality=95)
        label = a["category_id"] - 1
        rows.append({"path": f"{rel_dir}/{name}", "label": label, "class_name": CLASSES[label],
                     "ann_id": a["id"], "image_id": a["image_id"], "box_w": round(w, 1),
                     "box_h": round(h, 1), "area": a["area"], "size_bucket": size_bucket(a["area"])})
    return rows


def make_split_crops(data: Path, split: str, margin, short, min_side, workers):
    coco = load_json(data / "annotations" / f"instances_{split}.json")
    files = {im["id"]: im["file_name"] for im in coco["images"]}
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)

    out_dir = data / "crops" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_small = split == "test"
    jobs = [(data / "images" / files[i], anns, out_dir, f"crops/{split}", margin, short, min_side, keep_small)
            for i, anns in sorted(by_image.items())]

    rows = []
    if workers > 1:
        with Pool(workers) as pool:
            for r in pool.imap(process_image, jobs, chunksize=16):
                rows.extend(r)
    else:
        for job in jobs:
            rows.extend(process_image(job))

    with open(data / "crops" / f"{split}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    skipped = len(coco["annotations"]) - len(rows)
    print(f"{split:>5}: {len(rows):6d} crops written, {skipped} tiny boxes skipped")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--short-side", type=int, default=256)
    ap.add_argument("--min-side", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()
    data = resolve(args.data)
    for split in args.splits:
        make_split_crops(data, split, args.margin, args.short_side, args.min_side, args.workers)


if __name__ == "__main__":
    main()
