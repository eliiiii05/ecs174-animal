"""Step 1: read and download the Hugging Face dataset, save JPEGs,
make the fixed split, and write one COCO-format annotation file per split.

    python -m src.data.prepare_dataset                      # full dataset -> data/

Outputs:
    <out>/images/<file_name>.jpg
    <out>/annotations/instances_{train,val,test}.json
    <out>/stats.json
    splits/split_seed42.json
"""
import argparse
import io
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.common import CLASS_TO_IDX, CLASSES, HF_DATASET
from src.utils import REPO_ROOT, load_json, resolve, save_json

SPLITS = ("train", "val", "test")


COLUMNS = ["cocoid", "file_name", "width", "height", "image", "animal_bboxes_xywh", "animal_bbox_labels"]


def find_parquet_shards(name: str, parquet_dir: str | None):
    """Locate the downloaded parquet shards of the `images` config.

    We read the parquet files directly instead of calling datasets.load_dataset(),
    which would write a second ~12 GB Arrow copy of the data to the cache."""
    if parquet_dir:
        root = resolve(parquet_dir)
    else:
        from huggingface_hub import snapshot_download
        kw = dict(repo_id=name, repo_type="dataset", allow_patterns=["images/*.parquet"])
        try:
            root = Path(snapshot_download(local_files_only=True, **kw))  # already downloaded: no network
            print(f"Using cached download in {root}")
        except Exception:
            root = None
        shards = _list_shards(root) if root else []
        expected = _expected_total(shards)
        if not shards or (expected and len(shards) < expected):
            missing = (expected - len(shards)) if shards else "all"
            print(f"Downloading missing shards ({missing}) ...")
            root = Path(snapshot_download(**kw))  # only fetches files that are not in the cache
    shards = _list_shards(root)
    if not shards:
        raise FileNotFoundError(f"no parquet shards found under {root}")
    expected = _expected_total(shards)
    print(f"Found {len(shards)} parquet shards" + (f" (expected {expected})" if expected else ""))
    if expected and len(shards) < expected:
        print("WARNING: some shards are missing; the dataset will be incomplete")
    return shards


def _list_shards(root: Path):
    return sorted(root.glob("images/*.parquet")) or sorted(root.glob("*.parquet"))


def _expected_total(shards):
    """train-00003-of-00025.parquet -> 25"""
    m = re.search(r"-of-(\d+)\.parquet$", shards[0].name) if shards else None
    return int(m.group(1)) if m else None


def iter_parquet_rows(shards):
    """Yield one dict per image, streaming through the shards in small batches (low memory)."""
    import pyarrow.parquet as pq

    n = 0
    for shard in shards:
        try:
            pf = pq.ParquetFile(shard)
            cols = [c for c in COLUMNS if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=64, columns=cols):
                for ex in batch.to_pylist():
                    yield ex
                    n += 1
        except OSError as e:
            raise SystemExit(
                f"\nERROR: cannot read {shard.name}: {e}\n"
                f"This downloaded file is damaged. Delete it, then rerun the same command;\n"
                f"only the missing shard will be downloaded again:\n"
                f"  rm -f \"$(readlink -f '{shard}')\" '{shard}'\n")
        print(f"  read {shard.name} ({n} images so far)")


def write_image(img_field, path: Path):
    """Save as JPEG. Raw JPEG bytes are copied as-is; anything else (the dataset
    stores many images as PNG) is re-encoded as JPEG quality 95 to save disk space.
    Returns (width, height) when the image had to be decoded, else None."""
    data = img_field.get("bytes") if isinstance(img_field, dict) else None
    if not data and isinstance(img_field, dict) and img_field.get("path"):
        data = Path(img_field["path"]).read_bytes()
    if data[:3] == b"\xff\xd8\xff":
        path.write_bytes(data)
        return None
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.save(path, "JPEG", quality=95)
    return img.size


def save_images_and_collect(rows, img_dir: Path):
    """Write each image to disk and keep a clean record of its boxes."""
    img_dir.mkdir(parents=True, exist_ok=True)
    records, dropped = [], 0
    for ex in rows:
        path = img_dir / ex["file_name"]
        if not path.exists():
            size = write_image(ex["image"], path)
            if size and (not ex.get("width") or not ex.get("height")):
                ex["width"], ex["height"] = size
        W, H = int(ex["width"]), int(ex["height"])
        boxes = []
        for (x, y, w, h), label in zip(ex["animal_bboxes_xywh"], ex["animal_bbox_labels"]):
            if label not in CLASS_TO_IDX:
                dropped += 1
                continue
            x0, y0 = max(0.0, float(x)), max(0.0, float(y))
            x1, y1 = min(float(W), float(x) + float(w)), min(float(H), float(y) + float(h))
            if x1 - x0 < 1 or y1 - y0 < 1:
                dropped += 1
                continue
            boxes.append({"bbox": [x0, y0, x1 - x0, y1 - y0], "label": label})
        if boxes:
            records.append({"id": int(ex["cocoid"]), "file_name": ex["file_name"],
                            "width": W, "height": H, "boxes": boxes})
    records.sort(key=lambda r: r["id"])
    ann_id = 1  # globally unique annotation ids (crops are named by them)
    for r in records:
        for b in r["boxes"]:
            b["ann_id"] = ann_id
            ann_id += 1
    return records, dropped


def make_split(records, seed=42, val_frac=0.15, test_frac=0.15):
    """Split by IMAGE, stratified by the rarest class in each image,
    so rare classes (bear) appear in every split and no image is in two splits."""
    img_freq = Counter(lab for r in records for lab in {b["label"] for b in r["boxes"]})
    groups = defaultdict(list)
    for r in records:
        labels = {b["label"] for b in r["boxes"]}
        key = min(labels, key=lambda lab: (img_freq[lab], lab))
        groups[key].append(r["id"])

    rng = np.random.default_rng(seed)
    split = {s: [] for s in SPLITS}
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        n = len(ids)
        n_test, n_val = int(round(n * test_frac)), int(round(n * val_frac))
        split["test"] += ids[:n_test]
        split["val"] += ids[n_test:n_test + n_val]
        split["train"] += ids[n_test + n_val:]
    return {s: sorted(int(i) for i in split[s]) for s in SPLITS}


def to_coco(records, image_ids) -> dict:
    keep = set(image_ids)
    images, annotations = [], []
    for r in records:
        if r["id"] not in keep:
            continue
        images.append({"id": r["id"], "file_name": r["file_name"],
                       "width": r["width"], "height": r["height"]})
        for b in r["boxes"]:
            x, y, w, h = b["bbox"]
            annotations.append({"id": b["ann_id"], "image_id": r["id"],
                                "category_id": CLASS_TO_IDX[b["label"]] + 1,
                                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                                "area": round(w * h, 2), "iscrowd": 0})
    categories = [{"id": i + 1, "name": c, "supercategory": "animal"} for i, c in enumerate(CLASSES)]
    return {"images": images, "annotations": annotations, "categories": categories}


def split_stats(coco: dict) -> dict:
    names = {c["id"]: c["name"] for c in coco["categories"]}
    per_class = Counter(names[a["category_id"]] for a in coco["annotations"])
    return {"images": len(coco["images"]), "boxes": len(coco["annotations"]),
            "per_class": {c: per_class.get(c, 0) for c in CLASSES}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output folder (relative to repo root)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--split-file", default=None, help="default: splits/split_seed<seed>.json")
    ap.add_argument("--parquet-dir", default=None,
                    help="folder that holds images/*.parquet (default: find it in the Hugging Face cache)")
    args = ap.parse_args()

    out = resolve(args.out)
    if args.split_file:
        split_path = resolve(args.split_file)
    else:
        split_path = REPO_ROOT / "splits" / f"split_seed{args.seed}.json"

    out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out).free / 1024 ** 3
    print(f"Free disk space at {out}: {free_gb:.1f} GB")
    if free_gb < 8:
        print("WARNING: less than 8 GB free; the full image export needs about 5-6 GB")

    shards = find_parquet_shards(HF_DATASET, args.parquet_dir)
    records, dropped = save_images_and_collect(iter_parquet_rows(shards), out / "images")
    print(f"{len(records)} images with boxes kept, {dropped} boxes dropped (degenerate or unknown class)")

    if split_path.exists():
        split = load_json(split_path)
        print(f"Reusing existing split {split_path.relative_to(REPO_ROOT)}")
        known = {r["id"] for r in records}
        missing = sum(1 for s in SPLITS for i in split[s] if i not in known)
        if missing:
            print(f"WARNING: {missing} image ids in the split file were not found in the download")
    else:
        split = make_split(records, args.seed, args.val_frac, args.test_frac)
        save_json(split, split_path)
        print(f"Wrote new split {split_path.relative_to(REPO_ROOT)}")

    a, b, c = (set(split[s]) for s in SPLITS)
    assert not (a & b or a & c or b & c), "split overlap: an image is in two splits"

    stats = {}
    for s in SPLITS:
        coco = to_coco(records, split[s])
        save_json(coco, out / "annotations" / f"instances_{s}.json")
        stats[s] = split_stats(coco)
        print(f"{s:>5}: {stats[s]['images']:6d} images, {stats[s]['boxes']:6d} boxes")
    save_json(stats, out / "stats.json")
    print("Per-class box counts (train):", stats["train"]["per_class"])


if __name__ == "__main__":
    main()
