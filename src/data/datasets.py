"""PyTorch datasets: single-animal crops (classifier) and full images with boxes (detector)."""
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.common import NUM_CLASSES
from src.data.corruptions import apply_corruption


class CropDataset(Dataset):
    """Reads <data>/crops/<split>.csv. Returns (image_tensor, label, index)."""

    def __init__(self, csv_path, data_root, transform, corruption=None, severity=1):
        self.root = Path(data_root)
        with open(csv_path, newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.labels = [int(r["label"]) for r in self.rows]
        self.transform = transform
        self.corruption, self.severity = corruption, severity

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = Image.open(self.root / self.rows[i]["path"]).convert("RGB")
        img = apply_corruption(img, self.corruption, self.severity, seed=i)
        return self.transform(img), self.labels[i], i


def make_balanced_sampler(labels):
    """Draw every class with roughly equal probability (bear is ~8x rarer than bird)."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(labels), replacement=True)


class DetectionDataset(Dataset):
    """Full images + boxes from a COCO json. Returns (image_tensor, target, image_id).

    agnostic=True  -> every box gets label 1 ("animal")      [detector D1]
    agnostic=False -> label = COCO category_id (1..10)        [detector D2]
    """

    def __init__(self, coco_json, img_dir, agnostic=True, train=True, corruption=None, severity=1):
        from pycocotools.coco import COCO
        self.coco, self.img_dir = COCO(str(coco_json)), Path(img_dir)
        self.ids = sorted(self.coco.getImgIds())
        self.agnostic, self.train = agnostic, train
        self.corruption, self.severity = corruption, severity

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        img = apply_corruption(img, self.corruption, self.severity, seed=img_id)
        anns = [a for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
                if a["bbox"][2] > 1 and a["bbox"][3] > 1]
        boxes = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns)],
                             dtype=torch.float32).reshape(-1, 4)
        if self.agnostic:
            labels = torch.ones(len(anns), dtype=torch.int64)
        else:
            labels = torch.tensor([a["category_id"] for a in anns], dtype=torch.int64)
        img = TF.to_tensor(img)
        if self.train and random.random() < 0.5:  # horizontal flip, boxes flipped too
            img = img.flip(-1)
            W = img.shape[-1]
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        return img, {"boxes": boxes, "labels": labels}, img_id


def detection_collate(batch):
    """Top-level (picklable) collate: keeps variable-size images as a tuple."""
    return tuple(zip(*batch))
