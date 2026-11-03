"""Preprocessing configs for the classifier.

A: no augmentation  -> Resize(256) + CenterCrop(224) + ImageNet normalize
B: augmentation     -> RandomResizedCrop + flip + brightness/contrast/saturation jitter
Validation and test always use the A pipeline.
"""
from torchvision import transforms as T

from src.data.common import IMAGENET_MEAN, IMAGENET_STD


def build_crop_transform(preproc: str = "A", train: bool = False, size: int = 224):
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train and preproc == "B":
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
            T.ToTensor(),
            normalize,
        ])
    if preproc not in ("A", "B"):
        raise ValueError(f"preproc must be 'A' or 'B', got {preproc!r}")
    return T.Compose([T.Resize(256), T.CenterCrop(size), T.ToTensor(), normalize])
