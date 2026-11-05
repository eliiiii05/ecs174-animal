"""Shared constants for the animal dataset."""

# Fixed class order. Classifier label = index (0-9); COCO category_id = index + 1.
CLASSES = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HF_DATASET = "nirmalendu01/animal_dataset"


def size_bucket(area: float) -> str:
    """COCO size buckets: small < 32^2 <= medium < 96^2 <= large."""
    if area < 32 ** 2:
        return "small"
    if area < 96 ** 2:
        return "medium"
    return "large"


def expand_box(x, y, w, h, img_w, img_h, margin=0.10):
    """Grow an xywh box by `margin` of its size on every side, clipped to the image.
    Returns integer (x0, y0, x1, y1) with at least 1 pixel in each direction."""
    x0 = max(0.0, x - margin * w)
    y0 = max(0.0, y - margin * h)
    x1 = min(float(img_w), x + w + margin * w)
    y1 = min(float(img_h), y + h + margin * h)
    # keep x0/y0 inside the image so the crop is never empty (a box hugging the right or
    # bottom edge used to round to x0 == img_w and give a 0-pixel crop)
    x0, y0 = min(int(round(x0)), img_w - 1), min(int(round(y0)), img_h - 1)
    x1, y1 = max(x0 + 1, int(round(x1))), max(y0 + 1, int(round(y1)))
    return x0, y0, min(x1, img_w), min(y1, img_h)
