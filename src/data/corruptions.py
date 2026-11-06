"""Image corruptions for the robustness tests (RQ4)."""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

KINDS = ("none", "lowlight", "blur", "contrast")
# severity -> parameters
LOWLIGHT = {1: (2.0, 0.02), 2: (3.0, 0.04)}   # (gamma, gaussian noise sigma on 0-1 scale)
BLUR = {1: 1.0, 2: 2.0}                        # gaussian blur radius (~sigma) in pixels
CONTRAST = {1: 0.5, 2: 0.3}                    # contrast factor


def apply_corruption(img: Image.Image, kind: str | None, severity: int = 1, seed: int = 0) -> Image.Image:
    if kind in (None, "none"):
        return img
    if kind == "lowlight":
        gamma, sigma = LOWLIGHT[severity]
        x = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        x = x ** gamma
        x = x + np.random.default_rng(seed).normal(0.0, sigma, x.shape).astype(np.float32)
        return Image.fromarray((np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=BLUR[severity]))
    if kind == "contrast":
        return ImageEnhance.Contrast(img).enhance(CONTRAST[severity])
    raise ValueError(f"unknown corruption {kind!r}; choose from {KINDS}")
