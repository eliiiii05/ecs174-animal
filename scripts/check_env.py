"""Check the environment before training: PyTorch, CUDA, GPU memory, and one
forward/backward pass of each model on dummy data.

    python scripts/check_env.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    import torchvision
    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__}   torchvision {torchvision.__version__}")
    if not torch.cuda.is_available():
        print("CUDA        NOT available -> install the CUDA build of PyTorch (see README)")
        sys.exit(1)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU         {props.name}, {props.total_memory / 1024 ** 3:.1f} GB, CUDA {torch.version.cuda}")
    for pkg in ("pycocotools", "datasets", "sklearn", "yaml", "transformers"):
        try:
            __import__(pkg)
            print(f"{pkg:<11} ok")
        except ImportError:
            print(f"{pkg:<11} MISSING -> pip install -r requirements.txt")

    from src.models.classifier import build_classifier
    from src.models.detector import build_detector
    dev = torch.device("cuda")

    cls = build_classifier().to(dev)
    x, y = torch.randn(128, 3, 224, 224, device=dev), torch.randint(0, 10, (128,), device=dev)
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = torch.nn.functional.cross_entropy(cls(x), y)
    loss.backward()
    torch.cuda.synchronize()
    print(f"classifier  batch 128 fwd+bwd ok  {time.time() - t:.2f}s  "
          f"peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB")
    del cls, x, y, loss
    torch.cuda.empty_cache()

    det = build_detector(2, 640, 1066).to(dev).train()
    imgs = [torch.rand(3, 480, 640, device=dev) for _ in range(4)]
    tgts = [{"boxes": torch.tensor([[50.0, 60.0, 300.0, 400.0]], device=dev),
             "labels": torch.tensor([1], device=dev)} for _ in range(4)]
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = sum(det(imgs, tgts).values())
    loss.backward()
    torch.cuda.synchronize()
    print(f"detector    batch 4 @640 fwd+bwd ok  {time.time() - t:.2f}s  "
          f"peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB")
    print("All good.")


if __name__ == "__main__":
    main()
