"""Shared helpers for running models."""
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: CUDA not available, running on CPU (very slow)")
    return torch.device("cpu")


@torch.no_grad()
def predict_classifier(model, loader, device, amp=True):
    """Returns (y_true, y_pred, probs, indices) as Python lists / a numpy array."""
    model.eval()
    y_true, y_pred, probs, idx = [], [], [], []
    for x, y, i in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
        p = logits.float().softmax(dim=1).cpu()
        probs.append(p)
        y_pred += p.argmax(dim=1).tolist()
        y_true += list(map(int, y))
        idx += list(map(int, i))
    return y_true, y_pred, torch.cat(probs).numpy(), idx
