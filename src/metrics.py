"""Classification metrics and plots (no GPU needed)."""
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.data.common import CLASSES, NUM_CLASSES


def classification_metrics(y_true, y_pred) -> dict:
    labels = list(range(NUM_CLASSES))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    _, _, f_cls, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {"accuracy": accuracy_score(y_true, y_pred), "macro_precision": p, "macro_recall": r, "macro_f1": f}
    out.update({f"f1_{c}": v for c, v in zip(CLASSES, f_cls)})
    return out


def accuracy_by_bucket(y_true, y_pred, buckets) -> dict:
    hits, totals = defaultdict(int), defaultdict(int)
    for t, p, b in zip(y_true, y_pred, buckets):
        totals[b] += 1
        hits[b] += int(t == p)
    return {f"accuracy_{b}": hits[b] / totals[b] for b in ("small", "medium", "large") if totals[b]}


def plot_confusion(y_true, y_pred, path, title="Confusion matrix (row-normalized)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).astype(float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
