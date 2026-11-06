"""Classification metrics (no GPU needed)."""
from collections import defaultdict

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
