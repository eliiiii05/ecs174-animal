"""Majority-class baseline: predict the most common training class for every crop.

    python -m src.majority_baseline --split test

"""
import argparse
import csv
from collections import Counter

from src.data.common import CLASSES
from src.metrics import accuracy_by_bucket, classification_metrics
from src.utils import append_results, resolve


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    data = resolve(args.data_root)
    train = read_rows(data / "crops" / "train.csv")
    majority = Counter(int(r["label"]) for r in train).most_common(1)[0][0]

    rows = read_rows(data / "crops" / f"{args.split}.csv")
    y_true = [int(r["label"]) for r in rows]
    y_pred = [majority] * len(rows)
    metrics = classification_metrics(y_true, y_pred)
    metrics.update(accuracy_by_bucket(y_true, y_pred, [r["size_bucket"] for r in rows]))

    append_results("majority", "majority_class", f"predict_{CLASSES[majority]}", args.split, metrics)
    print(f"Majority class '{CLASSES[majority]}' on {args.split}: "
          f"accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
