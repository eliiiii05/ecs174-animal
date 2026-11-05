"""Figures for the report, from results/results.csv and runs/*/log.csv.

    python scripts/make_figures.py

Writes PNGs to results/figures/. A figure whose data does not exist yet (for
example the test-set figures before scripts/run_test.sh) is skipped.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from summarize_results import CLASSIFIERS, PIPELINE, REPO, load_latest, value  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# categorical colours, always assigned in this order (checked for colour-blind readers)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NAMES = dict(CLASSIFIERS)
KINDS = [("lowlight", "Low light"), ("blur", "Blur"), ("contrast", "Low contrast")]

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
})


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", color=INK)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def read_log(runs, run):
    path = Path(runs) / run / "log.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f)) or None


def save(fig, out, name):
    path = Path(out) / name
    fig.savefig(path)
    plt.close(fig)
    return path


def train_classifiers(latest, runs, out):
    logs = [(color, run, read_log(runs, run)) for color, (run, _) in zip(SLOTS, CLASSIFIERS[:4])]
    if not any(rows for _, _, rows in logs):
        return None
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    for color, run, rows in logs:
        if not rows:
            continue
        ep = [int(r["epoch"]) for r in rows]
        a.plot(ep, [float(r["val_macro_f1"]) for r in rows], color=color, lw=2, label=NAMES[run])
        b.plot(ep, [float(r["train_loss"]) for r in rows], color=color, lw=2, label=NAMES[run])
    style(a, "Validation macro-F1 by epoch", "epoch", "macro-F1")
    style(b, "Training loss by epoch", "epoch", "cross-entropy (label smoothing 0.1)")
    a.legend(loc="lower right")
    return save(fig, out, "train_classifiers.png")


def train_detectors(latest, runs, out):
    series = [(SLOTS[0], "D1_agnostic", "Detector, animal vs background"), (SLOTS[1], "D2_10class", "Detector, 10 classes")]
    logs = [(c, label, read_log(runs, run)) for c, run, label in series]
    if not any(rows for _, _, rows in logs):
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for x in (8.5, 11.5):  # lr x 0.1 from epoch 9 and epoch 12 (milestones 8, 11, 0-indexed)
        ax.axvline(x, color=GRID, lw=1)
        ax.text(x + 0.1, 0.02, "lr ×0.1", transform=ax.get_xaxis_transform(), color=INK2, fontsize=8.5)
    for color, label, rows in logs:
        if rows:
            ax.plot([int(r["epoch"]) for r in rows], [float(r["val_AP50"]) for r in rows],
                    color=color, lw=2, marker="o", ms=4.5, label=label)
    style(ax, "Validation AP50 by epoch", "epoch", "AP50")
    ax.legend(loc="center right")
    return save(fig, out, "train_detectors.png")


def test_classifiers(latest, runs, out):
    vals = [(NAMES[run], value(latest, run, "test", "macro_f1")) for run, _ in CLASSIFIERS]
    vals = sorted([(n, v) for n, v in vals if v is not None], key=lambda nv: nv[1])
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(vals) + 0.9))
    y = range(len(vals))
    ax.barh(y, [v for _, v in vals], height=0.5, color=SLOTS[0])
    for i, (_, v) in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(list(y), [n for n, _ in vals])
    ax.set_xlim(0, 1.08)
    style(ax, "Test macro-F1 on ground-truth crops")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    return save(fig, out, "test_classifiers.png")


def test_size_buckets(latest, runs, out):
    models = CLASSIFIERS[:5]
    buckets = ["small", "medium", "large"]
    data = [[value(latest, run, "test", f"accuracy_{b}") for b in buckets] for run, _ in models]
    if all(v is None for row in data for v in row):
        return None
    fig, ax = plt.subplots(figsize=(7, 3.4))
    width = 0.14
    for i, ((run, name), row, color) in enumerate(zip(models, data, SLOTS)):
        xs = [k + (i - 2) * width for k in range(len(buckets))]
        ax.bar(xs, [math.nan if v is None else v for v in row], width=width * 0.85, color=color, label=name)
    ax.set_xticks(range(len(buckets)), ["small (< 32² px)", "medium", "large (≥ 96² px)"])
    ax.set_ylim(0, 1)
    style(ax, "Test accuracy by box size", None, "accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8.5, handlelength=1)
    return save(fig, out, "test_size_buckets.png")


def robustness(latest, runs, out):
    cls_series = [(SLOTS[0], CLASSIFIERS[0]), (SLOTS[1], CLASSIFIERS[1]), (SLOTS[2], CLASSIFIERS[4])]

    def points(run, metric, kind):
        return [value(latest, run, s, metric) for s in ("test", f"test_{kind}1", f"test_{kind}2")]

    have_cls = any(v is not None for _, (run, _) in cls_series for k, _ in KINDS for v in points(run, "macro_f1", k)[1:])
    have_pipe = any(v is not None for k, _ in KINDS for v in points(PIPELINE, "AP50", k)[1:])
    if not (have_cls or have_pipe):
        return None
    nrows = int(have_cls) + int(have_pipe)
    fig, axes = plt.subplots(nrows, 3, figsize=(9, 3.0 * nrows), sharey="row", squeeze=False)
    r = 0
    if have_cls:
        for j, (kind, title) in enumerate(KINDS):
            ax = axes[r][j]
            for color, (run, name) in cls_series:
                ys = [math.nan if v is None else v for v in points(run, "macro_f1", kind)]
                ax.plot([0, 1, 2], ys, color=color, lw=2, marker="o", ms=4.5, label=name)
            style(ax, title, None, "macro-F1" if j == 0 else None)
            ax.set_xticks([0, 1, 2], ["clean", "severity 1", "severity 2"])
        handles, labels = axes[r][0].get_legend_handles_labels()
        r += 1
    if have_pipe:
        for j, (kind, title) in enumerate(KINDS):
            ax = axes[r][j]
            ys = [math.nan if v is None else v for v in points(PIPELINE, "AP50", kind)]
            ax.plot([0, 1, 2], ys, color=SLOTS[0], lw=2, marker="o", ms=4.5)
            style(ax, f"{title}: detector + classifier", None, "mAP50" if j == 0 else None)
            ax.set_xticks([0, 1, 2], ["clean", "severity 1", "severity 2"])
    fig.tight_layout(rect=(0, 0, 1, 0.94 if nrows == 1 else 0.96))
    if have_cls:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    return save(fig, out, "robustness.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=REPO / "results" / "results.csv")
    ap.add_argument("--runs", default=REPO / "runs")
    ap.add_argument("--out", default=REPO / "results" / "figures")
    args = ap.parse_args()
    latest = load_latest(args.csv) if Path(args.csv).exists() else {}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for make in (train_classifiers, train_detectors, test_classifiers, test_size_buckets, robustness):
        path = make(latest, args.runs, args.out)
        print(f"{make.__name__:>18}: {path if path else 'skipped (no data yet)'}")


if __name__ == "__main__":
    main()
