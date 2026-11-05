"""Turn results/results.csv into the tables used in the report.

    python scripts/summarize_results.py

Writes results/summary.md (one table per research question) and
results/results_latest.csv. results.csv is append-only, so a model that was
evaluated twice has two sets of rows; only the newest value of each
(run_id, eval_set, metric) is kept.

Logged training time includes any time the laptop slept, so the tables also give
an estimate from the run's log.csv: a typical epoch time x number of epochs, where
"typical" ignores epochs that took more than 1.5x the fastest one.
"""
import argparse
import csv
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLASSES = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
# run ids stay as in configs/ and runs/; tables and figures show what each run combines:
# initialisation · which layers train · augmentation
CLASSIFIERS = [("C1_finetune_aug", "ImageNet · all layers · aug"),
               ("C2_finetune_noaug", "ImageNet · all layers · no aug"),
               ("C3_linear_probe", "ImageNet · head only · aug"),
               ("C4_scratch", "Random init · all layers · aug"),
               ("clip_zeroshot", "CLIP zero-shot"),
               ("majority", "Majority class")]
PIPELINE = "D1_agnostic+C1_finetune_aug"
DETECTORS = [("D1_agnostic", "test_agnostic", "Detector, animal vs background"),
             ("D2_10class", "test", "One-stage: detector, 10 classes"),
             (PIPELINE, "test", "Two-stage: detector + classifier")]
CORRUPTIONS = [f"{k}{s}" for k in ("lowlight", "blur", "contrast") for s in (1, 2)]
DASH = "–"


def load_latest(path):
    """{(run_id, eval_set, metric): row}; later rows replace earlier ones."""
    latest = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            latest[(row["run_id"], row["eval_set"], row["metric"])] = row
    return latest


def value(latest, run, eval_set, metric):
    row = latest.get((run, eval_set, metric))
    return float(row["value"]) if row else None


def f3(v):
    return DASH if v is None else f"{v:.3f}"


def count(v):
    return DASH if v is None else f"{int(v)}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def estimated_minutes(runs_dir, run):
    """Typical epoch time x epochs from runs/<run>/log.csv. Epochs that took more than 1.5x the
    fastest one (the laptop slept) are left out of the typical time."""
    path = Path(runs_dir) / run / "log.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    if "epoch_seconds" in rows[0]:
        per_epoch = [float(r["epoch_seconds"]) / 60 for r in rows]
    else:
        per_epoch = [float(r["epoch_minutes"]) for r in rows]
    normal = [t for t in per_epoch if t <= 1.5 * min(per_epoch)]
    return statistics.median(normal) * len(per_epoch)


def validation_section(latest, runs_dir):
    out = ["## Validation (model selection during training)", ""]
    rows = []
    for run, name in CLASSIFIERS[:4]:
        logged, est = value(latest, run, "val", "train_minutes"), estimated_minutes(runs_dir, run)
        rows.append([name, f3(value(latest, run, "val", "accuracy")), f3(value(latest, run, "val", "macro_f1")),
                     count(value(latest, run, "val", "best_epoch")),
                     DASH if logged is None else f"{logged:.0f}", DASH if est is None else f"{est:.0f}"])
    out += [table(["Classifier", "val accuracy", "val macro-F1", "best epoch", "logged min", "est. min"], rows), ""]
    rows = []
    for run, ev, name in [("D1_agnostic", "val_agnostic", DETECTORS[0][2]), ("D2_10class", "val", DETECTORS[1][2])]:
        logged, est = value(latest, run, ev, "train_minutes"), estimated_minutes(runs_dir, run)
        rows.append([name, f3(value(latest, run, ev, "AP50")), f3(value(latest, run, ev, "AP")),
                     count(value(latest, run, ev, "best_epoch")),
                     DASH if logged is None else f"{logged / 60:.1f}", DASH if est is None else f"{est / 60:.1f}"])
    out += [table(["Detector", "val AP50", "val AP", "best epoch", "logged h", "est. h"], rows), "",
            "Logged time includes laptop sleep; est. = typical epoch time x epochs from `runs/<run>/log.csv`. "
            "The animal-vs-background detector's AP counts any box on an animal, so it is not comparable "
            "with the 10-class detector's AP.", ""]
    return out


def test_sections(latest):
    out = ["## RQ1: pretraining and fine-tuning (test, ground-truth crops)", ""]
    rows = [[name, *(f3(value(latest, run, "test", m)) for m in ("accuracy", "macro_precision", "macro_recall", "macro_f1")),
             *(f3(value(latest, run, "test", f"accuracy_{b}")) for b in ("small", "medium", "large"))]
            for run, name in CLASSIFIERS]
    out += [table(["Model", "accuracy", "macro precision", "macro recall", "macro-F1",
                   "acc. small", "acc. medium", "acc. large"], rows), "",
            "Test keeps every box; train/val dropped boxes with a short side under 32 px, "
            "so the small bucket is unseen during training.", ""]
    models = CLASSIFIERS[:5]
    rows = [[c, *(f3(value(latest, run, "test", f"f1_{c}")) for run, _ in models)] for c in CLASSES]
    out += ["Per-class F1 (test):", "", table(["Class", *(name for _, name in models)], rows), ""]

    out += ["## RQ2: detection, small objects, several animals (test, full images)", ""]
    cols = [("AP50", "AP50"), ("AP", "AP"), ("AP75", "AP75"), ("AP_small", "AP small"),
            ("AP_medium", "AP medium"), ("AP_large", "AP large"), ("AR100", "AR100")]
    rows = [[name, *(f3(value(latest, run, ev, m)) for m, _ in cols)] for run, ev, name in DETECTORS]
    out += [table(["Model", *(h for _, h in cols)], rows), ""]
    cols = [("multi_images", "imgs with ≥2 animals", count), ("multi_recall50", "recall@0.5", f3),
            ("multi_count_accuracy", "count acc.", f3), ("multispecies_images", "imgs with ≥2 species", count),
            ("multispecies_recall50", "recall@0.5", f3), ("multispecies_count_accuracy", "count acc.", f3)]
    rows = [[name, *(fmt(value(latest, run, ev, m)) for m, _, fmt in cols)] for run, ev, name in DETECTORS]
    out += [table(["Model", *(h for _, h, _ in cols)], rows), "",
            "Recall is class-agnostic (a box at IoU ≥ 0.5 counts whatever its label); "
            "count accuracy = boxes with score > 0.5 equals the number of animals.", ""]

    out += ["## RQ3: two-stage vs one-stage, per-class AP50 (test)", ""]
    rows = [[c, f3(value(latest, "D2_10class", "test", f"AP50_{c}")), f3(value(latest, PIPELINE, "test", f"AP50_{c}"))]
            for c in CLASSES]
    out += [table(["Class", DETECTORS[1][2], DETECTORS[2][2]], rows), ""]

    out += ["## RQ4: robustness (test with corruptions)", ""]
    sets = ["test"] + [f"test_{c}" for c in CORRUPTIONS]
    rows = [[f"{name} (macro-F1)", *(f3(value(latest, run, s, "macro_f1")) for s in sets)]
            for run, name in (CLASSIFIERS[0], CLASSIFIERS[1], CLASSIFIERS[4])]
    rows.append([f"{DETECTORS[2][2]} (mAP50)", *(f3(value(latest, PIPELINE, s, "AP50")) for s in sets)])
    out += [table(["Model", "clean", *CORRUPTIONS], rows), ""]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=REPO / "results" / "results.csv")
    ap.add_argument("--runs", default=REPO / "runs")
    ap.add_argument("--out", default=REPO / "results")
    args = ap.parse_args()

    latest = load_latest(args.csv)
    lines = ["# Results summary", "",
             "Generated by `scripts/summarize_results.py` from `results/results.csv` "
             "(newest value per run, eval set and metric).", ""]
    lines += validation_section(latest, args.runs)
    if any(k[1].startswith("test") for k in latest):
        lines += test_sections(latest)
    else:
        lines += ["## Test", "", "No test results yet: run `bash scripts/run_test.sh`.", ""]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines))
    with open(out / "results_latest.csv", "w", newline="") as f:
        rows = list(latest.values())
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id"])
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))
    print(f"\nwrote {out / 'summary.md'} and {out / 'results_latest.csv'}")


if __name__ == "__main__":
    main()
