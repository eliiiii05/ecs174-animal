"""Write the report's result tables as LaTeX (booktabs three-line tables) from results/results.csv.

    python scripts/make_latex_tables.py

Writes one file per table to report/tables/, which report/main.tex pulls in with \\input.
Every number comes from results.csv (newest row per run, eval set and metric), so rerunning
this after a new evaluation updates the report. Scores are percentages with one decimal;
the best value of each comparison is bold. The training-setup table is read from configs/.
"""
import argparse
import json
import math
from pathlib import Path

import yaml

from summarize_results import CLASSES, CORRUPTIONS, PIPELINE, REPO, load_latest, value

DASH = "--"
CHECK, CROSS = r"\cmark", r"\xmark"

# classifier runs as (run id, init, trained layers, augmentation)
CLASSIFIER_ROWS = [("C1_finetune_aug", "ImageNet", "all", CHECK),
                   ("C2_finetune_noaug", "ImageNet", "all", CROSS),
                   ("C3_linear_probe", "ImageNet", "head only", CHECK),
                   ("C4_scratch", "Random", "all", CHECK)]
BASELINE_ROWS = [("clip_zeroshot", "CLIP zero-shot (ViT-B/32)"), ("majority", "Majority class")]
DETECTOR_ROWS = [("D1_agnostic", "test_agnostic", r"Animal vs.\ background"),
                 ("D2_10class", "test", "One-stage, 10 classes"),
                 (PIPELINE, "test", "Two-stage, detector + classifier")]
# torchvision resnet_fpn_backbone(trainable_layers=k) freezes everything below the last k stages
FROZEN = {5: "none", 4: "conv1", 3: "conv1, layer1", 2: "conv1 to layer2", 1: "conv1 to layer3", 0: "whole backbone"}
# the per-class header names each classifier by its combination
PERCLASS_HEADER = [r" & \multicolumn{3}{c}{ImageNet init} & Random init & CLIP \\",
                   r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}",
                   r"Class & all, aug. & all, no aug. & head, aug. & all, aug. & zero-shot \\"]


def pct(v):
    """Percentage with one decimal. Values under 10 get a phantom digit, so every number in a
    centred column has the same width and the decimal points line up."""
    if v is None:
        return DASH
    text = f"{100 * v:.1f}"
    return r"\phantom{0}" + text if len(text) == 3 else text


def signed(points):
    """A signed difference in points, padded like pct()."""
    text = f"{abs(points):.1f}"
    return (r"\phantom{0}" if len(text) == 3 else "") + "$" + ("-" if points < 0 else "+") + text + "$"


def power10(x):
    e = round(math.log10(x))
    return rf"$10^{{{e}}}$" if abs(math.log10(x) - e) < 1e-9 else f"{x:g}"


def bold_best(matrix, axis="col", skip=()):
    """Format a matrix of fractions (None = missing) as percentages, bolding the maximum of each
    column (axis='col') or row (axis='row'). Rows listed in `skip` never count as best."""
    text = [[pct(v) for v in row] for row in matrix]
    lines = range(len(matrix[0])) if axis == "col" else range(len(matrix))
    for k in lines:
        cells = [(i, k) for i in range(len(matrix))] if axis == "col" else [(k, j) for j in range(len(matrix[0]))]
        cells = [(i, j) for i, j in cells if matrix[i][j] is not None and i not in skip]
        if len(cells) < 2:
            continue
        best = max(round(100 * matrix[i][j], 1) for i, j in cells)
        for i, j in cells:
            if round(100 * matrix[i][j], 1) == best:
                text[i][j] = rf"\textbf{{{text[i][j]}}}"
    return text


def table(caption, label, colspec, header, body, note=None, small=False):
    """A booktabs table; the note sits under it at the table's own width (threeparttable)."""
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{threeparttable}"]
    if small:
        lines.append(r"\small")
    lines += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *header, r"\midrule", *body,
              r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [r"\begin{tablenotes}[flushleft]\footnotesize", rf"\item[] {note}", r"\end{tablenotes}"]
    lines += [r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def row(*cells):
    return " & ".join(cells) + r" \\"


def dataset_table(stats_path):
    if not stats_path.exists():
        shown = stats_path.relative_to(REPO) if stats_path.is_relative_to(REPO) else stats_path
        return (f"% {shown} not found: run python -m src.data.prepare_dataset, then this script again.\n"
                r"\textit{Dataset table missing: run \texttt{scripts/make\_latex\_tables.py} after preparing the data.}" "\n")
    stats = json.loads(stats_path.read_text())
    splits = ["train", "val", "test"]
    body = [row(c, *(f"{stats[s]['per_class'].get(c, 0):,}" for s in splits)) for c in CLASSES]
    body += [r"\midrule",
             row("All boxes", *(f"{stats[s]['boxes']:,}" for s in splits)),
             row("Images", *(f"{stats[s]['images']:,}" for s in splits))]
    return table("Number of annotated boxes per class and split. The split is by image (70/15/15, seed 42), "
                 "so no image appears in two splits.", "tab:dataset", "lrrr",
                 [row("Class", "Train", "Val", "Test")], body,
                 note="Counts are before the crop filter: classifier training and validation crops drop boxes "
                      "whose short side is under 32 px; the test set keeps every box.")


def setup_table(configs):
    cls = yaml.safe_load((configs / "cls_C1_finetune_aug.yaml").read_text())
    scratch = yaml.safe_load((configs / "cls_C4_scratch.yaml").read_text())
    det = yaml.safe_load((configs / "det_D1_agnostic.yaml").read_text())
    warm = cls["warmup_epochs"]
    classifier = [("Model", "ResNet-18"),
                  ("Input", r"$224 \times 224$ crop"),
                  ("Optimizer", "AdamW"),
                  ("Learning rate", f"{power10(cls['lr_backbone'])} backbone, {power10(cls['lr_head'])} head$^*$"),
                  ("Weight decay", power10(cls["weight_decay"])),
                  ("Warm-up", f"{warm} epoch{'s' if warm != 1 else ''}"),
                  ("Decay", "cosine"),
                  ("Epochs", f"{cls['epochs']} ({scratch['epochs']} from scratch)"),
                  ("Batch size", str(cls["batch_size"])),
                  ("Label smoothing", f"{cls['label_smoothing']:g}"),
                  ("Sampling", "class-balanced" if cls["balanced_sampler"] else "uniform"),
                  ("Model selection", "best val macro-F1")]
    detector = [("Model", "Faster R-CNN, ResNet-18 + FPN"),
                ("Input", f"short side {det['min_size']} px"),
                ("Optimizer", f"SGD, momentum {det['momentum']:g}"),
                ("Learning rate", f"{det['lr']:g}"),
                ("Weight decay", power10(det["weight_decay"])),
                ("Warm-up", f"{det['warmup_iters']} iterations"),
                ("Decay", rf"$\times${det['gamma']:g} at epochs " + ", ".join(str(m + 1) for m in det["milestones"])),
                ("Epochs", str(det["epochs"])),
                ("Batch size", str(det["batch_size"])),
                ("Frozen layers", FROZEN.get(det["trainable_layers"], str(det["trainable_layers"]))),
                ("Augmentation", "horizontal flip"),
                ("Model selection", r"best val AP$_{50}$")]

    def subtable(title, rows, note=None):
        lines = [r"\begin{subtable}[t]{0.49\linewidth}", r"\centering", rf"\caption{{{title}}}",
                 r"\begin{threeparttable}", r"\begin{tabular}{@{}ll@{}}", r"\toprule",
                 *(row(k, v) for k, v in rows), r"\bottomrule", r"\end{tabular}"]
        if note:
            lines += [r"\begin{tablenotes}[flushleft]\footnotesize", rf"\item[] {note}", r"\end{tablenotes}"]
        return lines + [r"\end{threeparttable}", r"\end{subtable}"]

    note = f"$^*$From scratch: {power10(scratch['lr_backbone'])} for all layers."
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Training setup. Both models use mixed precision on one RTX 4060 Laptop GPU (8 GB).}",
             r"\label{tab:setup}", r"\small",
             *subtable("Crop classifiers", classifier, note), r"\hfill",
             *subtable("Detectors", detector), r"\end{table}"]
    return "\n".join(lines) + "\n"


def classification_table(latest):
    metrics = [("val", "macro_f1"), ("test", "accuracy"), ("test", "macro_precision"),
               ("test", "macro_recall"), ("test", "macro_f1")]
    runs = [r[0] for r in CLASSIFIER_ROWS] + [r[0] for r in BASELINE_ROWS]
    matrix = [[value(latest, run, ev, m) for ev, m in metrics] for run in runs]
    cells = bold_best(matrix, skip=(len(runs) - 1,))
    body = [row(init, layers, aug, *cells[i]) for i, (_, init, layers, aug) in enumerate(CLASSIFIER_ROWS)]
    body.append(r"\midrule")
    body += [rf"\multicolumn{{3}}{{l}}{{{name}}} & " + " & ".join(cells[len(CLASSIFIER_ROWS) + k]) + r" \\"
             for k, (_, name) in enumerate(BASELINE_ROWS)]
    header = [r"\multicolumn{3}{c}{Classifier} & Val & \multicolumn{4}{c}{Test} \\",
              r"\cmidrule(lr){1-3}\cmidrule(lr){4-4}\cmidrule(lr){5-8}",
              row("Init.", "Trained layers", "Aug.", "F1", "Acc.", "Prec.", "Rec.", "F1")]
    return table("Classification of ground-truth crops (RQ1), in \\%. Precision, recall and F1 are macro-averaged "
                 "over the 10 classes.", "tab:classification", "llcccccc", header, body,
                 note="Validation crops exclude boxes with a short side under 32~px; test crops include them, "
                      "which explains most of the gap between the validation and test F1 (Table~\\ref{tab:size}).")


def size_table(latest):
    buckets = ["small", "medium", "large"]
    runs = [r[0] for r in CLASSIFIER_ROWS] + [BASELINE_ROWS[0][0]]
    matrix = [[value(latest, run, "test", f"accuracy_{b}") for b in buckets] for run in runs]
    cells = bold_best(matrix)
    body = [row(init, layers, aug, *cells[i]) for i, (_, init, layers, aug) in enumerate(CLASSIFIER_ROWS)]
    body += [r"\midrule", rf"\multicolumn{{3}}{{l}}{{{BASELINE_ROWS[0][1]}}} & " + " & ".join(cells[-1]) + r" \\"]
    header = [row("Init.", "Trained layers", "Aug.", r"Small ($<32^2$)", "Medium", r"Large ($\geq 96^2$)")]
    return table("Test accuracy by object size (COCO area buckets, pixels), in \\%.", "tab:size", "llcccc",
                 header, body,
                 note="Every small box has a short side under 32 px, so no small box was seen during training.")


def perclass_f1_table(latest):
    runs = [r[0] for r in CLASSIFIER_ROWS] + [BASELINE_ROWS[0][0]]
    matrix = [[value(latest, run, "test", f"f1_{c}") for run in runs] for c in CLASSES]
    cells = bold_best(matrix, axis="row")
    body = [row(c, *cells[i]) for i, c in enumerate(CLASSES)]
    return table("Per-class test F1 of each classifier, in \\%. Bold marks the best model for each class.",
                 "tab:perclass_f1", "lccccc", PERCLASS_HEADER, body)


def detection_table(latest):
    metrics = ["AP50", "AP", "AP75", "AP_small", "AP_medium", "AP_large", "AR100"]
    val_sets = ["val_agnostic", "val", None]  # the two-stage pipeline was only run on test
    matrix = [[value(latest, run, val_sets[i], "AP50") if val_sets[i] else None,
               *(value(latest, run, ev, m) for m in metrics)] for i, (run, ev, _) in enumerate(DETECTOR_ROWS)]
    cells = bold_best(matrix, skip=(0,))
    names = [DETECTOR_ROWS[0][2] + r"$^\dagger$"] + [name for _, _, name in DETECTOR_ROWS[1:]]
    body = [row(names[0], *cells[0]), r"\midrule"] + [row(names[i], *cells[i]) for i in (1, 2)]
    header = [r" & Val & \multicolumn{7}{c}{Test} \\",
              r"\cmidrule(lr){2-2}\cmidrule(lr){3-9}",
              row("Method", r"AP$_{50}$", r"AP$_{50}$", "AP", r"AP$_{75}$", r"AP$_S$", r"AP$_M$", r"AP$_L$",
                  r"AR$_{100}$")]
    return table("Detection on full images (RQ2, RQ3), COCO metrics in \\%.", "tab:detection", "lcccccccc",
                 header, body, small=True,
                 note=r"$^\dagger$Class-agnostic: a box counts if it covers any animal, so this row is not "
                      r"comparable with the 10-class rows below it.")


def multi_animal_table(latest):
    metrics = ["multi_recall50", "multi_count_accuracy", "multispecies_recall50", "multispecies_count_accuracy"]
    matrix = [[value(latest, run, ev, m) for m in metrics] for run, ev, _ in DETECTOR_ROWS]
    cells = bold_best(matrix)
    n_multi = value(latest, DETECTOR_ROWS[0][0], DETECTOR_ROWS[0][1], "multi_images")
    n_species = value(latest, DETECTOR_ROWS[0][0], DETECTOR_ROWS[0][1], "multispecies_images")
    fmt = lambda n: DASH if n is None else f"{int(n):,}"  # noqa: E731
    header = [rf" & \multicolumn{{2}}{{c}}{{$\geq$2 animals ({fmt(n_multi)} images)}} & "
              rf"\multicolumn{{2}}{{c}}{{$\geq$2 species ({fmt(n_species)} images)}} \\",
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
              row("Method", "Recall", "Count acc.", "Recall", "Count acc.")]
    body = [row(name, *cells[i]) for i, (_, _, name) in enumerate(DETECTOR_ROWS)]
    return table("Test images with several animals (RQ2), in \\%.", "tab:multi_animal", "lcccc", header, body,
                 note=r"Recall: share of animals matched by a box at IoU $\geq 0.5$, whatever its label. "
                      r"Count accuracy: share of images where the number of boxes scoring above 0.5 equals "
                      r"the number of animals.", small=True)


def two_stage_table(latest):
    body = []
    for c in CLASSES + [None]:
        metric = "AP50" if c is None else f"AP50_{c}"
        one, two = value(latest, "D2_10class", "test", metric), value(latest, PIPELINE, "test", metric)
        cells = bold_best([[one, two]], axis="row")[0]
        delta = DASH if one is None or two is None else signed(100 * (two - one))
        if c is None:
            body.append(r"\midrule")
        body.append(row("Mean (mAP$_{50}$)" if c is None else c, *cells, delta))
    return table(r"Per-class test AP$_{50}$ of the one-stage and two-stage methods (RQ3), in \%.",
                 "tab:two_stage", "lccc",
                 [row("Class", "One-stage", "Two-stage", r"$\Delta$ (points)")], body)


def robustness_table(latest):
    sets = ["test"] + [f"test_{c}" for c in CORRUPTIONS]
    cls_runs = [(run, f"{init}, {layers}, {'aug.' if aug == CHECK else 'no aug.'}")
                for run, init, layers, aug in CLASSIFIER_ROWS[:2]] + [(BASELINE_ROWS[0][0], "CLIP zero-shot")]
    matrix = [[value(latest, run, s, "macro_f1") for s in sets] for run, _ in cls_runs]
    cells = bold_best(matrix)
    body = [row(name, "F1", *cells[i]) for i, (_, name) in enumerate(cls_runs)]
    body += [r"\midrule", row("Detector + classifier", r"mAP$_{50}$",
                              *(pct(value(latest, PIPELINE, s, "AP50")) for s in sets))]
    det = [value(latest, "D1_agnostic", f"{s}_agnostic", "AP50") for s in sets]
    if any(v is not None for v in det[1:]):  # detector-only robustness, if it was evaluated
        body.append(row(r"Detector, animal vs.\ background", r"AP$_{50}$", *(pct(v) for v in det)))
    header = [r" & & & \multicolumn{2}{c}{Low light} & \multicolumn{2}{c}{Blur} & \multicolumn{2}{c}{Low contrast} \\",
              r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
              row("Model", "Metric", "Clean", "1", "2", "1", "2", "1", "2")]
    return table("Robustness to image corruptions at two severities (RQ4), test set, in \\%.", "tab:robustness",
                 "lcccccccc", header, body, small=True,
                 note="Low light: gamma 2 / 3 plus Gaussian noise ($\\sigma$ = 0.02 / 0.04). Blur: Gaussian, radius "
                      "1 / 2~px. Low contrast: contrast factor 0.5 / 0.3. Bold marks the best classifier per column.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=REPO / "results" / "results.csv")
    ap.add_argument("--configs", default=REPO / "configs")
    ap.add_argument("--stats", default=REPO / "data" / "stats.json")
    ap.add_argument("--out", default=REPO / "report" / "tables")
    args = ap.parse_args()

    latest = load_latest(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"dataset": dataset_table(Path(args.stats)),
              "setup": setup_table(Path(args.configs)),
              "classification": classification_table(latest),
              "size": size_table(latest),
              "perclass_f1": perclass_f1_table(latest),
              "detection": detection_table(latest),
              "multi_animal": multi_animal_table(latest),
              "two_stage": two_stage_table(latest),
              "robustness": robustness_table(latest)}
    for name, tex in tables.items():
        (out / f"{name}.tex").write_text("% generated by scripts/make_latex_tables.py from results/results.csv; "
                                         "do not edit by hand\n" + tex)
        print(f"wrote {out / (name + '.tex')}")


if __name__ == "__main__":
    main()
