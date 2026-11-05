# Figure specifications for the ECS 174 report

This file is a brief for a coding agent. It says what each figure must show, where its data comes from and
how to check it. Paste the prompt below into the agent together with the IDs of the figures you want.

## Prompt to paste

> You are working in the repository `~/ecs174-animal` (WSL, Python venv in `.venv`). Read
> `report/figure_specs.md` in full, then build figures **[list the IDs, e.g. F01, F06, F09]** exactly as
> specified there. Follow the "Rules for every figure" section. Put CPU-only figures in
> `scripts/make_report_figures.py` (one function per figure, `--only F01 F06` to build a subset) and any figure
> that needs the GPU in its own script. Do not retrain or re-evaluate models and do not edit
> `results/results.csv`. When done, render every figure, look at it, and report which checks in its
> "Acceptance checks" list pass.

## Project context

- Task: find and name animals in photos. Dataset: the COCO train2017 animal subset (10 classes), split by image
  into train / val / test (16,792 / 3,598 / 3,598 images, `splits/split_seed42.json`).
- Models (run ids in brackets; never show the run ids in a figure, use the display names below):

| Run id | Display name |
|---|---|
| `C1_finetune_aug` | ImageNet · all layers · aug |
| `C2_finetune_noaug` | ImageNet · all layers · no aug |
| `C3_linear_probe` | ImageNet · head only · aug |
| `C4_scratch` | Random init · all layers · aug |
| `clip_zeroshot` | CLIP zero-shot |
| `majority` | Majority class |
| `D1_agnostic` | Detector, animal vs. background |
| `D2_10class` | One-stage: detector, 10 classes |
| `D1_agnostic+C1_finetune_aug` | Two-stage: detector + classifier |

- Two-stage pipeline: the animal-vs-background detector proposes boxes, each box is expanded by 10% and cropped,
  the `ImageNet · all layers · aug` classifier names it, final score = detection score × classifier confidence.
- Class order (index 0–9, COCO `category_id` = index + 1): bird, cat, dog, horse, sheep, cow, elephant, bear,
  zebra, giraffe (`src/data/common.py: CLASSES`).

## Data you can use

| What | Where | Notes |
|---|---|---|
| All metrics | `results/results.csv` | Columns `run_id, model, train_config, eval_set, metric, value, date, git_commit`. Append-only: use the **newest** row per (run_id, eval_set, metric) via `scripts/summarize_results.py: load_latest(), value()`. |
| Eval sets | — | `val`, `val_agnostic`, `test`, `test_agnostic`, `test_{lowlight,blur,contrast}{1,2}` |
| Classifier metrics | — | `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `f1_<class>`, `accuracy_{small,medium,large}` |
| Detector metrics | — | `AP`, `AP50`, `AP75`, `AP_{small,medium,large}`, `AR1/10/100`, `AP50_<class>` (10-class runs), `multi_images`, `multi_recall50`, `multi_count_accuracy`, `multispecies_*` |
| Per-crop classifier predictions | `results/predictions/<run>_<eval_set>.csv` (local, not in git) | Columns `ann_id, image_id, path, label, pred, confidence, size_bucket`; `path` is relative to `data/`. Exists for the four ResNet classifiers (clean test and each corruption). **CLIP has no per-crop file.** |
| Detector predictions | `results/predictions/D1_agnostic_test_agnostic.json`, `results/predictions/D2_10class_test.json` | COCO result format: list of `{image_id, category_id, bbox: [x, y, w, h], score}`, score ≥ 0.05. D1 uses `category_id` 1 for every box. |
| Two-stage predictions | `results/predictions/pipeline_D1_agnostic+C1_finetune_aug_<eval_set>.json` | Same format, 10 classes; one file for clean test and each corruption. |
| Ground truth | `data/annotations/instances_{train,val,test}.json` | COCO format; images in `data/images/`. |
| Crops index | `data/crops/{train,val,test}.csv` | `path, label, class_name, ann_id, image_id, box_w, box_h, area, size_bucket`. Train/val drop boxes with a short side < 32 px; test keeps all. |
| Dataset counts | `data/stats.json` | Per split: `images`, `boxes`, `per_class`. |
| Training logs | `runs/<run>/log.csv` | Classifiers: `epoch, train_loss, val_accuracy, val_macro_f1, lr_head, epoch_seconds`. Detectors: `epoch, train_loss, val_AP50, val_AP, epoch_minutes`. |
| Code to reuse | `src/data/corruptions.py: apply_corruption(img, kind, severity, seed)`, `src/pipeline.py: load_models(), detect_and_classify()`, `src/det_eval.py: _iou()` | Scripts in `scripts/` must add the repo root to `sys.path` before importing `src.*`. |

## Rules for every figure

1. **Numbers only from data.** Every plotted value is read from the files above at run time. Never type a
   number into the code. If a file is missing, stop and report it instead of substituting anything.
2. **Names.** Use the display names in the table above. Never show `C1`–`C4`, `D1`, `D2` or other run ids.
3. **Style.** Reuse `scripts/make_figures.py`: its `style()` helper, `rcParams`, and colours. Categorical
   colours are always assigned in this order: `#2a78d6` (blue), `#eb6834` (orange), `#1baf7a` (aqua),
   `#eda100` (yellow), `#e87ba4` (magenta). Text `#0b0b0b` / secondary `#52514e`, grid `#e4e3df`. Within one
   figure an entity keeps one colour. No dual y-axes, no 3-D, no rainbow colour maps, no chart junk.
4. **Legends and labels.** A legend whenever there are ≥ 2 series; axis labels with units; percentages shown as
   % (0–100). Titles say what the figure shows, in sentence case.
5. **Size and output.** Full-width figures 6.5 in wide, half-width 3.2 in. Save to `results/figures/<file>` as
   both `.pdf` (for LaTeX) and `.png` at 200 dpi, white background.
6. **One home per number.** Figures marked "duplicates Table N" are for slides only; the report keeps the table.
7. **Verify.** Render each figure and look at it: no overlapping labels, nothing cut off, legend not covering
   data, values match `results/summary.md`.

## Figures

Status: **new** = to build; **exists** = already produced by `scripts/make_figures.py` (only restyle if asked).

### F01 Dataset overview (new, report: Dataset section)
- Purpose: show class imbalance, how many animals an image holds, and how many boxes are tiny.
- Data: `data/stats.json` (panel a); `data/annotations/instances_*.json`, all three splits (panels b, c).
- Layout: one row, three panels, 6.5 in wide.
  - (a) Horizontal bars: boxes per class, all splits summed, sorted largest first, one colour (blue), count at
    each bar tip.
  - (b) Bars: number of images by animals per image, bins 1, 2, 3, 4, 5, ≥ 6.
  - (c) Histogram of box short side `min(w, h)` in px, log-scaled x axis, a vertical line at 32 px labelled
    "dropped from train/val crops", and the share of boxes below 32 px written next to it.
- Acceptance checks: class totals equal the sum of the three splits in `stats.json`; image counts in (b) sum to
  23,988; the share in (c) is computed, not typed.
- File: `dataset_overview`.

### F02 Method diagram (new, report: Method section)
- Purpose: show how the two-stage and one-stage methods work.
- Content, two rows read left to right:
  - Two-stage: input image → "Faster R-CNN, ResNet-18 + FPN (animal vs. background)" → boxes → "expand each box
    by 10% and crop" → "ResNet-18 classifier (10 classes)" → labelled boxes, with "score = detection × class
    confidence" under the last arrow.
  - One-stage: input image → "Faster R-CNN, ResNet-18 + FPN (10 classes)" → labelled boxes.
  - Note under both rows: "backbones initialized from ImageNet; no COCO detection weights".
- Form: vector diagram (TikZ in `report/figures/method.tex`, or draw.io / SVG exported to PDF). Optional: one real
  test image with its boxes as the input and output thumbnails, drawn from the two-stage predictions file.
- Acceptance checks: every box label matches the text above; arrows run one way; readable at 6.5 in width.
- File: `method_diagram`.

### F03 Training curves (exists: `train_classifiers.png`, `train_detectors.png`; appendix or slides)
- Validation macro-F1 and training loss by epoch for the four classifiers; validation AP50 by epoch for both
  detectors with the two learning-rate drops marked. Only save an extra `.pdf` copy if the report uses them.

### F04 Classifier comparison bars (exists: `test_classifiers.png`; duplicates Table 2, slides only)

### F05 Accuracy versus object size, continuous (new, report or slides: replaces Table 3 if used)
- Purpose: show finer than three buckets how accuracy falls for small animals.
- Data: `results/predictions/<run>_test.csv` for the four ResNet classifiers, joined on `ann_id` with
  `data/crops/test.csv` to get `box_w`, `box_h`. CLIP is left out (no per-crop predictions).
- Computation: short side = `min(box_w, box_h)`; bins < 16, 16–32, 32–64, 64–128, 128–256, ≥ 256 px; accuracy per
  bin per model; number of crops per bin.
- Chart: line with markers, x = bin (categorical, in order), y = accuracy (%), one line per classifier in the
  colour order above; crop count per bin printed under each x label; a shaded band over the bins below 32 px
  labelled "never seen in training".
- Acceptance checks: bin counts sum to the number of rows in `test.csv`; the overall accuracy recomputed from the
  bins matches `accuracy` in `results.csv` for each model.
- File: `accuracy_vs_size`.

### F06 Most confused class pairs (new, report: Error Analysis)
- Purpose: which similar animals the main classifier mixes up.
- Data: `results/predictions/C1_finetune_aug_test.csv`.
- Computation: for each ordered pair (true, predicted) with true ≠ predicted: count and share of the true class
  (count / crops of that true class). Keep the 8 largest shares.
- Chart: horizontal bars, labels "cow → sheep", sorted, value at the tip as "12.3% (n = 45)", one colour.
- Also regenerate the row-normalized confusion matrix of the same classifier with display names (sequential
  single-hue blue scale, values ≥ 1% printed, true class on y, predicted on x) as a second file.
- Acceptance checks: counts match the confusion matrix; shares use the true-class totals.
- Files: `confused_pairs`, `confusion_main_classifier`.

### F07 Two-stage vs. one-stage per class (new; duplicates Table 6, slides only)
- Data: `results.csv`, metric `AP50_<class>`, eval set `test`, runs `D2_10class` and
  `D1_agnostic+C1_finetune_aug`.
- Chart: dumbbell plot, one row per class sorted by the gap (largest loss at top), one-stage dot blue,
  two-stage dot orange, grey connecting line, x = AP50 (%), vertical dashed lines at each method's mean
  (mAP50) labelled.
- File: `two_stage_per_class`.

### F08 Detection AP by object size (new; duplicates Table 4, slides only)
- Data: `AP_small`, `AP_medium`, `AP_large` for `D2_10class` (test) and the two-stage run (test). Leave the
  animal-vs-background detector out: its AP is class-agnostic and not comparable.
- Chart: grouped bars, x = size bucket, two bars per bucket, value labels on the bars.
- File: `detection_by_size`.

### F09 Detection versus number of animals in the image (new, report: RQ2)
- Purpose: does finding and counting get harder as the image holds more animals?
- Data: ground truth `instances_test.json`; predictions for the animal-vs-background detector, the 10-class
  detector and the two-stage pipeline (files above).
- Computation, per test image, grouped by its number of animals n ∈ {1, 2, 3, 4, ≥ 5}:
  - Recall: greedy matching per image (detections sorted by score, every saved detection, score ≥ 0.05), a
    ground-truth box is found if a detection has IoU ≥ 0.5 with it, class ignored. Same rule as
    `src/det_eval.py: multi_animal_stats()`.
  - Count error: (number of detections with score > 0.5) − n; report the share of images with count error < 0
    (too few), = 0 (exact), > 0 (too many).
- Chart: two panels, 6.5 in. (a) Recall vs. n, one line per method. (b) For the two-stage pipeline only, stacked
  100% bars per n: too few / exact / too many (three shades of one hue, legend).
- Acceptance checks: the recall over images with n ≥ 2 equals `multi_recall50` in `results.csv` for each method
  (to 0.1 points); the exact-count share over n ≥ 2 equals `multi_count_accuracy`.
- File: `detection_vs_animal_count`.

### F10 What the corruptions look like (new, report: RQ4 or slides)
- Data: one test image with one large, clearly visible animal (pick the largest-area single-animal image from
  `instances_test.json`, deterministic); `apply_corruption(img, kind, severity, seed=image_id)`, the same seed
  rule the evaluation uses.
- Chart: one row of 7 thumbnails, same size: clean, low light 1, low light 2, blur 1, blur 2, low contrast 1,
  low contrast 2, label under each. No axes.
- File: `corruption_examples`.

### F11 Robustness curves (exists: `robustness.png`; duplicates Table 7, slides only)

### F12 Detection examples (new, report: Results or Error Analysis; slides; video)
- Purpose: show successes and typical failures of the two-stage pipeline.
- Data: `results/predictions/pipeline_D1_agnostic+C1_finetune_aug_test.json`, ground truth, images. No GPU needed.
- Selection (by rule, then check by eye): per image match detections with score ≥ 0.5 to ground truth at IoU ≥
  0.5. Pick 3 images with ≥ 2 animals all found with the right class; 2 with a small animal (area < 32²) missed;
  2 with a found animal given the wrong class; 1 with an extra box on background or a duplicate.
- Chart: 2 × 4 grid. Ground truth as dashed green boxes, predictions (score ≥ 0.5) as solid boxes with a small
  label "class 0.87"; panel caption says which case it shows.
- Optional extra: photos the team took themselves (not downloaded from the web), run through
  `python -m src.pipeline --det runs/D1_agnostic/best.pt --cls runs/C1_finetune_aug/best.pt --image <photo>
  --out-image results/figures/demo_<name>.png` (needs the GPU).
- File: `detection_examples`.

### F13 Confident mistakes gallery (new, appendix or slides)
- Data: `results/predictions/C1_finetune_aug_test.csv`, crop images at `data/<path>`.
- Selection: wrong predictions sorted by confidence, highest first; take 18, at most 3 per (true, predicted) pair
  so one pair does not fill the grid.
- Chart: 3 × 6 grid of crops, caption under each "true → predicted (0.97)".
- Use: the team tags each crop with a cause (tiny or blurry, occluded or cut off, two animals in the crop, toy /
  statue / drawing, annotation error, genuinely similar) for the Error Analysis text.
- File: `confident_mistakes`.

### F14 Grad-CAM heatmaps (new, optional, GPU, slides)
- Data: `runs/C1_finetune_aug/best.pt`, a few test crops: 2 correct and 2 wrong for cow and for sheep.
- Method: Grad-CAM on the last block (`layer4`) of the ResNet-18, for the predicted class; overlay on the crop
  with alpha 0.45 using a perceptually uniform colour map (e.g. `magma`).
- Chart: 2 rows (original, overlay) × 8 crops, caption "true → predicted".
- File: `gradcam_examples`.

### F15 Feature space, t-SNE (new, optional, GPU, appendix)
- Data: penultimate-layer features (512-d, `fc` replaced by identity) of `C1_finetune_aug` and `C4_scratch` on
  a class-balanced sample of 2,000 test crops (fixed seed).
- Method: `sklearn.manifold.TSNE(perplexity=30, init="pca", random_state=0)` per model.
- Chart: two scatter panels side by side, one point per crop, coloured by class with a 10-colour qualitative map,
  and each class name written at its cluster centre so identity does not rely on colour alone.
- File: `tsne_features`.

### F16 Where detections go wrong (new, report: Error Analysis, RQ3)
- Purpose: split each method's outcome per animal, to show the two-stage gap comes from wrong classes.
- Data: ground truth and the predictions of the 10-class detector and the two-stage pipeline (clean test).
- Computation, detections with score ≥ 0.5, per image, greedy by score: each ground-truth animal is
  **correct** (matched at IoU ≥ 0.5 with the right class), **wrong class** (matched at IoU ≥ 0.5, other class),
  **poorly localized** (best IoU in [0.1, 0.5)) or **missed** (best IoU < 0.1). Also count unmatched detections
  as **extra boxes** (background or duplicate).
- Chart: horizontal 100% stacked bars, one bar per method, segments in the order above (correct = blue, then
  three shades for the error types); number of extra boxes written at the right of each bar.
- Acceptance checks: the four shares add to 100%; the number of ground-truth animals equals the box count of
  `instances_test.json`.
- File: `detection_error_breakdown`.
