# ECS 174 Final Project: Find, Crop, Classify

Two-stage animal detection and identification on the COCO animal subset
([nirmalendu01/animal_dataset](https://huggingface.co/datasets/nirmalendu01/animal_dataset), 10 classes, ~24k images).

1. **Detector (D1)**: Faster R-CNN with an ImageNet-pretrained ResNet-18 + FPN backbone, class-agnostic ("animal" vs background).
2. **Classifier (C1)**: ImageNet-pretrained ResNet-18 fine-tuned on single-animal crops (10 classes).
3. **Pipeline**: D1 boxes every animal → each box is cropped → C1 names it.

Baselines: C2 (no augmentation), C3 (linear probe), C4 (random init), CLIP zero-shot, D2 (one detector that predicts 10 classes).

Team: Jiacheng Chen, Justin Chan, Fanyou Liu, Becky Young.

## Setup (WSL2 + NVIDIA GPU)

```bash
cd ~/ecs174-animal
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# 1) PyTorch with CUDA: pick the command for your CUDA version at https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 2) everything else
pip install -r requirements.txt
# 3) check GPU and both models
python scripts/check_env.py
```

Keep the repo and `data/` inside the WSL filesystem (`~/...`), not under `/mnt/c/`; reading images from the Windows drive is much slower.

## Smoke test (about 5 minutes, 300 images)

Run this first to make sure every step works before the full runs.

```bash
python -m src.data.prepare_dataset --out data_debug --limit 300
python -m src.data.make_crops --data data_debug
python -m src.train_classifier --config configs/cls_C1_finetune_aug.yaml \
    --set data_root=data_debug epochs=1 run_name=debug_C1 batch_size=32
python -m src.train_detector --config configs/det_D1_agnostic.yaml \
    --set data_root=data_debug epochs=1 run_name=debug_D1 limit_images=100
```

Delete the debug rows from `results/results.csv` afterwards.

## Full run order

```bash
# data (once). splits/split_seed42.json is created on the first run and committed;
# later runs reuse it so everyone has the same split.
python -m src.data.prepare_dataset
python -m src.data.make_crops --data data

# training process 1: classifiers (~15-30 min each)
python -m src.train_classifier --config configs/cls_C1_finetune_aug.yaml
python -m src.train_classifier --config configs/cls_C2_finetune_noaug.yaml
python -m src.train_classifier --config configs/cls_C3_linear_probe.yaml
python -m src.train_classifier --config configs/cls_C4_scratch.yaml

# training process 2: detectors (~3-5 h each; add --resume to continue after an interruption)
python -m src.train_detector --config configs/det_D1_agnostic.yaml
python -m src.train_detector --config configs/det_D2_10class.yaml

# evaluation (test set, run once at the end)
python -m src.eval_classifier --ckpt runs/C1_finetune_aug/best.pt --split test     # repeat for C2-C4
python -m src.clip_zeroshot --split test
python -m src.eval_detector --ckpt runs/D1_agnostic/best.pt --split test
python -m src.eval_detector --ckpt runs/D2_10class/best.pt --split test
python -m src.pipeline --det runs/D1_agnostic/best.pt --cls runs/C1_finetune_aug/best.pt --split test

# robustness (E5): corruption in {lowlight, blur, contrast}, severity in {1, 2}
python -m src.eval_classifier --ckpt runs/C1_finetune_aug/best.pt --corruption lowlight --severity 2
python -m src.clip_zeroshot --corruption lowlight --severity 2
python -m src.pipeline --det runs/D1_agnostic/best.pt --cls runs/C1_finetune_aug/best.pt --corruption lowlight --severity 2

# demo on any photo
python -m src.pipeline --det runs/D1_agnostic/best.pt --cls runs/C1_finetune_aug/best.pt \
    --image path/to/photo.jpg --out-image results/figures/demo.png
```

Any config value can be overridden from the command line with `--set key=value`.
If the detector runs out of GPU memory, use `--set min_size=512 max_size=853`.

## Rules that keep results valid

- Split by **image**, never by crop. Everyone uses `splits/split_seed42.json`.
- **Never** use COCO-pretrained detection weights: the test images come from COCO train2017. Only ImageNet backbone weights.
- Tune on `val`; run `test` once at the end.
- Every number in the report comes from `results/results.csv`.

## Layout

```
configs/            one YAML per experiment (C1-C4, D1, D2)
splits/             the fixed train/val/test split (committed)
src/data/           prepare_dataset, make_crops, datasets, transforms, corruptions
src/models/         classifier.py (ResNet-18), detector.py (Faster R-CNN R18-FPN)
src/                train_classifier, train_detector, eval_*, pipeline, clip_zeroshot
scripts/            check_env.py
notebooks/          EDA, error analysis, demo (see notebooks/README.md)
results/            results.csv (tracked), predictions/ (ignored), figures/
runs/, data/        created locally, not tracked
```
