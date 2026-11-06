
# ECS 174 Final Project: Animal Detection and Identification

Find every animal in a photo and name it with an ImageNet-pretrained ResNet-18.
Data: the animal subset of COCO train2017
([nirmalendu01/animal_dataset](https://huggingface.co/datasets/nirmalendu01/animal_dataset)), 10 classes, ~24k images.

Main method (detector + classifier): a Faster R-CNN detector (ResNet-18 + FPN) finds every animal, each box is cropped, and a ResNet-18 classifier names it. It is compared with a single Faster R-CNN that predicts the 10 classes directly, and with CLIP zero-shot and majority-class baselines. All backbones start from ImageNet weights.

| Model                                                               | Config                                 |
| ------------------------------------------------------------------- | -------------------------------------- |
| Classifier: ImageNet · all layers · aug (used in the main method) | `configs/cls_C1_finetune_aug.yaml`   |
| Classifier: ImageNet · all layers · no aug                        | `configs/cls_C2_finetune_noaug.yaml` |
| Classifier: ImageNet · head only · aug                            | `configs/cls_C3_linear_probe.yaml`   |
| Classifier: random init · all layers · aug                        | `configs/cls_C4_scratch.yaml`        |
| Detector: animal vs. background (used in the main method)           | `configs/det_D1_agnostic.yaml`       |
| Detector: 10 classes (comparison)                                   | `configs/det_D2_10class.yaml`        |

Team: Jiacheng Chen, Justin Chan, Fanyou Liu, Becky Young.

## Setup (WSL2 + NVIDIA GPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # match your CUDA, see pytorch.org
pip install -r requirements.txt
python scripts/check_env.py
```

Keep the repo and `data/` in the WSL filesystem (`~/...`), not under `/mnt/c/`.

## Reproduce

```bash
python -m src.data.prepare_dataset          # download images, COCO annotations for the split in splits/
python -m src.data.make_crops --data data   # classifier crops
bash scripts/run_all.sh                     # train all six models -> runs/<name>/best.pt
bash scripts/run_test.sh                    # evaluate on the test set -> results/results.csv
```

Train one model with `python -m src.train_classifier --config configs/<name>.yaml` (`src.train_detector` for
detectors); `--resume` continues a run and `--set key=value` overrides a config value


## Layout

```
configs/    one YAML per model
splits/     the fixed train/val/test split
src/        data preparation, models, training, evaluation, pipeline, baselines
scripts/    check_env.py, run_all.sh (train), run_test.sh (evaluate)
results/    results.csv with every metric; figures/ and predictions/ stay local
report/     LaTeX source of the report
```
