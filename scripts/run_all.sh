#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
mkdir -p logs

run() {
  name=$(basename "$2" .yaml)
  echo "=== $(date '+%m-%d %H:%M') start $name ===" | tee -a logs/run_all.log
  python -m "$1" --config "$2" 2>&1 | tee "logs/$name.log"
  echo "=== $(date '+%m-%d %H:%M') end $name (exit ${PIPESTATUS[0]}) ===" | tee -a logs/run_all.log
}

for cfg in configs/cls_C*.yaml; do run src.train_classifier "$cfg"; done
for cfg in configs/det_D*.yaml; do run src.train_detector "$cfg"; done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ===" | tee -a logs/run_all.log
