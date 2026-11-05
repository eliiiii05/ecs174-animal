#!/usr/bin/env bash
# Run every test-set evaluation once (E1-E5), then build the summary tables and figures.
#
#   bash scripts/run_test.sh                    # all groups: cls det pipe robust report
#   bash scripts/run_test.sh robust report      # only some groups (e.g. after fixing a failure)
#   SYNC_DIR=/mnt/c/Users/<you>/Documents/x bash scripts/run_test.sh   # also copy logs/results there
#
# Each step's output goes to logs/test/<step>.log. A failed step is recorded with its
# exit code and the next step still runs. Test results are appended to results/results.csv.
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
mkdir -p logs/test
LOG=logs/test/run_test.log
GROUPS_TO_RUN=${*:-cls det pipe robust report}

DET=runs/D1_agnostic/best.pt
CLS=runs/C1_finetune_aug/best.pt

mirror() {  # copy progress to SYNC_DIR (if set) so it can be read from Windows
  [ -z "${SYNC_DIR:-}" ] && return
  mkdir -p "$SYNC_DIR/test_logs"
  cp results/results.csv "$SYNC_DIR/" 2>/dev/null
  cp logs/test/*.log "$SYNC_DIR/test_logs/" 2>/dev/null
}

step() {  # $1 = step name, rest = command
  local name=$1; shift
  echo "=== $(date '+%m-%d %H:%M:%S') start $name ===" | tee -a "$LOG"
  "$@" > "logs/test/$name.log" 2>&1
  local code=$?
  echo "=== $(date '+%m-%d %H:%M:%S') end $name (exit $code) ===" | tee -a "$LOG"
  mirror
}

wants() { [[ " $GROUPS_TO_RUN " == *" $1 "* ]]; }

echo "=== $(date '+%m-%d %H:%M:%S') run_test.sh groups: $GROUPS_TO_RUN ===" | tee -a "$LOG"

if wants cls; then  # E1 / RQ1: classifiers on ground-truth crops
  for c in C1_finetune_aug C2_finetune_noaug C3_linear_probe C4_scratch; do
    step "cls_${c%%_*}" python -m src.eval_classifier --ckpt runs/$c/best.pt --split test
  done
  step majority python -m src.majority_baseline --split test
  step clip python -m src.clip_zeroshot --split test
fi

if wants det; then  # E2, E4 / RQ2: detectors on full images
  step det_D1 python -m src.eval_detector --ckpt runs/D1_agnostic/best.pt --split test
  step det_D2 python -m src.eval_detector --ckpt runs/D2_10class/best.pt --split test
fi

if wants pipe; then  # E3 / RQ3: two-stage D1 + C1 vs one-stage D2
  step pipeline python -m src.pipeline --det $DET --cls $CLS --split test
fi

if wants robust; then  # E5 / RQ4: low light, blur, low contrast at two severities
  for k in lowlight blur contrast; do
    for s in 1 2; do
      step "robust_C1_$k$s" python -m src.eval_classifier --ckpt $CLS --corruption $k --severity $s
      step "robust_C2_$k$s" python -m src.eval_classifier --ckpt runs/C2_finetune_noaug/best.pt --corruption $k --severity $s
      step "robust_clip_$k$s" python -m src.clip_zeroshot --corruption $k --severity $s
      step "robust_pipeline_$k$s" python -m src.pipeline --det $DET --cls $CLS --corruption $k --severity $s
    done
  done
fi

if wants report; then
  step summary python scripts/summarize_results.py
  step figures python scripts/make_figures.py
  step latex_tables python scripts/make_latex_tables.py
  if [ -n "${SYNC_DIR:-}" ]; then
    mkdir -p "$SYNC_DIR/figures"
    cp results/summary.md "$SYNC_DIR/" 2>/dev/null
    cp results/figures/*.png "$SYNC_DIR/figures/" 2>/dev/null
  fi
fi

echo "=== $(date '+%m-%d %H:%M:%S') ALL DONE ===" | tee -a "$LOG"
mirror
