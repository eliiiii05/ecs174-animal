#!/usr/bin/env bash
# Read every downloaded parquet shard of the dataset end to end and report any
# file that cannot be read (Input/output error = damaged file / disk block).
#   bash scripts/check_shards.sh
set -u
SNAP=$(ls -d ~/.cache/huggingface/hub/datasets--nirmalendu01--animal_dataset/snapshots/*/images 2>/dev/null | head -1)
if [ -z "$SNAP" ]; then
  echo "No downloaded shards found in ~/.cache/huggingface/hub"
  exit 1
fi
echo "Checking shards in $SNAP"
bad=0
for f in "$SNAP"/*.parquet; do
  if cat "$f" > /dev/null 2>/tmp/shard_err; then
    printf "ok    %s  %s\n" "$(du -h "$(readlink -f "$f")" | cut -f1)" "$(basename "$f")"
  else
    printf "BAD   %s  (%s)\n" "$(basename "$f")" "$(tr -d '\n' < /tmp/shard_err)"
    echo "      delete with: rm -f \"$(readlink -f "$f")\" \"$f\""
    bad=$((bad + 1))
  fi
done
echo "$(ls "$SNAP"/*.parquet | wc -l) shards found, $bad unreadable"
