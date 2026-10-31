#!/usr/bin/env bash
# Fine-tune the published recognizer on your own text lines.
#
#   bash training/finetune.sh DATA [DATA ...] [--name NAME] [train options]
#
# DATA is a folder of line images and their text:
#     my_lines/images/0001.png     my_lines/labels/0001.txt   (the line's text, UTF-8)
# or a Hugging Face dataset id, user/name[:split[:text_column]].
#
# 5% of the lines are held out to pick the best epoch (or pass --val-data DIR).
# The run goes to training/runs/NAME/ (default: finetune-<date-time>):
#     best.pth  last.pth  metrics.csv  train.log
#
# Defaults: 10 epochs, learning rate 3e-5 with 1 warm-up epoch, batch 32,
# augmentation on. Any train option overrides them, e.g. --epochs 20 or
# --batch-size 16 (less GPU memory). See: netra-train train --help
#
# Examples:
#   bash training/finetune.sh ~/data/receipts --name receipts
#   bash training/finetune.sh ~/data/receipts --val-data ~/data/receipts_test --epochs 20

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
[[ $# -eq 0 || "$1" == -h || "$1" == --help ]] && { usage; exit 0; }

data=() name="finetune-$(date +%Y%m%d-%H%M%S)" extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --*) extra=("$@"); break ;;
    *) data+=("$1"); shift ;;
  esac
done
[[ ${#data[@]} -gt 0 ]] || { echo "Give at least one DATA folder or dataset id" >&2; exit 1; }
run="$TRAINING_DIR/runs/$name"

netra-train train --init published --data "${data[@]}" --out-dir "$run" \
  --epochs 10 --lr 3e-5 --min-lr 1e-6 --warmup-epochs 1 --batch-size 32 "${extra[@]}"

cat <<MSG

Fine-tuned model: $run/best.pth  (per-epoch scores in $run/metrics.csv)

Compare it with the published model on a test set:
  bash training/evaluate.sh published $run/best.pth --test YOUR_TEST_LINES
Then export it for the app:
  bash training/export.sh $run/best.pth
MSG
