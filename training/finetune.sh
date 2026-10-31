#!/usr/bin/env bash
# Fine-tune the published recognizer on your own text lines.
#
#   bash training/finetune.sh DATA [DATA ...] [--name NAME] [train options]
#
# DATA is your own lines, in any of these forms (or several at once):
#   my_lines/                    my_lines/images/0001.png + my_lines/labels/0001.txt (the line's text)
#                                or my_lines/0001.png + my_lines/0001.txt (or .gt.txt) side by side
#   my_lines/labels.tsv          a manifest: image path <TAB> text per line (also .csv, .jsonl)
#   user/name[:split[:column]]   a Hugging Face dataset, e.g. user/name:train[:5000]
#   khob, printed-words, synthetic   a preset
# Check it first with: bash training/check_data.sh DATA
#
# 5% of the lines are held out to pick the best epoch (or pass --val-data DATA).
# A progress bar follows each epoch; a summary line after it gives the
# validation scores and the time left. The run goes to training/runs/NAME/
# (default: finetune-<date-time>):
#     best.pth  last.pth  metrics.csv  train.log
#
# Defaults: up to 10 epochs, stopping after 3 without improvement; learning
# rate 3e-5 with 1 warm-up epoch; batch 32; augmentation on. Any train option
# overrides them, e.g. --epochs 20, --batch-size 16 (less GPU memory),
# --patience 0 (never stop early). See: netra-train train --help
#
# Examples:
#   bash training/finetune.sh training/examples/lines --name try-it
#   bash training/finetune.sh ~/data/receipts --name receipts
#   bash training/finetune.sh ~/data/receipts/labels.tsv --val-data ~/data/receipts_test --epochs 20

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
  --epochs 10 --patience 3 --lr 3e-5 --min-lr 1e-6 --warmup-epochs 1 --batch-size 32 "${extra[@]}"

cat <<MSG

Fine-tuned model: $run/best.pth  (per-epoch scores in $run/metrics.csv)

Compare it with the published model (add --test YOUR_TEST_LINES for your own test set):
  bash training/evaluate.sh published $run/best.pth
Then export it for the app:
  bash training/export.sh $run/best.pth
MSG
