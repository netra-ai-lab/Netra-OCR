#!/usr/bin/env bash
# Train the recognizer from scratch (or continue a run).
#
#   bash training/train.sh [--name NAME] [train options]
#   bash training/train.sh --resume training/runs/NAME/last.pth --name NAME   # continue
#
# Default data: the "synthetic" preset, the four public Hugging Face datasets
# the published model was trained on (~1.3M lines, ~16 GB, downloaded once into
# training/.hf_cache). Note: SoyVitou/KhmerSynthetic1M is licensed for research
# only; for a model you'll use commercially, pass your own --data instead.
# --data takes the same forms as finetune.sh (folders, manifests, Hugging Face
# ids, presets); check them first with training/check_data.sh.
#
# Defaults: 100 epochs of 50,000 random lines each, learning rate 1e-4 with 5
# warm-up epochs, batch 32. A progress bar follows each epoch, and a summary
# line after it gives the validation scores and the time left. The run goes to
# training/runs/NAME/ (best.pth, last.pth, metrics.csv, train.log).
# See: netra-train train --help
#
# Examples:
#   bash training/train.sh --name scratch
#   bash training/train.sh --name mine --data ~/data/lines_a ~/data/lines_b.tsv --val-data khob

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
[[ "${1:-}" == -h || "${1:-}" == --help ]] && { usage; exit 0; }

name="train-$(date +%Y%m%d-%H%M%S)" extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    *) extra+=("$1"); shift ;;
  esac
done
run="$TRAINING_DIR/runs/$name"

netra-train train --out-dir "$run" --epochs 100 --epoch-sample-size 50000 --batch-size 32 "${extra[@]}"

echo
echo "Model: $run/best.pth. Score it with: bash training/evaluate.sh $run/best.pth"
