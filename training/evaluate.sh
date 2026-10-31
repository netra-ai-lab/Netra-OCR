#!/usr/bin/env bash
# Score recognition models: CER, WER, exact match and ms per line.
#
#   bash training/evaluate.sh MODEL [MODEL ...] [--test SET [SET ...]] [options]
#
# MODEL is one of
#   published                  the model the app ships, run exactly as the app runs it
#   training/runs/NAME/best.pth  a checkpoint from finetune.sh / train.sh
#   training/runs/NAME/bundles/recognition   a folder from export.sh
#
# SET is a folder (images/ + labels/, like the fine-tuning data) or a Hugging
# Face dataset id. Default: training/test_set/khob_eval if you have it, else
# the public SoyVitou/KHOB-LEVEL-1.
#
# Writes each model's predictions (CSV) and summary.json to
# training/runs/eval-<date-time>/ (or --out DIR). Other options: --limit N,
# --beam N (checkpoints only), --device cpu|cuda.
#
# Examples:
#   bash training/evaluate.sh published
#   bash training/evaluate.sh published training/runs/receipts/best.pth --test ~/data/receipts_test

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
[[ $# -eq 0 || "$1" == -h || "$1" == --help ]] && { usage; exit 0; }

netra-train evaluate "$@"
