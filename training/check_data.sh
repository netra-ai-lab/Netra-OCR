#!/usr/bin/env bash
# Check a dataset before training on it.
#
#   bash training/check_data.sh DATA [DATA ...]
#
# DATA is any of
#   my_lines/                    images/ + labels/ (0001.png + 0001.txt), or images with a .txt beside each
#   my_lines/labels.tsv          a manifest: image path <TAB> text (also .csv and .jsonl)
#   user/name[:split[:column]]   a Hugging Face dataset, e.g. user/name:train[:1000]
#   khob, printed-words, synthetic   a preset
#
# For each one it prints the format it found, the number of lines, image
# sizes, text lengths, unreadable images, empty labels, characters the model
# can't learn, and a few examples. Exits with 1 when something must be fixed.
#
# Examples:
#   bash training/check_data.sh training/examples/lines
#   bash training/check_data.sh ~/data/receipts/labels.tsv

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
[[ $# -eq 0 || "$1" == -h || "$1" == --help ]] && { usage; exit 0; }

netra-train check "$@"
