#!/usr/bin/env bash
# Export a checkpoint as the ONNX model folder the app runs (no PyTorch needed).
#
#   bash training/export.sh CHECKPOINT [OUT]
#
# Writes OUT/recognition/ (default OUT: a "bundles" folder next to the
# checkpoint), then checks on a test set that the folder reads every line the
# same as the PyTorch checkpoint. The test set is training/test_set/khob_eval,
# or CHECK=DIR (a folder with images/ + labels/); CHECK= skips the check.
#
# Use it in the app (the line detector and layout model still come from
# Hugging Face):
#   NETRA_BUNDLES=OUT netra_ocr serve
#   NETRA_BUNDLES=OUT netra_ocr --image page.jpg --output page.docx --layout
#   python -c 'from netra_ocr.recognition import recognize; print(recognize("line.png", bundle="OUT/recognition"))'

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
[[ $# -eq 0 || "$1" == -h || "$1" == --help ]] && { usage; exit 0; }

checkpoint="$1"
out="${2:-$(dirname "$checkpoint")/bundles}"
check="${CHECK-$TRAINING_DIR/test_set/khob_eval}"

args=(--model "$checkpoint" --out "$out/recognition" --name netra-ocr-recognition-finetuned)
if [[ -n "$check" && -d "$check" ]]; then
  args+=(--check "$check")
else
  echo "Skipping the parity check (no test set at '${check}'; set CHECK=DIR)."
fi
netra-train export "${args[@]}"

cat <<MSG

Model folder: $out/recognition
Run the app with it:   NETRA_BUNDLES=$out netra_ocr serve
Score it:              bash training/evaluate.sh $out/recognition
MSG
