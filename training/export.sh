#!/usr/bin/env bash
# Export a checkpoint as the ONNX model folder the app runs (no PyTorch needed).
#
#   bash training/export.sh CHECKPOINT [OUT]
#
# Writes OUT/recognition/ (default OUT: a "bundles" folder next to the
# checkpoint), then checks on a test set that the folder reads every line the
# same as the PyTorch checkpoint (with a progress bar). The test set is the
# khob preset, or CHECK=DATA (any dataset form); CHECK= skips the check.
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
check="${CHECK-khob}"

args=(--model "$checkpoint" --out "$out/recognition" --name netra-ocr-recognition-finetuned)
if [[ -n "$check" ]]; then
  args+=(--check "$check")
else
  echo "Skipping the parity check (CHECK is empty)."
fi
netra-train export "${args[@]}"

cat <<MSG

Model folder: $out/recognition
Run the app with it:   NETRA_BUNDLES=$out netra_ocr serve
Score it:              bash training/evaluate.sh $out/recognition
MSG
