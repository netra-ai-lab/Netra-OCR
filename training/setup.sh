#!/usr/bin/env bash
# Set up training, fine-tuning and evaluation: a virtual environment with the
# app (netra_ocr, on ONNX Runtime) and the training package (netra_ocr_train,
# on PyTorch). Safe to re-run.
#
#   bash training/setup.sh                 # .venv in the repository root
#   VENV=~/envs/netra bash training/setup.sh
#   PYTHON=python3.12 bash training/setup.sh
#
# Uses uv when it's installed (much faster), else venv + pip. The ONNX Runtime
# build follows the machine: GPU when nvidia-smi sees one, else CPU.

set -euo pipefail
[[ "${1:-}" == -h || "${1:-}" == --help ]] && { sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 0; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3}"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' ||
    { echo "Python 3.10 or newer is needed ($PYTHON is $("$PYTHON" -V 2>&1)). Set PYTHON=..." >&2; exit 1; }
  echo "Creating $VENV"
  if command -v uv >/dev/null; then uv venv --python "$PYTHON" "$VENV"; else "$PYTHON" -m venv "$VENV"; fi
fi

# Keep whichever ONNX Runtime the environment already has: the CPU and GPU
# packages conflict.
if "$VENV/bin/python" -c 'import onnxruntime' 2>/dev/null; then
  runtime=""
elif command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  runtime="[gpu]"
else
  runtime="[cpu]"
fi

if command -v uv >/dev/null; then
  install=(uv pip install --python "$VENV/bin/python")
else
  install=("$VENV/bin/python" -m pip install)
fi
"${install[@]}" -e "$ROOT$runtime" -e "$ROOT/training"

"$VENV/bin/python" - <<'PY'
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print(f"\nReady. PyTorch {torch.__version__}, " + (f"GPU: {gpu}" if gpu else "no GPU (training will be slow)"))
PY
cat <<MSG

Next (from the repository root):
  bash training/evaluate.sh published               # score the published model
  bash training/finetune.sh path/to/your_lines      # fine-tune it on your data
MSG
