# Sourced by the training scripts: use the project's virtual environment
# (made by setup.sh) unless one is already active.
TRAINING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$TRAINING_DIR/../.venv}"
if [[ -z "${VIRTUAL_ENV:-}" && -x "$VENV/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
if ! python -c "import netra_ocr, netra_ocr_train" 2>/dev/null; then
  echo "The training environment isn't set up. Run: bash training/setup.sh" >&2
  exit 1
fi

# Print the comment block at the top of the calling script (its usage).
usage() { sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; }
