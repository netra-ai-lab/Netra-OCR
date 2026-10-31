#!/usr/bin/env bash
# Block-size sweep for the frozen-base blockwise parallel decoder
# (Stern et al. 2018, Table 2): train one proposal head per block size on top
# of the SAME converged, frozen cluster-AR base, then benchmark them all
# against AR in a single comparison table.
#
# Each block size needs its own init + trained head -- the proposal FFN has
# `block_size - 1` output heads, so a head trained for one block size can't be
# reused for another. Each one gets its own run folder,
# training/runs/blockwise-bs<N>/.
#
# Usage (from the repository root, after training/setup.sh):
#   source .venv/bin/activate
#   bash training/netra_ocr_train/scripts/sweep_block_size.sh
#
# Override defaults via env vars, e.g.:
#   BLOCK_SIZES="6 8" EPOCHS=15 bash .../sweep_block_size.sh
set -euo pipefail

VOCAB="training/netra_ocr_train/char2idx_cluster.json"
AR_BASE="${AR_BASE:-training/weights/khmerocr_cluster_ar.pth}"   # converged, frozen cluster-AR base
BLOCK_SIZES="${BLOCK_SIZES:-6 8 10}"                              # bs=4 is the published one
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCH_SAMPLE_SIZE="${EPOCH_SAMPLE_SIZE:-50000}"

if [[ ! -f "$AR_BASE" ]]; then
  echo "ERROR: AR base checkpoint not found: $AR_BASE" >&2
  exit 1
fi

echo "=== Block-size sweep: [$BLOCK_SIZES] on base $AR_BASE ==="

BW_MODELS=(published-blockwise)
for BS in $BLOCK_SIZES; do
  RUN="training/runs/blockwise-bs${BS}"
  mkdir -p "$RUN"

  echo "--- [bs=$BS] init proposal head ---"
  python -m netra_ocr_train.scripts.init_blockwise_decoder \
    --src "$AR_BASE" --dst "$RUN/init.pth" --vocab "$VOCAB" --block-size "$BS"

  echo "--- [bs=$BS] train proposal head ($EPOCHS epochs) ---"
  python -m netra_ocr_train.scripts.train \
    --vocab "$VOCAB" --init "$RUN/init.pth" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --epoch-sample-size "$EPOCH_SAMPLE_SIZE" \
    --out-dir "$RUN"

  BW_MODELS+=("$RUN/best.pth")
  echo "--- [bs=$BS] done -> $RUN/best.pth ---"
done

# The text is identical to AR by construction; compare ms/line.
echo "=== Benchmarking AR + all block sizes ==="
python -m netra_ocr_train.scripts.evaluate "$AR_BASE" "${BW_MODELS[@]}" --out training/runs/eval-sweep
