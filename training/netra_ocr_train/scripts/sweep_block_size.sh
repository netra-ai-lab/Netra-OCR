#!/usr/bin/env bash
# Block-size sweep for the frozen-base blockwise parallel decoder
# (Stern et al. 2018, Table 2): train one proposal head per block size on top
# of the SAME converged, frozen cluster-AR base, then benchmark them all
# against AR in a single comparison table.
#
# Each block size needs its own init + trained head -- the proposal FFN has
# `block_size - 1` output heads, so a head trained for one block size can't be
# reused for another. train.py always writes khmerocr_se_blockwise_last.pth, so
# we rename after each run to keep them side by side.
#
# Usage:
#   cd /run/media/pc/disk1/Netra-OCR && source .venv/bin/activate
#   bash training/netra_ocr_train/scripts/sweep_block_size.sh
#
# Override defaults via env vars, e.g.:
#   BLOCK_SIZES="6 8" EPOCHS=15 bash .../sweep_block_size.sh
set -euo pipefail

WEIGHT_DIR="training/netra_ocr_train/weight"
VOCAB="training/netra_ocr_train/char2idx_cluster.json"
AR_BASE="${AR_BASE:-$WEIGHT_DIR/khmerocr_se_ar_last.pth}"   # converged, frozen cluster-AR base
BLOCK_SIZES="${BLOCK_SIZES:-6 8 10}"                        # bs=4 already trained
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCH_SAMPLE_SIZE="${EPOCH_SAMPLE_SIZE:-50000}"

if [[ ! -f "$AR_BASE" ]]; then
  echo "ERROR: AR base checkpoint not found: $AR_BASE" >&2
  exit 1
fi

echo "=== Block-size sweep: [$BLOCK_SIZES] on base $AR_BASE ==="

for BS in $BLOCK_SIZES; do
  INIT="$WEIGHT_DIR/khmerocr_cluster_blockwise_bs${BS}_init.pth"
  FINAL="$WEIGHT_DIR/khmerocr_blockwise_bs${BS}_last.pth"

  echo "--- [bs=$BS] init proposal head ---"
  python -m netra_ocr_train.scripts.init_blockwise_decoder \
    --src "$AR_BASE" --dst "$INIT" --vocab "$VOCAB" --block-size "$BS"

  echo "--- [bs=$BS] train proposal head ($EPOCHS epochs) ---"
  python -m netra_ocr_train.scripts.train \
    --vocab "$VOCAB" --resume "$INIT" \
    --decoder-type blockwise --block-size "$BS" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --epoch-sample-size "$EPOCH_SAMPLE_SIZE" \
    --log-file "$WEIGHT_DIR/train_blockwise_bs${BS}.log" \
    --out-dir "$WEIGHT_DIR"

  # train.py writes a fixed filename; rename so the next block size doesn't clobber it.
  mv "$WEIGHT_DIR/khmerocr_se_blockwise_last.pth" "$FINAL"
  echo "--- [bs=$BS] done -> $FINAL ---"
done

# Build the list of all trained blockwise checkpoints (include bs=4 if present).
BW_MODELS=()
[[ -f "$WEIGHT_DIR/khmerocr_se_blockwise_last.pth" ]] && BW_MODELS+=("$WEIGHT_DIR/khmerocr_se_blockwise_last.pth")
for BS in $BLOCK_SIZES; do
  BW_MODELS+=("$WEIGHT_DIR/khmerocr_blockwise_bs${BS}_last.pth")
done

echo "=== Benchmarking AR + all block sizes ==="
python -m netra_ocr_train.scripts.evaluate_benchmark \
  --local-dir training/test_set/khob_eval \
  --ar-beam-width 1 \
  --blockwise-model "${BW_MODELS[@]}" \
  --out-dir eval_out_sweep
