"""One-time bootstrap: wraps an already-trained, fully-autoregressive
checkpoint with Stern et al. 2018's frozen-base blockwise parallel decoding
head (see research/Blockwise Parallel Decoding for Deep Autoregressive
Models.pdf and model/blockwise_decoder.py).

Unlike migrate_checkpoint.py, this never touches the vocabulary: the base
AR model's tok_emb/out_proj rows were all trained on real data already
(unlike the cluster vocab's ~1900 cold-started rows), and since the base is
frozen forever after this, any never-fine-tuned embedding row would stay
broken permanently. So this script keeps the exact vocab the source
checkpoint was trained with (the 176-token char2idx_new.json by default)
and only adds a small new trainable proposal head on top -- everything else
(CNN/patch/encoder/BiLSTM/decoder attention stack/tok_emb/out_proj) is
copied verbatim from the source checkpoint and frozen.

Usage:
    python -m netra_ocr_train.scripts.init_blockwise_decoder \\
        --src training/weights/khmerocr_epoch570.pth \\
        --dst training/weights/khmerocr_blockwise_init.pth \\
        --vocab training/netra_ocr_train/char2idx_new.json \\
        --block-size 4
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from ..model.blockwise_decoder import BlockwiseParallelWrapper
from ..utils import autodetect_config

logger = logging.getLogger(__name__)


def _model_class_for(path: Path):
    from netra_ocr_train.model.model import KhmerOCR as SE_KhmerOCR
    return SE_KhmerOCR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Source, already-trained AR .pth checkpoint")
    parser.add_argument("--dst", type=Path, required=True, help="Output .pth path")
    parser.add_argument("--vocab", type=Path, required=True,
                         help="The vocab --src was trained with (not migrated)")
    parser.add_argument("--block-size", type=int, default=4)
    args = parser.parse_args()

    with open(args.vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    detected = autodetect_config(args.src)
    emb_dim = detected.get("emb_dim", 384)
    max_seq_len = detected.get("max_seq_len", 4096)

    model_class = _model_class_for(args.src)
    model = model_class(
        vocab_size=len(vocab),
        pad_idx=vocab.get("<pad>", 0),
        emb_dim=emb_dim,
        max_global_len=max_seq_len,
        decoder_type="ar",
    )

    checkpoint = torch.load(args.src, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded AR weights from {args.src}")

    # Freeze everything from the source checkpoint before wrapping, so only
    # the fresh proposal_ffn (created inside BlockwiseParallelWrapper) ends
    # up trainable.
    for p in model.parameters():
        p.requires_grad_(False)

    model.dec = BlockwiseParallelWrapper(model.dec, block_size=args.block_size)
    model.decoder_type = "blockwise"
    model.block_size = args.block_size

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {n_trainable:,} / {n_total:,} total "
                f"({100 * n_trainable / n_total:.2f}%)")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "meta": {
            "decoder_type": "blockwise",
            "block_size": args.block_size,
        },
    }, args.dst)
    logger.info(f"Wrote blockwise-init checkpoint to {args.dst}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
