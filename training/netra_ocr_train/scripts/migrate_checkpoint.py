"""Migrate an AR checkpoint's vocabulary (e.g. old 176-token codepoint vocab
-> new cluster vocab).

(For bootstrapping the frozen-base "blockwise" decoder instead of migrating
vocab, see `scripts/init_blockwise_decoder.py` -- that one deliberately does
NOT touch the vocab, since the base model is frozen afterward and any
never-fine-tuned embedding row introduced by a vocab migration would stay
broken permanently.)

Copied unchanged (vocab-independent, purely visual): ``cnn.*``, ``patch.*``,
``enc.*``, ``global_pos``, ``context_bilstm.*`` (SE variant only).

Reinitialized (vocab/seq-length dependent), with an id-aligned warm-start
row-copy for the ids shared between old and new vocabs (ids 0-175, since
``build_cluster_vocab.py`` constructs the new vocab as a strict superset that
keeps those ids identical): ``dec.tok_emb.weight``, ``dec.out_proj.weight``,
``dec.out_proj.bias``, ``dec.pos_emb`` (length-prefix copy only).

``dec.decoder.*`` (the actual self-/cross-attention + FFN stack) is copied
unchanged across both migrations, since neither changes its shape.

This script does not run any training -- it only produces a new checkpoint
file that a future training/fine-tuning session can load as a warm start.

Usage:
    python -m netra_ocr_train.scripts.migrate_checkpoint \\
        --src training/weights/khmerocr_epoch570.pth \\
        --dst training/weights/khmerocr_cluster_ar.pth \\
        --old-vocab training/netra_ocr_train/char2idx_new.json \\
        --new-vocab training/netra_ocr_train/char2idx_cluster.json
"""

import argparse
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _model_class_for(path: Path):
    from netra_ocr_train.model.model import KhmerOCR as SE_KhmerOCR
    return SE_KhmerOCR


# Prefixes that are entirely vocab/decoder-architecture independent and can
# always be copied verbatim between old and new checkpoints.
_UNCHANGED_PREFIXES = ("cnn.", "patch.", "enc.", "global_pos", "context_bilstm.")

# The decoder self-/cross-attention + FFN stack: shape depends only on
# emb_dim/nhead/num_layers, not on vocab_size, so it always transfers
# unchanged too.
_DECODER_STACK_PREFIX = "dec.decoder."

# Vocab/seq-length-dependent keys that need reinit + (optional) warm start.
_VOCAB_DEPENDENT_KEYS = ("dec.tok_emb.weight", "dec.out_proj.weight", "dec.out_proj.bias")


def migrate_state_dict(old_state: dict, new_model_state: dict,
                        old_vocab: dict, new_vocab: dict) -> tuple[dict, dict]:
    """Returns (migrated_state_dict, log) where log maps each key to
    'copied' | 'warm-started' | 'reinitialized' for auditability."""
    migrated = dict(new_model_state)
    log = {}

    for key, new_tensor in new_model_state.items():
        if key.startswith(_UNCHANGED_PREFIXES) or key.startswith(_DECODER_STACK_PREFIX):
            if key in old_state and old_state[key].shape == new_tensor.shape:
                migrated[key] = old_state[key].clone()
                log[key] = "copied"
            else:
                log[key] = "reinitialized (shape mismatch or missing in source)"
            continue

        if key == "dec.pos_emb":
            if key in old_state:
                old_len = old_state[key].shape[0]
                new_len = new_tensor.shape[0]
                prefix_len = min(old_len, new_len)
                migrated_tensor = new_tensor.clone()
                migrated_tensor[:prefix_len] = old_state[key][:prefix_len]
                migrated[key] = migrated_tensor
                log[key] = f"warm-started (copied first {prefix_len} positions)"
            else:
                log[key] = "reinitialized"
            continue

        if key in _VOCAB_DEPENDENT_KEYS and key in old_state:
            # Row-copy for every token string present in both vocabs (ids
            # 0-175 by construction -- build_cluster_vocab.py keeps those ids
            # identical, so this is effectively an id-for-id copy there, but
            # matching by token string is robust even if ids ever diverge).
            migrated_tensor = new_tensor.clone()
            n_copied = 0
            for tok, old_id in old_vocab.items():
                new_id = new_vocab.get(tok)
                if new_id is None:
                    continue
                migrated_tensor[new_id] = old_state[key][old_id]
                n_copied += 1
            migrated[key] = migrated_tensor
            log[key] = f"warm-started ({n_copied} rows copied by shared token id)"
            continue

        log[key] = "reinitialized (no migration rule matched)"

    return migrated, log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Source .pth checkpoint")
    parser.add_argument("--dst", type=Path, required=True, help="Output .pth path")
    parser.add_argument("--old-vocab", type=Path, required=True)
    parser.add_argument("--new-vocab", type=Path, required=True)
    parser.add_argument("--emb-dim", type=int, default=384)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    args = parser.parse_args()

    with open(args.old_vocab, "r", encoding="utf-8") as f:
        old_vocab = json.load(f)
    with open(args.new_vocab, "r", encoding="utf-8") as f:
        new_vocab = json.load(f)

    old_checkpoint = torch.load(args.src, map_location="cpu")
    old_state = old_checkpoint.get("model_state_dict", old_checkpoint)

    model_class = _model_class_for(args.src)
    new_model = model_class(
        vocab_size=len(new_vocab),
        pad_idx=new_vocab.get("<pad>", 0),
        emb_dim=args.emb_dim,
        max_global_len=args.max_seq_len,
        decoder_type="ar",
    )
    new_state = new_model.state_dict()

    migrated_state, log = migrate_state_dict(old_state, new_state, old_vocab, new_vocab)
    new_model.load_state_dict(migrated_state, strict=True)

    print("=== Migration log ===")
    for key, status in log.items():
        print(f"  {key:40s} {status}")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": new_model.state_dict(),
        "meta": {
            "decoder_type": "ar",
        },
    }, args.dst)
    print()
    print(f"Wrote migrated checkpoint to {args.dst}")


if __name__ == "__main__":
    main()
