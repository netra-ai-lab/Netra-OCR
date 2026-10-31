"""Training entrypoint for the recognition model.

Supports both decoder architectures:
  - "ar": standard shift-by-1 teacher forcing, single CrossEntropyLoss.
    Trains the whole model from scratch/resume.
  - "blockwise": Stern et al. 2018's frozen-base blockwise parallel decoding
    (see research/Blockwise Parallel Decoding for Deep Autoregressive
    Models.pdf and model/blockwise_decoder.py). The *entire* model --
    CNN/encoder/BiLSTM/decoder attention stack/tok_emb/out_proj -- is frozen
    exactly as it was in the source AR checkpoint; only a small new
    feedforward head (added by `init_blockwise_decoder.py`) that guesses
    `block_size - 1` further tokens ahead is trained. Requires `--resume`
    pointing at a checkpoint produced by
    `scripts/init_blockwise_decoder.py` (not a raw AR checkpoint -- that
    script adds the trainable head first). Because almost the whole network
    is frozen, this needs far less training time/data than "ar" from
    scratch or the old semi-AR schemes this replaced.

Usage:
    python -m netra_ocr_train.scripts.init_blockwise_decoder \\
        --src training/weights/khmerocr_epoch570.pth \\
        --dst training/weights/khmerocr_blockwise_init.pth \\
        --vocab training/netra_ocr_train/char2idx_new.json --block-size 4

    python -m netra_ocr_train.scripts.train \\
        --vocab training/netra_ocr_train/char2idx_new.json \\
        --resume training/weights/khmerocr_blockwise_init.pth \\
        --decoder-type blockwise --block-size 4 \\
        --epochs 20 --batch-size 32 --epoch-sample-size 50000 \\
        --out-dir training/netra_ocr_train/weight
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path

import torch
import torch.multiprocessing
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

# Each sample yields several small chunk tensors (not one big tensor), so the
# default "file_descriptor" strategy for handing tensors from DataLoader
# workers to the main process can exhaust the process's open-file limit under
# num_workers>0 + persistent_workers + prefetch. "file_system" avoids fds
# entirely by backing shared tensors with named temp files instead.
torch.multiprocessing.set_sharing_strategy("file_system")

from ..cluster_tokenizer import ClusterTokenizer
from ..config import OCRConfig
from .dataset import DATASET_SOURCES, OCRLineDataset, load_training_datasets, ocr_collate_fn
from ..predictor import OCRPredictor
from ..utils import setup_logging

logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id):
    """DataLoader workers are created via fork() on Linux, so each worker
    inherits an *identical copy* of the parent process's Python `random`
    module state at fork time. PyTorch's DataLoader machinery already reseeds
    each worker's own torch RNG uniquely (`torch.initial_seed()` returns that
    per-worker seed), but never touches the global `random` module -- so
    without this, `augment.py`'s `random.random()`/`random.uniform()` calls
    (pixelation/erosion/blur/noise/shift) produce correlated, less-diverse
    sequences across workers instead of independent ones.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)


def _model_class(name: str):
    from ..model.model import KhmerOCR
    return KhmerOCR


def build_model(model_name: str, tokenizer: ClusterTokenizer, cfg: OCRConfig):
    KhmerOCR = _model_class(model_name)
    return KhmerOCR(
        vocab_size=len(tokenizer),
        pad_idx=tokenizer.pad_idx,
        emb_dim=cfg.emb_dim,
        max_global_len=cfg.max_seq_len,
        decoder_type=cfg.decoder_type,
        block_size=cfg.block_size,
    )


def ar_loss_step(model, chunk_lists, label_pad, criterion):
    """Standard shift-by-1 teacher forcing."""
    logits = model(chunk_lists, label_pad[:, :-1])
    vocab_size = logits.size(-1)
    loss = criterion(logits.reshape(-1, vocab_size), label_pad[:, 1:].reshape(-1))
    return loss


def blockwise_loss_step(model, chunk_lists, label_pad, criterion):
    """Trains only the small proposal head added by `BlockwiseParallelWrapper`
    (Stern et al. 2018) -- everything else (CNN, encoder, BiLSTM, base AR
    decoder) is frozen (see `model/blockwise_decoder.py` and
    `init_blockwise_decoder.py`), so this is vastly cheaper than the old
    semi-AR schemes' full-decoder retraining. Standard shift-by-1 AR input,
    but the model now returns `block_size` predictions per position: index 0
    (p1) is the frozen base decoder's own next-token logits and is skipped
    here (nothing to train, and it's identical to `ar_loss_step`'s target
    anyway); auxiliary head `a` (0-indexed, a=0 is p2) predicts the target
    token `a + 2` positions ahead of its own input position.
    """
    dec_input = label_pad[:, :-1]
    target = label_pad[:, 1:]
    logits = model(chunk_lists, dec_input)  # (B, T, block_size, vocab)
    vocab_size = logits.size(-1)
    n_aux = logits.size(2) - 1
    T = target.size(1)

    total_loss = 0.0
    for a in range(n_aux):
        shift = a + 1
        if shift >= T:
            continue
        aux_logits = logits[:, :T - shift, a + 1, :]
        aux_target = target[:, shift:]
        total_loss = total_loss + criterion(aux_logits.reshape(-1, vocab_size), aux_target.reshape(-1))

    return total_loss / n_aux


def _set_train_mode(model, decoder_type):
    """Standard `model.train()`, except for "blockwise": `requires_grad=False`
    alone doesn't stop BatchNorm layers (in the frozen CNN) from updating
    their running mean/var in train() mode, which would silently drift the
    frozen backbone away from what it was actually trained with. Nothing
    else in this model has persistent running statistics (LayerNorm/LSTM
    don't), so freezing the CNN back to eval() is sufficient."""
    model.train()
    if decoder_type == "blockwise":
        model.cnn.eval()


def lr_at(global_step, total_steps, warmup_steps, base_lr, min_lr):
    """Linear warmup for `warmup_steps`, then cosine decay from `base_lr`
    down to `min_lr` over the remaining steps. Pure function of step count --
    unlike CyclicLR, there's no scheduler object whose internal cycle
    position must be reconstructed/re-armed on resume; resuming from any
    epoch just re-evaluates this formula at the right point.
    """
    if warmup_steps > 0 and global_step < warmup_steps:
        return base_lr * (global_step + 1) / warmup_steps
    if global_step >= total_steps:
        return min_lr
    progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or (out_dir / "train.log")
    setup_logging(log_file=log_file)

    # Must be set before any `import datasets`/`huggingface_hub` (done lazily
    # inside dataset.py) so the Hub's download + processed-dataset caches land
    # under the project instead of the default ~/.cache/huggingface/.
    os.environ.setdefault("HF_HOME", str(Path(args.data_dir).resolve()))

    cfg = OCRConfig(
        emb_dim=args.emb_dim,
        max_seq_len=args.max_seq_len,
        decode_max_len=args.decode_max_len,
        decoder_type=args.decoder_type,
        block_size=args.block_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Fresh high-entropy seed every run (not derived from a fixed constant),
    # so RandomSampler's epoch subsampling and each worker's augmentation
    # (see `_worker_init_fn`) are genuinely different run-to-run rather than
    # riding on whatever the global RNG happened to inherit. Logged so a run
    # can be told apart from a previous one after the fact.
    run_seed = int.from_bytes(os.urandom(4), "little")
    random.seed(run_seed)
    sampler_generator = torch.Generator().manual_seed(run_seed)
    logger.info(f"Run RNG seed: {run_seed}")

    if cfg.decoder_type == "blockwise" and not args.resume:
        raise ValueError(
            "--decoder-type blockwise requires --resume pointing at a checkpoint produced by "
            "scripts/init_blockwise_decoder.py -- the base decoder is frozen and randomly "
            "initialized otherwise, which is not trainable from scratch."
        )

    tokenizer = ClusterTokenizer(args.vocab)

    model = build_model(args.model, tokenizer, cfg).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {n_trainable:,} / {n_total:,} total "
                f"({100 * n_trainable / n_total:.2f}%)")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, betas=(0.9, 0.999))
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # LR is recomputed from (epoch, step) every training step below
            # (see `lr_at`), so whatever LR this restores is irrelevant and
            # gets overwritten before the first optimizer.step() -- resuming
            # can never silently inherit a stale/collapsed LR.
            start_epoch = checkpoint.get("epoch", 0)
            logger.info(f"Resumed training from epoch {start_epoch}")
        else:
            logger.info(f"Loaded weights from {args.resume} (fresh optimizer/epoch=0)")

    if args.epochs <= start_epoch:
        raise ValueError(
            f"--epochs {args.epochs} is <= the resumed checkpoint's epoch ({start_epoch}); "
            f"--epochs is the absolute target epoch to train up to, not a number of additional "
            f"epochs, so range(start_epoch, args.epochs) would be empty and train() would "
            f"silently do nothing. Pass --epochs {start_epoch + args.epochs} to run "
            f"{args.epochs} more epochs from here."
        )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    logger.info("Loading training datasets (this downloads/caches them on first run)...")
    hf_dataset = load_training_datasets(DATASET_SOURCES)
    logger.info(f"Loaded {len(hf_dataset)} total rows")
    full_dataset = OCRLineDataset(hf_dataset, tokenizer, cfg, augment=True)

    # Fixed reference sample held out for a qualitative per-epoch check --
    # does the model still transcribe correctly, in plain text, alongside the loss?
    reference_row = hf_dataset[0]
    reference_image, reference_label = reference_row["image"], reference_row["label"]
    predictor = OCRPredictor(model=model, tokenizer=tokenizer, config=cfg)

    ckpt_path = out_dir / f"khmerocr_{args.model}_{cfg.decoder_type}_last.pth"

    # Built once and reused across epochs -- RandomSampler draws a fresh
    # random sample_size subset every time the loader is iterated, so there's
    # no need to rebuild the Dataset/DataLoader (and respawn the worker pool)
    # each epoch like the old Subset(randperm(...)) approach did.
    sample_size = min(args.epoch_sample_size, len(full_dataset))
    sampler = RandomSampler(full_dataset, replacement=False, num_samples=sample_size,
                             generator=sampler_generator)
    loader = DataLoader(
        full_dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=lambda b: ocr_collate_fn(b, tokenizer.pad_idx),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    )

    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    for epoch in range(start_epoch, args.epochs):
        _set_train_mode(model, cfg.decoder_type)
        running_loss = 0.0
        for step, (chunk_lists, label_pad, _texts) in enumerate(loader):
            chunk_lists = [[c.to(device) for c in img] for img in chunk_lists]
            label_pad = label_pad.to(device)

            global_step = epoch * steps_per_epoch + step
            lr = lr_at(global_step, total_steps, warmup_steps, args.lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
                if cfg.decoder_type == "blockwise":
                    loss = blockwise_loss_step(model, chunk_lists, label_pad, criterion)
                else:
                    loss = ar_loss_step(model, chunk_lists, label_pad, criterion)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % args.log_every == 0:
                logger.info(f"epoch {epoch} step {step}/{len(loader)} loss={loss.item():.4f} lr={lr:.2e}")

        avg_loss = running_loss / max(1, len(loader))
        logger.info(f"epoch {epoch} done, avg_loss={avg_loss:.4f}")

        model.eval()
        predicted = predictor.predict(reference_image)
        _set_train_mode(model, cfg.decoder_type)
        logger.info(f"epoch {epoch} sample check: predicted={predicted!r} expected={reference_label!r}")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "meta": {
                "decoder_type": cfg.decoder_type,
                "block_size": cfg.block_size,
            },
        }, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", default="training/netra_ocr_train/char2idx_cluster.json")
    parser.add_argument("--data-dir", default="training/.hf_cache",
                         help="Hugging Face cache dir (datasets + hub downloads) local to the project")
    parser.add_argument("--resume", default=None, help="Checkpoint to warm-start/resume from")
    parser.add_argument("--model", choices=["se"], default="se")
    parser.add_argument("--decoder-type", choices=["ar", "blockwise"], default="ar")
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--emb-dim", type=int, default=384)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--decode-max-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epoch-sample-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6,
                         help="Floor LR the cosine decay approaches at the final epoch")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                         help="Epochs of linear warmup from 0 to --lr before cosine decay begins")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--log-file", default=None,
                         help="Path to write training logs to (default: <out-dir>/train.log)")
    parser.add_argument("--out-dir", default="training/weights")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
