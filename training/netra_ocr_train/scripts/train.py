"""Train or fine-tune the recognition model.

Usually run through the bash wrappers (from the repository root):

    bash training/finetune.sh my_lines/            # start from the published model
    bash training/train.sh                         # from scratch, public datasets

Data (``--data``, ``--val-data``) is one or more specs: a folder with
``images/`` + ``labels/`` (one .txt per image), or a Hugging Face dataset id
``user/name[:split[:text_column]]`` -- see ``netra_ocr_train/data.py``. Without
``--val-data``, 5% of ``--data`` (50-500 lines) is held out.

Starting points:
  --init CKPT|published   weights only: a fresh optimizer from epoch 0 (fine-tuning)
  --resume CKPT           continue an interrupted run (weights, optimizer, epoch)

Every epoch decodes the validation lines greedily (as the app does) and
writes, in --out-dir: last.pth, best.pth (lowest validation CER), metrics.csv
and train.log.

Decoder types:
  - "ar": standard shift-by-1 teacher forcing, single CrossEntropyLoss.
    Trains the whole model.
  - "blockwise": Stern et al. 2018's frozen-base blockwise parallel decoding
    (model/blockwise_decoder.py). The whole AR model is frozen; only a small
    proposal head that guesses ``block_size - 1`` further tokens is trained.
    Needs ``--init``/``--resume`` pointing at a checkpoint produced by
    ``scripts/init_blockwise_decoder.py`` (which adds the head).
"""

import argparse
import csv
import logging
import math
import os
import random
import time
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
from ..data import DEFAULT_TRAIN_DATA, OCRLineDataset, load_lines, ocr_collate_fn, split_off, unknown_characters
from ..metrics import score
from ..predictor import OCRPredictor
from ..utils import HF_CACHE_DIR, RUNS_DIR, autodetect_config, resolve_checkpoint, setup_logging

DEFAULT_VOCAB = Path(__file__).resolve().parents[1] / "char2idx_cluster.json"
VAL_LIMIT = 1000   # validation lines decoded per epoch, at most

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


def validate(predictor: OCRPredictor, images, labels) -> dict:
    """Greedy-decode the validation lines with the live model and score them."""
    predictor.model.eval()
    with torch.no_grad():
        predictions = predictor.predict_batch(images, beam_width=1, batch_size=16, progress=False)
    metrics, _ = score(labels, predictions)
    return metrics


def _load_state(model, path: Path, device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise SystemExit(f"{path} doesn't match this model (wrong --vocab?):\n{e}") from None
    return checkpoint


def train(args):
    out_dir = Path(args.out_dir or RUNS_DIR / time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=out_dir / "train.log")
    logger.info(f"Run folder: {out_dir}")

    # Must be set before any `import datasets`/`huggingface_hub` so the Hub's
    # download + processed-dataset caches land under the project instead of
    # the default ~/.cache/huggingface/.
    os.environ.setdefault("HF_HOME", str(Path(args.data_dir).resolve()))

    start_from = resolve_checkpoint(args.resume or args.init) if (args.resume or args.init) else None
    cfg = OCRConfig(
        emb_dim=args.emb_dim,
        max_seq_len=args.max_seq_len,
        decode_max_len=args.decode_max_len,
        decoder_type=args.decoder_type,
        block_size=args.block_size,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    if start_from:
        # The checkpoint's own shape (dims, decoder type, block size) wins over the flags.
        for key, value in autodetect_config(start_from).items():
            setattr(cfg, key, value)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        logger.warning("No GPU: training on the CPU is very slow (fine for a smoke test only).")

    # Fresh high-entropy seed every run (not derived from a fixed constant),
    # so RandomSampler's epoch subsampling and each worker's augmentation
    # (see `_worker_init_fn`) are genuinely different run-to-run. Logged so a
    # run can be told apart from a previous one after the fact.
    run_seed = int.from_bytes(os.urandom(4), "little")
    random.seed(run_seed)
    sampler_generator = torch.Generator().manual_seed(run_seed)
    logger.info(f"Run RNG seed: {run_seed}")

    if cfg.decoder_type == "blockwise" and not start_from:
        raise SystemExit(
            "--decoder-type blockwise needs --init or --resume pointing at a checkpoint produced by "
            "scripts/init_blockwise_decoder.py -- the base decoder is frozen and would stay random otherwise.")

    tokenizer = ClusterTokenizer(args.vocab)
    model = build_model(args.model, tokenizer, cfg).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {n_trainable:,} / {n_total:,} total "
                f"({100 * n_trainable / n_total:.2f}%)")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, betas=(0.9, 0.999))
    start_epoch = 0
    best_cer = float("inf")

    if args.resume:
        checkpoint = _load_state(model, start_from, device)
        if "optimizer_state_dict" in checkpoint:
            # LR is recomputed from (epoch, step) every training step below
            # (see `lr_at`), so whatever LR this restores is overwritten
            # before the first optimizer.step().
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_cer = checkpoint.get("meta", {}).get("best_val_cer", best_cer)
        logger.info(f"Resumed {args.resume} at epoch {start_epoch}")
    elif args.init:
        _load_state(model, start_from, device)
        logger.info(f"Starting from the weights of {args.init} (fresh optimizer, epoch 0)")

    if args.epochs <= start_epoch:
        raise SystemExit(
            f"--epochs {args.epochs} is <= the resumed checkpoint's epoch ({start_epoch}); --epochs is the "
            f"epoch to train up to, so pass --epochs {start_epoch + args.epochs} to run {args.epochs} more.")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    logger.info("Loading data (Hugging Face datasets download on the first run)...")
    rows = load_lines(args.data)
    unknown = unknown_characters(rows, tokenizer)
    if unknown:
        top = ", ".join(f"{ch!r}×{n}" for ch, n in sorted(unknown.items(), key=lambda kv: -kv[1])[:15])
        logger.warning(f"Characters the vocabulary can't encode (learned as <unk>, read back as a space): {top}")
    if args.val_data:
        train_rows, val_rows = rows, load_lines(args.val_data)
    else:
        train_rows, val_rows = split_off(rows)
        logger.info(f"Held out {len(val_rows)} of the {len(rows):,} lines for validation")
    val_idx = list(range(len(val_rows)))
    if len(val_idx) > VAL_LIMIT:
        val_idx = sorted(random.Random(0).sample(val_idx, VAL_LIMIT))
    val_images = [val_rows[i]["image"] for i in val_idx]
    val_labels = [val_rows[i]["label"] for i in val_idx]
    logger.info(f"Training on {len(train_rows):,} lines, validating on {len(val_images)}")

    full_dataset = OCRLineDataset(train_rows, tokenizer, cfg, augment=not args.no_augment)
    predictor = OCRPredictor(model=model, tokenizer=tokenizer, config=cfg)

    # Built once and reused across epochs -- RandomSampler draws a fresh
    # random sample_size subset every time the loader is iterated.
    sample_size = min(args.epoch_sample_size or len(full_dataset), len(full_dataset))
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

    metrics_path = out_dir / "metrics.csv"
    if not metrics_path.exists():
        metrics_path.write_text("epoch,train_loss,val_cer,val_wer,val_exact,lr,minutes\n")

    if start_epoch == 0 and not args.no_initial_eval:
        initial = validate(predictor, val_images, val_labels)
        logger.info(f"Before training: val CER {initial['cer']:.4f}  WER {initial['wer']:.4f}  "
                    f"exact {initial['exact_match']:.4f}")

    for epoch in range(start_epoch, args.epochs):
        _set_train_mode(model, cfg.decoder_type)
        started = time.time()
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
                logger.info(f"epoch {epoch + 1}/{args.epochs} step {step}/{steps_per_epoch} "
                            f"loss={loss.item():.4f} lr={lr:.2e}")

        avg_loss = running_loss / max(1, steps_per_epoch)
        val = validate(predictor, val_images, val_labels)
        minutes = (time.time() - started) / 60
        improved = val["cer"] < best_cer
        best_cer = min(best_cer, val["cer"])
        logger.info(f"epoch {epoch + 1}/{args.epochs}: loss {avg_loss:.4f}, val CER {val['cer']:.4f} "
                    f"WER {val['wer']:.4f} exact {val['exact_match']:.4f} ({minutes:.1f} min)"
                    + ("  * best so far" if improved else ""))
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_loss:.5f}", f"{val['cer']:.5f}", f"{val['wer']:.5f}",
                                    f"{val['exact_match']:.5f}", f"{lr:.3e}", f"{minutes:.2f}"])

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "meta": {
                "decoder_type": cfg.decoder_type,
                "block_size": cfg.block_size,
                "vocab": Path(args.vocab).name,
                "val_cer": val["cer"],
                "best_val_cer": best_cer,
            },
        }
        torch.save(checkpoint, out_dir / "last.pth")
        if improved:
            # best.pth is for evaluating/exporting: weights only, a third of the size.
            torch.save({k: v for k, v in checkpoint.items() if k != "optimizer_state_dict"}, out_dir / "best.pth")

    logger.info(f"Done. Best validation CER {best_cer:.4f}: {out_dir / 'best.pth'}")
    return out_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", default=DEFAULT_TRAIN_DATA,
                        help="Training data: folders and/or Hugging Face dataset ids (default: the public datasets)")
    parser.add_argument("--val-data", nargs="+", help="Validation data (default: hold out 5%% of --data)")
    parser.add_argument("--init", help='Start from these weights: a .pth, or "published" (fine-tuning)')
    parser.add_argument("--resume", help="Continue an interrupted run from its last.pth")
    parser.add_argument("--out-dir", help="Run folder (default: training/runs/<date-time>)")
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--data-dir", default=str(HF_CACHE_DIR),
                        help="Hugging Face cache dir (datasets + hub downloads) local to the project")
    parser.add_argument("--model", choices=["se"], default="se")
    parser.add_argument("--decoder-type", choices=["ar", "blockwise"], default="ar")
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--emb-dim", type=int, default=384)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--decode-max-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epoch-sample-size", type=int, default=None,
                        help="Lines drawn per epoch (default: all of them)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Floor LR the cosine decay approaches at the final epoch")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Epochs of linear warmup from 0 to --lr before cosine decay begins")
    parser.add_argument("--no-augment", action="store_true", help="Train on the images as they are")
    parser.add_argument("--no-initial-eval", action="store_true", help="Skip scoring the model before training")
    parser.add_argument("--device", help="cuda, cuda:1 or cpu (default: cuda if available)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args(argv)


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
