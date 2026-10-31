"""Evaluate AR vs. blockwise-parallel recognition checkpoints against an
external OCR benchmark dataset (CER / WER / exact-match).

Both the ground-truth label and the model's raw decoded output are put
through the same normalization before comparison (Unicode NFC + the
project's own `clean_text` decoder-artifact cleanup + whitespace collapse),
so neither side is unfairly penalized for cosmetic differences that aren't
real transcription errors.

Usage (current cluster-vocab AR model -- the defaults):
    python -m netra_ocr_train.scripts.evaluate_benchmark \\
        --local-dir training/test_set/khob_eval \\
        --skip-blockwise \\
        --out-dir eval_out

The --ar-* defaults point at the cluster-vocab AR model now being trained
(`khmerocr_se_ar_last.pth` + `char2idx_cluster.json`). Blockwise decoding
isn't trained yet (Pending #3), so pass --skip-blockwise until an
`init_blockwise_decoder.py`-produced checkpoint exists. To benchmark the
original 176-vocab AR base instead, pass:
    --ar-model training/weights/khmerocr_epoch570.pth \\
    --ar-vocab training/netra_ocr_train/char2idx_new.json
"""

import argparse
import logging
import os
import time
from pathlib import Path

import torch
from khmerspell import khnormal
from PIL import Image, ImageOps

from ..config import OCRConfig
from ..cluster_tokenizer import ClusterTokenizer
from ..postprocess import clean_text
from ..predictor import OCRPredictor
from ..utils import autodetect_config, setup_logging
from .ocr_eval_common import levenshtein, load_local_dataset as _load_local_dataset, write_records

logger = logging.getLogger(__name__)


def _model_class(model_path: str):
    from ..model.model import KhmerOCR
    return KhmerOCR


def normalize(text: str) -> str:
    """Normalize a transcription for fair comparison: the project's own
    decoder-artifact cleanup (`clean_text`), Khmer-specific canonical
    reordering (`khmerspell.khnormal` -- unlike generic Unicode NFC, this
    also fixes Khmer-specific issues like out-of-order combining vowels/
    coeng-stacks and confusable vowel sequences, so two visually-identical
    renderings of the same word can't count as a mismatch), then whitespace
    collapse/strip.

    KHOB-LEVEL-1's ground truth uses a literal two-character ``\\t`` escape
    (backslash + "t") as a multi-field separator rather than a real tab --
    `str.split()` doesn't break on that, so left as-is it silently collapses
    multi-word targets into one giant "word" while the model's real-space
    output stays several words, wrecking WER. Normalize it to a space like
    any other field separator before comparing.
    """
    text = text.replace("\\t", " ").replace("\t", " ")
    text = clean_text(text)
    text = khnormal(text)
    return text.strip()


def pad_image(image, pad_px: int):
    """Add a solid white border on all four sides. White (255) matches the
    padding value ImagePreprocessor already uses to pad ragged chunk widths
    (see preprocessor.py's `_chunk_tensor`), so this only adds margin -- it
    doesn't introduce a color the model wasn't already trained to see."""
    if pad_px <= 0:
        return image
    return ImageOps.expand(image, border=pad_px, fill=(255, 255, 255))


def build_predictor(model_path: str, vocab_path: str) -> OCRPredictor:
    detected_cfg = autodetect_config(model_path)
    config = OCRConfig(**detected_cfg)
    tokenizer = ClusterTokenizer(vocab_path)
    return OCRPredictor(
        model_path=model_path,
        tokenizer=tokenizer,
        config=config,
        model_class=_model_class(model_path),
    )


def evaluate(predictor: OCRPredictor, images, targets_raw, beam_width: int, label: str):
    is_blockwise = predictor.cfg.decoder_type == "blockwise"
    # Enable the predictor's per-round accepted-block-size recorder for
    # blockwise; leave it None otherwise so no overhead on the AR path.
    predictor.decode_stats = {"accepted": []} if is_blockwise else None

    # Wall-clock the full recognition pass (preprocess + CNN/encoder + decode).
    # Sync CUDA on both sides so we time real GPU work, not just kernel launch.
    if predictor.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    predictions_raw = predictor.predict_batch(images, beam_width=beam_width, batch_size=8)
    if predictor.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_imgs = len(images)
    ms_per_image = 1000.0 * elapsed / n_imgs if n_imgs else 0.0
    lines_per_sec = n_imgs / elapsed if elapsed else 0.0
    # Mean accepted block size: tokens committed per decoder forward pass.
    # 1.0 == no speculative speedup (greedy-equivalent); higher == fewer
    # passes per line == faster. Only meaningful for blockwise.
    accepted = predictor.decode_stats["accepted"] if is_blockwise else []
    mean_block = sum(accepted) / len(accepted) if accepted else float("nan")
    predictor.decode_stats = None

    total_char_edits = total_target_chars = 0
    total_word_edits = total_target_words = 0
    exact = 0
    records = []
    for target_raw, pred_raw in zip(targets_raw, predictions_raw):
        target = normalize(target_raw)
        pred = normalize(pred_raw)

        total_char_edits += levenshtein(list(target), list(pred))
        total_target_chars += len(target)
        total_word_edits += levenshtein(target.split(), pred.split())
        total_target_words += len(target.split())
        exact += int(pred == target)
        records.append({
            "target_raw": target_raw, "prediction_raw": pred_raw,
            "target_norm": target, "prediction_norm": pred,
        })

    n = len(records)
    metrics = {
        "label": label,
        "n": n,
        "cer": total_char_edits / total_target_chars if total_target_chars else 0.0,
        "wer": total_word_edits / total_target_words if total_target_words else 0.0,
        "exact_match": exact / n if n else 0.0,
        "ms_per_image": ms_per_image,
        "lines_per_sec": lines_per_sec,
        "mean_block_size": mean_block,
    }
    return metrics, records


def load_local_dataset(local_dir: Path):
    """Loads an {images/, labels/}-directory benchmark as PIL images (this
    script's own callers expect images, not paths -- see ocr_eval_common's
    version, shared with the other engines' scripts, for the path-based
    variant used where lazy per-image loading matters more)."""
    image_paths, targets = _load_local_dataset(local_dir)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    return images, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SoyVitou/KHOB-LEVEL-1",
                         help="HF dataset repo id (ignored if --local-dir is given)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-dir", default="training/.hf_cache",
                         help="HF cache dir local to the project (same convention as train.py)")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--local-dir", default=None,
                         help="Path to a local {images/, labels/} benchmark dir "
                              "(e.g. training/test_set/khob_eval) -- takes precedence over --dataset")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows")

    # Defaults target the current cluster-vocab AR model (Pending #2). To
    # benchmark the original 176-vocab AR base instead, pass
    # --ar-model .../khmerocr_epoch570.pth --ar-vocab .../char2idx_new.json
    parser.add_argument("--ar-model", default="training/weights/khmerocr_se_ar_last.pth")
    parser.add_argument("--ar-vocab", default="training/netra_ocr_train/char2idx_cluster.json")
    parser.add_argument("--ar-beam-width", type=int, default=3)

    parser.add_argument("--blockwise-model", nargs="+",
                         default=["training/weights/khmerocr_se_blockwise_last.pth"],
                         help="One or more blockwise checkpoints. block_size is auto-detected "
                              "from each checkpoint's meta; each is labeled blockwise_bs<N>. "
                              "Pass several to compare block sizes in one table (e.g. a 4/6/8/10 sweep).")
    parser.add_argument("--blockwise-vocab", default="training/netra_ocr_train/char2idx_cluster.json")

    parser.add_argument("--skip-ar", action="store_true")
    parser.add_argument("--skip-blockwise", action="store_true")
    parser.add_argument("--pad-px", type=int, nargs="+", default=[0],
                         help="Sweep: evaluate once per white border width (px) added to each image, e.g. 0 1 2 3")
    parser.add_argument("--out-dir", default="eval_out")
    args = parser.parse_args()

    setup_logging()

    if args.local_dir:
        logger.info(f"Loading local benchmark dir {args.local_dir}...")
        base_images, targets_raw = load_local_dataset(Path(args.local_dir))
    else:
        os.environ.setdefault("HF_HOME", str(Path(args.data_dir).resolve()))
        import datasets as hf_datasets
        logger.info(f"Loading benchmark dataset {args.dataset} [{args.split}]...")
        ds = hf_datasets.load_dataset(args.dataset, split=args.split)
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        base_images = [row.convert("RGB") for row in ds["image"]]
        targets_raw = ds[args.text_column]

    if args.limit:
        base_images = base_images[:args.limit]
        targets_raw = targets_raw[:args.limit]
    logger.info(f"Loaded {len(base_images)} rows")

    out_dir = Path(args.out_dir)
    results = []

    ar_predictor = None
    if not args.skip_ar:
        logger.info(f"Building AR predictor from {args.ar_model} / {args.ar_vocab}")
        ar_predictor = build_predictor(args.ar_model, args.ar_vocab)

    blockwise_predictors = []  # list of (label, predictor)
    if not args.skip_blockwise:
        seen_labels = {}
        for bw_model in args.blockwise_model:
            logger.info(f"Building blockwise predictor from {bw_model} / {args.blockwise_vocab}")
            pred = build_predictor(bw_model, args.blockwise_vocab)
            label = f"blockwise_bs{pred.cfg.block_size}"
            # Disambiguate if two checkpoints share the same block size.
            if label in seen_labels:
                seen_labels[label] += 1
                label = f"{label}_{seen_labels[label]}"
            else:
                seen_labels[label] = 0
            blockwise_predictors.append((label, pred))

    for pad_px in args.pad_px:
        logger.info(f"--- pad_px={pad_px} ---")
        images = [pad_image(img, pad_px) for img in base_images]

        if ar_predictor is not None:
            metrics, records = evaluate(ar_predictor, images, targets_raw, args.ar_beam_width, "ar")
            metrics["pad_px"] = pad_px
            results.append(metrics)
            write_records(records, out_dir / f"ar_predictions_pad{pad_px}.csv")

        for label, pred in blockwise_predictors:
            # beam_width is ignored for decoder_type="blockwise" (Predict/Verify/Accept only).
            metrics, records = evaluate(pred, images, targets_raw, 1, label)
            metrics["pad_px"] = pad_px
            results.append(metrics)
            write_records(records, out_dir / f"{label}_predictions_pad{pad_px}.csv")

    logger.info("=" * 98)
    logger.info(f"{'model':<16} {'pad_px':>6} {'n':>5} {'CER':>8} {'WER':>8} "
                f"{'exact':>7} {'ms/img':>9} {'lines/s':>9} {'blk_size':>9}")
    for m in results:
        blk = f"{m['mean_block_size']:.3f}" if m['mean_block_size'] == m['mean_block_size'] else "-"
        logger.info(f"{m['label']:<16} {m['pad_px']:>6} {m['n']:>5} {m['cer']:>8.4f} {m['wer']:>8.4f} "
                    f"{m['exact_match']:>7.4f} {m['ms_per_image']:>9.2f} {m['lines_per_sec']:>9.2f} {blk:>9}")
    logger.info("=" * 98)

    # For each blockwise model that ran at the same pad_px as AR, report the
    # measured end-to-end decode speedup -- the whole point of Stage 2. With a
    # block-size sweep this prints one line per block size so they're directly
    # comparable.
    by_pad = {}
    for m in results:
        by_pad.setdefault(m["pad_px"], {})[m["label"]] = m
    for pad_px, row in by_pad.items():
        ar = row.get("ar")
        for label, m in row.items():
            if label.startswith("blockwise") and m["ms_per_image"] and ar and ar["ms_per_image"]:
                speedup = ar["ms_per_image"] / m["ms_per_image"]
                logger.info(f"pad_px={pad_px}: {label} is {speedup:.2f}x faster than AR "
                            f"(mean accepted block size {m['mean_block_size']:.3f})")


if __name__ == "__main__":
    main()
