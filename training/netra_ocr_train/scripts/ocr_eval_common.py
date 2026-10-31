"""Shared, dependency-light helpers for OCR benchmark scripts.

Deliberately has no torch/cv2/ultralytics imports so it can be imported from
environments that don't have this project's full stack installed (e.g. the
separate venv used for DeepSeek-OCR, which pins an old transformers that
conflicts with this project's own dependencies).
"""

import csv
import logging
from pathlib import Path

from khmerspell import khnormal
from PIL import Image

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Minimal decoder-artifact cleanup shared across benchmark scripts.

    Kept intentionally small here (just whitespace collapse) -- the fuller
    `clean_text` in ..postprocess also strips model-specific special tokens,
    which don't apply to third-party OCR engines' raw output.
    """
    return " ".join(text.split())


def normalize(text: str) -> str:
    """Normalize a transcription for fair comparison across OCR engines:
    the `\\t`-escape fix, whitespace collapse, then Khmer-specific canonical
    reordering (`khmerspell.khnormal`) so two visually-identical renderings
    of the same word can't count as a mismatch.
    """
    text = text.replace("\\t", " ").replace("\t", " ")
    text = clean_text(text)
    text = khnormal(text)
    return text.strip()


def levenshtein(a: list, b: list) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def score(targets_raw, predictions_raw):
    """Compute CER/WER/exact-match over a batch of (target, prediction) pairs,
    returning (metrics_dict_without_timing, per-row records)."""
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
        "n": n,
        "cer": total_char_edits / total_target_chars if total_target_chars else 0.0,
        "wer": total_word_edits / total_target_words if total_target_words else 0.0,
        "exact_match": exact / n if n else 0.0,
    }
    return metrics, records


def load_local_dataset(local_dir: Path):
    """Loads an {images/, labels/}-directory benchmark: one image per file in
    `images/`, ground truth in `labels/<same-stem>.txt`. Matched by filename
    stem (not by sorted position) so the two directories don't need to be in
    sync ordering-wise. Sorted by filename so --limit N always selects the
    same N rows across scripts/engines."""
    images_dir = Path(local_dir) / "images"
    labels_dir = Path(local_dir) / "labels"
    image_paths, targets = [], []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            logger.warning(f"No label for {img_path.name}, skipping")
            continue
        image_paths.append(img_path)
        targets.append(label_path.read_text(encoding="utf-8").strip())
    return image_paths, targets


def load_images(image_paths):
    return [Image.open(p).convert("RGB") for p in image_paths]


def write_records(records, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["target_raw", "prediction_raw", "target_norm", "prediction_norm"])
        writer.writeheader()
        writer.writerows(records)


def print_summary(rows):
    """rows: list of dicts with keys label,n,cer,wer,exact_match,ms_per_image,lines_per_sec"""
    logger.info("=" * 90)
    logger.info(f"{'model':<16} {'n':>5} {'CER':>8} {'WER':>8} {'exact':>7} {'ms/img':>9} {'lines/s':>9}")
    for m in rows:
        logger.info(f"{m['label']:<16} {m['n']:>5} {m['cer']:>8.4f} {m['wer']:>8.4f} "
                     f"{m['exact_match']:>7.4f} {m['ms_per_image']:>9.2f} {m['lines_per_sec']:>9.2f}")
    logger.info("=" * 90)
