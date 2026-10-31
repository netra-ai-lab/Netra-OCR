"""Training data: text-line images and their labels, from local folders or
Hugging Face datasets, prepared into exactly the tensors
``model.KhmerOCR.forward()`` expects (see preprocessor.py).

A data *spec* is either

    path/to/folder              images/<name>.png + labels/<name>.txt (the line's text)
    user/dataset[:split[:col]]  a Hugging Face dataset with an "image" column and a
                                text column (default split "train", column "label",
                                falling back to "text")

Every source becomes rows of ``{"image": PIL.Image, "label": str}``.
"""

import logging
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Subset

from .augment import OCRAugmenter
from .cluster_tokenizer import ClusterTokenizer
from .config import OCRConfig
from .metrics import load_local_dataset
from .preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)

# Invisible in the image, so the model can't learn them: zero-width space (a
# common Khmer word separator in typed text) and the byte-order mark.
_INVISIBLE = str.maketrans("", "", "\u200b\ufeff")


def clean_label(text: str) -> str:
    return text.translate(_INVISIBLE).strip()

# The public datasets the published model was trained on. KhmerSynthetic1M is
# licensed for research only (no commercial use).
DEFAULT_TRAIN_DATA = [
    "SoyVitou/KhmerSynthetic1M",
    "seanghay/khmer-hanuman-100k:train:text",
    "Darayut/khmer-scene-text-synthetic-contrast",
    "Darayut/khmer-document-synthetic-low-res",
]


class FolderLines(Dataset):
    """An {images/, labels/} folder, images opened lazily."""

    def __init__(self, folder: str | Path):
        self.paths, self.labels = load_local_dataset(Path(folder))
        if not self.paths:
            raise ValueError(f"No labelled images in {folder} (expected images/<name>.png + labels/<name>.txt)")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return {"image": Image.open(self.paths[idx]).convert("RGB"), "label": self.labels[idx]}


def _load_hf(spec: str):
    import datasets as hf_datasets

    repo_id, split, column = (spec.split(":") + [None, None])[:3]
    ds = hf_datasets.load_dataset(repo_id, split=split or "train")
    if column is None:
        column = "label" if "label" in ds.column_names else "text"
    if column != "label":
        ds = ds.rename_column(column, "label")
    extra = [c for c in ds.column_names if c not in ("image", "label")]
    return ds.remove_columns(extra) if extra else ds


def load_lines(specs: str | list[str]) -> Dataset:
    """Rows of {"image", "label"} from one or more data specs, concatenated."""
    parts = []
    for spec in [specs] if isinstance(specs, str) else specs:
        part = FolderLines(spec) if Path(spec).is_dir() else _load_hf(spec)
        logger.info(f"{spec}: {len(part):,} lines")
        parts.append(part)
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def split_off(rows: Dataset, fraction: float = 0.05, minimum: int = 50, maximum: int = 2000,
              seed: int = 0) -> tuple[Dataset, Dataset]:
    """(train, validation): a fixed random ``fraction`` of ``rows`` held out."""
    n_val = min(maximum, max(minimum, int(len(rows) * fraction)))
    if n_val >= len(rows):
        raise ValueError(f"Only {len(rows)} lines: too few to hold out {n_val} for validation; pass --val-data")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return Subset(rows, order[n_val:]), Subset(rows, order[:n_val])


def _labels(rows, limit: int) -> list[str]:
    """Up to ``limit`` labels, without decoding any image."""
    if isinstance(rows, FolderLines):
        return rows.labels[:limit]
    if isinstance(rows, ConcatDataset):
        out = []
        for part in rows.datasets:
            out += _labels(part, limit - len(out))
            if len(out) >= limit:
                break
        return out
    return rows.select(range(min(limit, len(rows))))["label"]   # a Hugging Face dataset


def unknown_characters(rows: Dataset, tokenizer: ClusterTokenizer, limit: int = 20000) -> dict[str, int]:
    """Characters in (up to ``limit``) labels the vocabulary can't encode, with
    counts. ``rows`` comes straight from `load_lines` (not a split)."""
    counts = {}
    for label in _labels(rows, limit):
        for ch in clean_label(label):
            if ch not in tokenizer.char2idx and not ch.isspace():
                counts[ch] = counts.get(ch, 0) + 1
    return counts


class OCRLineDataset(Dataset):
    """Wraps rows of {"image": PIL.Image, "label": str}, applying the same
    resize->[augment]->chunk->normalize pipeline as inference
    (`ImagePreprocessor`) plus cluster-tokenizer label encoding."""

    def __init__(self, rows, tokenizer: ClusterTokenizer, config: OCRConfig, augment: bool = False):
        self.data = rows
        self.tokenizer = tokenizer
        self.preprocessor = ImagePreprocessor(config)
        self.augmenter = OCRAugmenter() if augment else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image = example["image"]
        text = clean_label(example["label"])

        tensor = self.preprocessor.load_and_resize(image)
        if self.augmenter is not None:
            tensor = self.augmenter(tensor)
        chunks = self.preprocessor.chunk_and_normalize(tensor)

        label = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return chunks, label, text


def ocr_collate_fn(batch, pad_idx: int):
    """Mirrors the notebook's khmer_collate_fn, but pads with the tokenizer's
    actual pad_idx instead of a hardcoded 0."""
    chunk_lists, labels, texts = zip(*batch)
    max_len = max(len(l) for l in labels)
    label_pad = torch.full((len(labels), max_len), pad_idx, dtype=torch.long)
    for i, l in enumerate(labels):
        label_pad[i, :len(l)] = l
    return list(chunk_lists), label_pad, list(texts)
