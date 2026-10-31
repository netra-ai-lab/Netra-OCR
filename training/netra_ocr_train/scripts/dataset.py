"""Training dataset: loads line-image/text-label pairs from Hugging Face
datasets and prepares them into exactly the tensor format
`model.KhmerOCR.forward()`/`encode()` expects (see preprocessor.py's
docstrings for the full preprocessing contract).
"""

import torch
from torch.utils.data import Dataset

from ..config import OCRConfig
from ..preprocessor import ImagePreprocessor
from .augment import OCRAugmenter
from ..cluster_tokenizer import ClusterTokenizer

# (repo_id, split, text_column) -- text_column is renamed to "label" so every
# source ends up with the same {"image", "label"} schema before concatenation.
DATASET_SOURCES = [
    ("SoyVitou/KhmerSynthetic1M", "train", "label"),
    ("seanghay/khmer-hanuman-100k", "train", "text"),
    ("Darayut/khmer-scene-text-synthetic-contrast", "train", "label"),
    ("Darayut/khmer-document-synthetic-low-res", "train", "label"),
]


def load_training_datasets(sources=DATASET_SOURCES, split_override: str | None = None):
    """Loads and concatenates HF datasets into one {"image", "label"} dataset.

    Each source may use a different text column name and carry extra
    metadata columns (e.g. KhmerSynthetic1M's `id`/`file_name`) -- those are
    normalized away so `datasets.concatenate_datasets` (which requires
    matching features across all inputs) can combine them.
    """
    import datasets as hf_datasets

    parts = []
    for repo_id, split, text_column in sources:
        ds = hf_datasets.load_dataset(repo_id, split=split_override or split)
        if text_column != "label":
            ds = ds.rename_column(text_column, "label")
        extra_columns = [c for c in ds.column_names if c not in ("image", "label")]
        if extra_columns:
            ds = ds.remove_columns(extra_columns)
        parts.append(ds)

    return hf_datasets.concatenate_datasets(parts) if len(parts) > 1 else parts[0]


class OCRLineDataset(Dataset):
    """Wraps a HF dataset of {"image": PIL.Image, "label": str} rows,
    applying the same resize->[augment]->chunk->normalize pipeline as
    inference (`ImagePreprocessor`) plus cluster-tokenizer label encoding."""

    def __init__(self, hf_dataset, tokenizer: ClusterTokenizer, config: OCRConfig,
                 augment: bool = False):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.preprocessor = ImagePreprocessor(config)
        self.augmenter = OCRAugmenter() if augment else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image = example["image"]
        text = example["label"]

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
