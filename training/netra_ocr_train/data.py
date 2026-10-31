"""Training data: text-line images and their labels, from your own files or
published datasets, prepared into exactly the tensors ``model.KhmerOCR.forward()``
expects (see preprocessor.py).

A data *spec* (``--data``, ``--val-data``, ``--test``) is any of:

    khob, printed-words, synthetic   a preset (PRESETS below)
    my_lines/                        a folder: images/ + labels/, or images with a .txt / .gt.txt beside each
    my_lines/labels.tsv              a manifest: .tsv/.txt (path<TAB>text), .csv or .jsonl
    user/name[:split[:text_column]]  a Hugging Face dataset; the split may be a slice, e.g. train[:5000]

Every source becomes rows of ``{"image": PIL.Image, "label": str}``. FORMATS
(printed by the scripts on a bad spec) shows each layout.
"""

import csv
import json
import logging
import random
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Subset

from .augment import OCRAugmenter
from .cluster_tokenizer import ClusterTokenizer
from .config import OCRConfig
from .metrics import IMAGE_SUFFIXES
from .preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)

# Invisible in the image, so the model can't learn them: zero-width space (a
# common Khmer word separator in typed text) and the byte-order mark.
_INVISIBLE = str.maketrans("", "", "​﻿")


def clean_label(text: str) -> str:
    return text.translate(_INVISIBLE).strip()

# The public datasets the published model was trained on. KhmerSynthetic1M is
# licensed for research only (no commercial use).
DEFAULT_TRAIN_DATA = [
    "SoyVitou/KhmerSynthetic1M",
    "seanghay/khmer-hanuman-100k:train:text",
    "Netra-Lab/khmer-scene-text-synthetic-contrast",
    "Netra-Lab/khmer-document-synthetic-low-res",
]

PRESETS = {
    "synthetic": DEFAULT_TRAIN_DATA,
    "khob": ["Netra-Lab/khob-eval"],
    "printed-words": ["Netra-Lab/khmer-printed-words"],
}

FORMATS = """\
Data can be given as:
  a folder         my_lines/images/0001.png  +  my_lines/labels/0001.txt (the line's text)
                   or my_lines/0001.png  +  my_lines/0001.txt (or 0001.gt.txt) side by side
  a manifest       labels.tsv    images/0001.png<TAB>the line's text
  (paths relative  labels.csv    a header row: image,text (or path/file_name, label/transcription)
   to the file)    labels.jsonl  {"image": "images/0001.png", "text": "the line's text"}
  Hugging Face     user/name   user/name:validation   user/name:train[:5000]:transcription
  a preset         """ + ", ".join(PRESETS) + "\n"

PATH_KEYS = ("image", "path", "file_name", "filename", "file", "img", "image_path")
TEXT_KEYS = ("text", "label", "transcription", "ground_truth", "gt", "sentence")
_HF_SPEC = re.compile(r"^(?P<repo>[\w.\-]+/[\w.\-]+)(?::(?P<split>[\w.\-]+(?:\[[^\]]*\])?))?(?::(?P<column>[\w.\-]+))?$")


class DataError(ValueError):
    """A data spec that can't be read; the message says what's expected."""


class LocalLines(Dataset):
    """Line images on disk with their labels, opened lazily.

    ``problems`` lists what was skipped while reading (an image without a
    label, a manifest row pointing at a missing file, an empty label).
    """

    def __init__(self, paths: list[Path], labels: list[str], source: str, kind: str, problems: list[str]):
        if not paths:
            raise DataError(f"No labelled lines in {source}" + (f" ({problems[0]})" if problems else "")
                            + "\n\n" + FORMATS)
        self.paths, self.labels, self.source, self.kind, self.problems = paths, labels, source, kind, problems

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.paths[idx]).convert("RGB")
        except Exception as e:
            raise DataError(f"Can't read {self.paths[idx]}: {e}  (bash training/check_data.sh finds these)") from None
        return {"image": image, "label": self.labels[idx]}


def _read_label(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split("\n")).strip()


def _from_folder(folder: Path) -> LocalLines:
    paths, labels, problems = [], [], []
    if (folder / "images").is_dir():
        kind, images, label_dir = "folder: images/ + labels/", folder / "images", folder / "labels"
        candidates = lambda img: [label_dir / f"{img.stem}.txt"]
    else:
        kind, images = "folder: image + .txt beside it", folder
        candidates = lambda img: [img.with_suffix(".gt.txt"), img.with_suffix(".txt")]
    for img in sorted(images.iterdir()):
        if img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = next((c for c in candidates(img) if c.is_file()), None)
        if label_path is None:
            problems.append(f"no label for {img.name}")
            continue
        text = _read_label(label_path)
        if not text:
            problems.append(f"empty label {label_path.name}")
            continue
        paths.append(img)
        labels.append(text)
    return LocalLines(paths, labels, str(folder), kind, problems)


def _pick(row: dict, keys: tuple, what: str, where: str):
    for key in keys:
        if key in row:
            return row[key]
    raise DataError(f"{where}: no {what} column (looked for {', '.join(keys)}; found {', '.join(row) or 'none'})"
                    "\n\n" + FORMATS)


def _from_manifest(manifest: Path) -> LocalLines:
    suffix = manifest.suffix.lower()
    rows = []
    with open(manifest, encoding="utf-8-sig", newline="") as f:
        if suffix == ".jsonl":
            for n, line in enumerate(f, 1):
                if line.strip():
                    item = json.loads(line)
                    rows.append((n, _pick(item, PATH_KEYS, "image path", manifest.name),
                                 _pick(item, TEXT_KEYS, "text", manifest.name)))
        elif suffix == ".csv":
            for n, item in enumerate(csv.DictReader(f), 2):
                rows.append((n, _pick(item, PATH_KEYS, "image path", manifest.name),
                             _pick(item, TEXT_KEYS, "text", manifest.name)))
        elif suffix in (".tsv", ".txt"):
            for n, line in enumerate(f, 1):
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                if "\t" not in line:
                    raise DataError(f"{manifest.name} line {n} has no tab between the image path and the text"
                                    "\n\n" + FORMATS)
                rows.append((n, *line.split("\t", 1)))
        else:
            raise DataError(f"{manifest}: not a manifest (.tsv, .txt, .csv or .jsonl)\n\n" + FORMATS)

    paths, labels, problems = [], [], []
    for n, image, text in rows:
        path = Path(image) if Path(image).is_absolute() else manifest.parent / image
        text = " ".join(str(text or "").split("\n")).strip()
        if not path.is_file():
            problems.append(f"line {n}: no file {image}")
        elif not text:
            problems.append(f"line {n}: empty text for {image}")
        else:
            paths.append(path)
            labels.append(text)
    return LocalLines(paths, labels, str(manifest), f"manifest: {suffix[1:].upper()}", problems)


def _load_hf(spec: str):
    import datasets as hf_datasets

    match = _HF_SPEC.match(spec)
    repo_id, split, column = match["repo"], match["split"] or "train", match["column"]
    try:
        ds = hf_datasets.load_dataset(repo_id, split=split)
    except (FileNotFoundError, ValueError) as e:
        raise DataError(f"Couldn't load the Hugging Face dataset {repo_id} [{split}]: {e}") from None
    image_column = next((name for name, feature in ds.features.items()
                         if isinstance(feature, hf_datasets.Image)), None)
    if image_column is None:
        raise DataError(f"{repo_id} has no image column (columns: {', '.join(ds.column_names)})")
    if column is None:
        column = next((key for key in TEXT_KEYS if key in ds.column_names), None)
        if column is None:
            raise DataError(f"{repo_id}: which column is the text? Columns: {', '.join(ds.column_names)}. "
                            f"Give it as {repo_id}:{split}:COLUMN")
    elif column not in ds.column_names:
        raise DataError(f"{repo_id} has no column '{column}' (columns: {', '.join(ds.column_names)})")
    ds = ds.select_columns([image_column, column])
    if image_column != "image":
        ds = ds.rename_column(image_column, "image")
    if column != "label":
        ds = ds.rename_column(column, "label")
    ds.source, ds.kind = spec, "Hugging Face dataset"
    return ds


def resolve(spec: str) -> list[str]:
    """A spec with presets expanded (a local path wins over a preset name)."""
    if not Path(spec).expanduser().exists() and spec in PRESETS:
        return list(PRESETS[spec])
    return [spec]


def load_source(spec: str):
    """One spec (not a preset) as rows of {"image", "label"}."""
    path = Path(spec).expanduser()
    if path.is_dir():
        return _from_folder(path)
    if path.is_file():
        return _from_manifest(path)
    if _HF_SPEC.match(spec):
        return _load_hf(spec)
    raise DataError(f"'{spec}' isn't a folder, a manifest file, a preset or a Hugging Face dataset id.\n\n"
                    + FORMATS)


def load_lines(specs: str | list[str]) -> Dataset:
    """Rows of {"image", "label"} from one or more data specs, concatenated."""
    parts = []
    for spec in [specs] if isinstance(specs, str) else specs:
        for source in resolve(spec):
            part = load_source(source)
            logger.info(f"{source}: {len(part):,} lines ({getattr(part, 'kind', 'dataset')})")
            problems = getattr(part, "problems", [])
            if problems:
                logger.warning(f"{source}: skipped {len(problems)} ({'; '.join(problems[:3])}"
                               + (", ..." if len(problems) > 3 else "") + ")")
            parts.append(part)
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def split_off(rows: Dataset, fraction: float = 0.05, minimum: int = 50, maximum: int = 500,
              seed: int = 0) -> tuple[Dataset, Dataset]:
    """(train, validation): a fixed random ``fraction`` of ``rows`` held out,
    at least ``minimum`` and at most ``maximum`` lines, but never more than a fifth."""
    n_val = min(maximum, max(minimum, int(len(rows) * fraction)), len(rows) // 5)
    if n_val < 1:
        raise DataError(f"Only {len(rows)} lines: too few to hold some out for validation. "
                        "Add more, or pass --val-data")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return Subset(rows, order[n_val:]), Subset(rows, order[:n_val])


def labels_of(rows, limit: int | None = None) -> list[str]:
    """Up to ``limit`` labels, without decoding any image."""
    limit = len(rows) if limit is None else limit
    if isinstance(rows, LocalLines):
        return rows.labels[:limit]
    if isinstance(rows, ConcatDataset):
        out = []
        for part in rows.datasets:
            out += labels_of(part, limit - len(out))
            if len(out) >= limit:
                break
        return out
    return rows.select(range(min(limit, len(rows))))["label"]   # a Hugging Face dataset


def unknown_characters(rows: Dataset, tokenizer: ClusterTokenizer, limit: int = 20000) -> dict[str, int]:
    """Characters in (up to ``limit``) labels the vocabulary can't encode, with
    counts. ``rows`` comes straight from `load_lines` (not a split)."""
    counts = {}
    for label in labels_of(rows, limit):
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
