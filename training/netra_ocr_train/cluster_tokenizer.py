import json
from pathlib import Path

from .postprocess import clean_text
from .tokenization.dispatch import MultiScriptSegmenter


class ClusterTokenizer:
    """Tokenizer using grapheme-cluster segmentation (Khmer Character Clusters
    for Khmer text, per-character for Latin/digits/other -- see
    ``netra_ocr_train.tokenization``).

    Pointed at a vocab file with no multi-codepoint entries (e.g. the old
    176-token ``char2idx_new.json``), ``encode()`` degrades to exactly
    per-codepoint behavior.
    """

    def __init__(self, char2idx_path: str | Path, segmenter: MultiScriptSegmenter | None = None):
        self.char2idx_path = Path(char2idx_path)
        self.char2idx, self.idx2char = self._load_vocab()
        self.segmenter = segmenter if segmenter is not None else MultiScriptSegmenter()

        # Special Tokens (ensure these match your JSON keys)
        self.sos_idx = self.char2idx.get("<sos>", 1)
        self.eos_idx = self.char2idx.get("<eos>", 2)
        self.pad_idx = self.char2idx.get("<pad>", 0)
        self.unk_idx = self.char2idx.get("<unk>", 3)

    def _load_vocab(self) -> tuple[dict, dict]:
        if not self.char2idx_path.exists():
            raise FileNotFoundError(f"Vocab file not found: {self.char2idx_path}")

        with open(self.char2idx_path, 'r', encoding='utf-8') as f:
            char2idx = json.load(f)

        idx2char = {v: k for k, v in char2idx.items()}
        return char2idx, idx2char

    def encode(self, text: str) -> list[int]:
        ids = [self.sos_idx]
        for piece in self.segmenter.segment(text):
            ids.extend(self._encode_piece(piece))
        ids.append(self.eos_idx)
        return ids

    def _encode_piece(self, piece: str) -> list[int]:
        if piece in self.char2idx:
            return [self.char2idx[piece]]
        # OOV cluster: decompose to codepoints, look up each individually.
        # <unk> only for codepoints absent from the base vocab entirely
        # (i.e. genuinely foreign/unregistered scripts).
        return [self.char2idx.get(ch, self.unk_idx) for ch in piece]

    def decode(self, token_ids: list[int]) -> str:
        """Converts a list of token IDs back to a string."""
        result = []
        for idx in token_ids:
            if idx == self.sos_idx or idx == self.pad_idx:
                continue
            if idx == self.eos_idx:
                break
            if idx == self.unk_idx:
                result.append(" ")
                continue
            char = self.idx2char.get(idx, "")
            # Skip any other special token that maps back to a "<...>" string
            # so it is never emitted literally.
            if len(char) > 1 and char.startswith("<") and char.endswith(">"):
                result.append(" ")
                continue
            result.append(char)
        return clean_text("".join(result))

    def __len__(self):
        return len(self.char2idx)
