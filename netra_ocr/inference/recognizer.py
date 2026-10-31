"""Text-line recognizer: one line image -> Khmer/English text.

Two ONNX graphs, exported from the PyTorch model by
``training/netra_ocr_train/scripts/export_runtime_bundle.py``:

    encoder       chunks (N, 1, height, chunk_width) -> cross_k, cross_v
                  [layers, 1, heads, S, head_dim]: the decoder's attention
                  keys/values over the image, computed once per line
    decoder_step  tokens [B, n], past_k/past_v [layers, B, heads, P, head_dim],
                  cross_k/cross_v [layers, B, heads, S, head_dim],
                  cross_mask [B, S] (0 = real, -inf = padding)
                  -> logits [B, n, vocab], present_k/present_v (the cache grown
                  by n positions), and for blockwise models proposal_ids
                  [B, n, block_size - 1]

Pre-processing (``chunk_line``), decoding and text cleanup live here; their
settings come from the model folder's ``netra-model.json``.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import hub


@dataclass
class Text:
    text: str
    # Mean probability of the generated tokens (end-of-sequence included): a
    # cheap signal for flagging lines a human should double-check.
    confidence: float


def to_pil(image) -> Image.Image:
    """A path, a PIL image, or a uint8 numpy array (H x W gray or H x W x 3 RGB)."""
    if isinstance(image, (str, Path)):
        if not Path(image).exists():
            raise FileNotFoundError(f"Image not found: {image}")
        return Image.open(image)
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    raise TypeError(f"Unsupported image input: {type(image).__name__}")


# ── pre-processing ───────────────────────────────────────────────────────────

def chunk_line(image, height: int, chunk_width: int, chunk_overlap: int, min_width: int,
               pad_value: float = 1.0, mean: float = 0.5, std: float = 0.5, **_) -> np.ndarray:
    """Text-line image -> overlapping chunks ``(N, 1, height, chunk_width)`` float32.

    Grayscale, resize to ``height`` keeping the aspect ratio (never narrower
    than ``min_width``), scale to [0, 1], cut ``chunk_width``-wide windows every
    ``chunk_width - chunk_overlap`` pixels (the last one padded with
    ``pad_value``), then normalize ``(x - mean) / std`` -- the same contract the
    model was trained with.
    """
    image = to_pil(image).convert("L")
    width = max(min_width, int(height * image.width / image.height))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    chunks = []
    for start in range(0, width, chunk_width - chunk_overlap):
        chunk = pixels[:, start:start + chunk_width]
        if chunk.shape[1] < chunk_width:
            chunk = np.pad(chunk, ((0, 0), (0, chunk_width - chunk.shape[1])), constant_values=pad_value)
        chunks.append(chunk)

    batch = np.stack(chunks)[:, None, :, :]
    return ((batch - mean) / std).astype(np.float32)


# ── tokens -> text ───────────────────────────────────────────────────────────

class Vocab:
    """Token id -> string, from a ``{"token": id}`` JSON file. ``<unk>`` and any
    other ``<...>`` special token become a space, so they never leak into the text."""

    def __init__(self, path: Path, sos="<sos>", eos="<eos>", pad="<pad>", unk="<unk>"):
        token_to_id = json.loads(path.read_text(encoding="utf-8"))
        self.id_to_token = {i: t for t, i in token_to_id.items()}
        self.sos_id = token_to_id[sos]
        self.eos_id = token_to_id[eos]
        self.pad_id = token_to_id[pad]
        self.unk_id = token_to_id[unk]

    def decode(self, ids: list[int]) -> str:
        pieces = []
        for i in ids:
            if i in (self.sos_id, self.pad_id):
                continue
            if i == self.eos_id:
                break
            token = self.id_to_token.get(i, "")
            if i == self.unk_id or (len(token) > 1 and token.startswith("<") and token.endswith(">")):
                pieces.append(" ")
            else:
                pieces.append(token)
        return "".join(pieces)


_NOISE_CHARS = re.compile(r"[�\x00-\x08\x0b\x0c\x0e-\x1f]")
# A 2-20 char cluster repeated inside a word, e.g. "រាំរាំ" -> "រាំ".
_REPEAT_CLUSTER = re.compile(r"(\S{2,20}?)\1{1,}")
# The same whitespace-separated token repeated, e.g. "H H H" -> "H".
_REPEAT_TOKEN = re.compile(r"(\S+)(?:\s+\1)+")
_MRZ_CHARS = re.compile(r"[A-Z0-9<]")


def _is_mrz_line(text: str) -> bool:
    stripped = text.replace(" ", "")
    if len(stripped) < 10 or "<<" not in stripped:
        return False
    return len(_MRZ_CHARS.findall(stripped)) / len(stripped) >= 0.85


def clean_text(text: str, max_repeat: int = 3) -> str:
    """Remove decoder artifacts: control characters, runaway repeats of a
    character / cluster / word, extra whitespace. Passport MRZ lines keep their
    legitimate ``<<<`` filler."""
    text = _NOISE_CHARS.sub("", text)
    if _is_mrz_line(text):
        return re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"(.)\1{" + str(max_repeat) + r",}", r"\1", text)
    for _ in range(3):
        collapsed = _REPEAT_CLUSTER.sub(r"\1", text)
        if collapsed == text:
            break
        text = collapsed
    text = _REPEAT_TOKEN.sub(r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# ── decoding ─────────────────────────────────────────────────────────────────

def _probability(logits: np.ndarray, token: int) -> float:
    logits = logits.astype(np.float64)
    return float(1.0 / np.exp(logits - logits[token]).sum())


class GreedyDecoder:
    """Greedy decoding with a KV cache: each step feeds only the newest token.

    A batch decodes its lines in lockstep: a line that has emitted ``<eos>``
    keeps being fed (its output ignored) until every line is done, so batches
    work best when their lines have similar lengths. The cache tensors stay on
    the device between steps; only the last logits row is copied back.
    """

    def __init__(self, layers: int, heads: int, head_dim: int, max_len: int, **_):
        self.layers = layers
        self.heads = heads
        self.head_dim = head_dim
        self.max_len = max_len

    def decode(self, step, cross_k, cross_v, cross_mask: np.ndarray, sos_id: int, eos_id: int
               ) -> list[tuple[list[int], list[float]]]:
        """Per line: the generated ids (starting with ``sos_id``) and each new token's probability."""
        batch = cross_mask.shape[0]
        empty = np.zeros((self.layers, batch, self.heads, 0, self.head_dim), dtype=np.float32)
        past_k, past_v = empty, empty
        ids = [[sos_id] for _ in range(batch)]
        probs = [[] for _ in range(batch)]
        running = np.ones(batch, dtype=bool)
        last = np.full((batch, 1), sos_id, dtype=np.int64)

        for _ in range(self.max_len):
            out = step.run(
                {"tokens": last, "past_k": past_k, "past_v": past_v,
                 "cross_k": cross_k, "cross_v": cross_v, "cross_mask": cross_mask},
                keep=("present_k", "present_v"),
            )
            logits = out["logits"][:, -1].astype(np.float64)
            next_ids = logits.argmax(axis=1)
            next_probs = 1.0 / np.exp(logits - logits[np.arange(batch), next_ids][:, None]).sum(axis=1)
            for b in np.flatnonzero(running):
                probs[b].append(float(next_probs[b]))
                if next_ids[b] == eos_id:
                    running[b] = False
                else:
                    ids[b].append(int(next_ids[b]))
            if not running.any():
                break
            last = next_ids[:, None].astype(np.int64)
            past_k, past_v = out["present_k"], out["present_v"]
        return list(zip(ids, probs))


class BlockwiseDecoder(GreedyDecoder):
    """Blockwise parallel decoding (Stern et al. 2018) on top of the KV cache.

    Each round feeds the last accepted token followed by the previous round's
    proposals. The model's own next-token prediction at each of those positions
    verifies them: proposals are accepted up to the first disagreement, then
    the model's prediction there is appended too. Every emitted token is
    therefore exactly what greedy decoding would emit; the proposals only decide
    how many arrive per forward pass. The cache is trimmed back to the accepted
    positions after each round.

    ``rounds`` / ``tokens`` count forward passes and emitted tokens
    (``tokens / rounds`` = mean accepted block size).
    """

    def __init__(self, block_size: int, **options):
        super().__init__(**options)
        self.block_size = block_size
        self.rounds = 0
        self.tokens = 0

    def decode(self, step, cross_k, cross_v, cross_mask: np.ndarray, sos_id: int, eos_id: int):
        if cross_mask.shape[0] == 1:
            return [self._decode_line(step, cross_k, cross_v, cross_mask, sos_id, eos_id)]
        # Lines accept different numbers of tokens per round, so their caches
        # diverge in length: decode a batch one line at a time.
        results = []
        for b in range(cross_mask.shape[0]):
            length = int(np.isfinite(cross_mask[b]).sum())
            results.append(self._decode_line(
                step, cross_k[:, b:b + 1, :, :length], cross_v[:, b:b + 1, :, :length],
                cross_mask[b:b + 1, :length], sos_id, eos_id))
        return results

    def _decode_line(self, step, cross_k, cross_v, cross_mask, sos_id, eos_id):
        empty = np.zeros((self.layers, 1, self.heads, 0, self.head_dim), dtype=np.float32)
        past_k, past_v = empty, empty
        generated, probs, proposals = [sos_id], [], []

        # The cache always holds every generated token except the last one.
        while len(generated) - 1 < self.max_len:
            # Never feed a position past max_len - 1 (the position table's end).
            proposals = proposals[:self.max_len - len(generated)]
            out = step.run({"tokens": np.array([[generated[-1], *proposals]], dtype=np.int64),
                            "past_k": past_k, "past_v": past_v,
                            "cross_k": cross_k, "cross_v": cross_v, "cross_mask": cross_mask})
            logits = out["logits"][0]
            predicted = logits.argmax(axis=-1)
            accepted = 0
            while accepted < len(proposals) and predicted[accepted] == proposals[accepted]:
                accepted += 1
            new_tokens = [*proposals[:accepted], int(predicted[accepted])]
            self.rounds += 1

            cache_len = past_k.shape[3] + accepted + 1
            past_k = out["present_k"][:, :, :, :cache_len]
            past_v = out["present_v"][:, :, :, :cache_len]

            for i, token in enumerate(new_tokens):
                probs.append(_probability(logits[i], token))
                self.tokens += 1
                if token == eos_id:
                    return generated, probs
                generated.append(token)
                if len(generated) - 1 >= self.max_len:
                    return generated, probs
            proposals = [int(t) for t in out["proposal_ids"][0, accepted]]
        return generated, probs


# ── the recognizer ───────────────────────────────────────────────────────────

class Recognizer:
    """Image of one text line -> ``Text``. ``source``: a model folder or ``hf://`` path."""

    def __init__(self, source, device: str | None = None):
        root, config, self.variant, sessions, self.device = hub.load_sessions(source, device)
        self.name = f"{config['name']} {config['version']}"
        settings = config["components"]
        self.chunking = settings["preprocess"]
        decoder = dict(settings["decoder"])
        kind = decoder.pop("type")
        self.decoder = BlockwiseDecoder(**decoder) if kind == "blockwise-kv" else GreedyDecoder(**decoder)
        tokens = dict(settings["tokenizer"])
        tokens.pop("type", None)
        self.vocab = Vocab(hub.model_file(root, tokens.pop("vocab")), **tokens)
        self.max_repeat = next((p.get("max_repeat", 3) for p in settings.get("postprocess", [])
                                if p.get("type") == "ocr-text-cleanup"), None)
        self.encoder = sessions["encoder"]
        self.step = sessions["decoder_step"]
        self._encoder_outputs = tuple(self.encoder.output_names)

    def __repr__(self):
        return f"Recognizer({self.name}, device={self.device}, variant={self.variant or 'fp32'})"

    def preprocess(self, image) -> np.ndarray:
        return chunk_line(image, **self.chunking)

    def __call__(self, image) -> Text:
        encoded = self.encoder.run({"chunks": self.preprocess(image)}, keep=self._encoder_outputs)
        cross_k, cross_v = encoded["cross_k"], encoded["cross_v"]
        mask = np.zeros((1, cross_k.shape[3]), dtype=np.float32)
        [(ids, probs)] = self.decoder.decode(self.step, cross_k, cross_v, mask,
                                             self.vocab.sos_id, self.vocab.eos_id)
        return self._finish(ids, probs)

    def batch(self, images, batch_size: int = 16) -> list[Text]:
        """Recognize many lines, decoding up to ``batch_size`` of them together.

        Each line is encoded on its own (its encoder sequence length depends on
        its width); lines are then sorted by that length and decoded in groups,
        with shorter lines' encoder outputs zero-padded and masked out. Results
        come back in input order.
        """
        encoded = []
        for image in images:
            out = self.encoder.run({"chunks": self.preprocess(image)})
            encoded.append((out["cross_k"], out["cross_v"]))
        order = sorted(range(len(encoded)), key=lambda i: encoded[i][0].shape[3])

        results: list[Text | None] = [None] * len(encoded)
        for start in range(0, len(order), batch_size):
            group = order[start:start + batch_size]
            longest = max(encoded[i][0].shape[3] for i in group)
            ks, vs = [], []
            mask = np.zeros((len(group), longest), dtype=np.float32)
            for row, i in enumerate(group):
                k, v = encoded[i]
                pad = longest - k.shape[3]
                ks.append(np.pad(k, ((0, 0), (0, 0), (0, 0), (0, pad), (0, 0))))
                vs.append(np.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad), (0, 0))))
                mask[row, k.shape[3]:] = -np.inf
            decoded = self.decoder.decode(self.step, np.concatenate(ks, axis=1),
                                          np.concatenate(vs, axis=1), mask,
                                          self.vocab.sos_id, self.vocab.eos_id)
            for i, (ids, probs) in zip(group, decoded):
                results[i] = self._finish(ids, probs)
        return results

    def _finish(self, ids: list[int], probs: list[float]) -> Text:
        text = self.vocab.decode(ids)
        if self.max_repeat is not None:
            text = clean_text(text, self.max_repeat)
        return Text(text=text, confidence=sum(probs) / len(probs) if probs else 0.0)
