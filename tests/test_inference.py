import json
import os

import numpy as np
import pytest
from PIL import Image

from netra_ocr.inference import hub
from netra_ocr.inference.detector import (decode_yolo_end2end, decode_yolo_raw, letterbox, nms,
                                          resize_bilinear, restore_boxes)
from netra_ocr.inference.recognizer import (BlockwiseDecoder, GreedyDecoder, Recognizer, Vocab,
                                            chunk_line, clean_text)

CHUNKING = dict(height=48, chunk_width=100, chunk_overlap=16, min_width=50)


def write_model(root, variants=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "encoder.onnx").write_bytes(b"graph")
    (root / "vocab.json").write_text(json.dumps({"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3,
                                                 "ក": 4, "ា": 5, "<mask>": 6}))
    config = {"name": "toy", "version": "0", "graphs": {"encoder": "encoder.onnx"},
              "files": {"encoder.onnx": "x", "vocab.json": "x"}}
    if variants:
        config["variants"] = variants
        for spec in variants.values():
            for name in spec["graphs"].values():
                (root / name).write_bytes(b"graph")
                config["files"][name] = "x"
    (root / hub.CONFIG_NAME).write_text(json.dumps(config))
    return root


def test_line_chunks_shapes_and_padding():
    chunks = chunk_line(Image.new("L", (200, 24), color=0), **CHUNKING)   # -> 400 x 48
    # windows start every 84 px across 400 px: 0, 84, ..., 336 -> 5 chunks
    assert chunks.shape == (5, 1, 48, 100) and chunks.dtype == np.float32
    assert chunks[0].max() == -1.0                    # black, normalized
    assert chunks[-1, 0, 0, -1] == 1.0                # right padding is white
    assert chunk_line(np.zeros((10, 5, 3), np.uint8), **CHUNKING).shape == (1, 1, 48, 100)  # min_width


def test_vocab_decode(tmp_path):
    vocab = Vocab(write_model(tmp_path / "m") / "vocab.json")
    assert vocab.decode([1, 4, 5, 3, 6, 4, 2, 4]) == "កា  ក"


def test_text_cleanup():
    assert clean_text("kkkkk") == "k"
    assert clean_text("H H H H") == "H"
    assert clean_text("P<KHMNHEM<<DARAYUT<<<<<<<<<<<<") == "P<KHMNHEM<<DARAYUT<<<<<<<<<<<<"


def test_letterbox_and_restore_round_trip():
    tensor, info = letterbox(np.zeros((300, 1000, 3), np.uint8), imgsz=640)
    # long side -> 640, short side 192 already a multiple of 32: no padding needed
    assert tensor.shape == (1, 3, 192, 640)
    tensor, info = letterbox(np.zeros((310, 1000, 3), np.uint8), imgsz=640)   # 198 -> padded to 224
    assert tensor.shape == (1, 3, 224, 640) and info.pad_y == 13
    assert tensor[0, 0, 0, 0] == np.float32(114 / 255)
    assert letterbox(np.zeros((310, 1000, 3), np.uint8), imgsz=640, auto=False)[0].shape == (1, 3, 640, 640)
    box = np.array([[100.0, 50.0, 400.0, 300.0]])
    letterboxed = box * [info.gain_x, info.gain_y, info.gain_x, info.gain_y] + [info.pad_x, info.pad_y] * 2
    np.testing.assert_allclose(restore_boxes(letterboxed, info), box, atol=1e-3)


def test_resize_close_to_opencv():
    cv2 = pytest.importorskip("cv2")
    image = np.random.default_rng(0).integers(0, 256, (97, 203, 3), dtype=np.uint8)
    for size in [(150, 61), (640, 300), (203, 97)]:
        reference = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        assert np.abs(resize_bilinear(image, *size).astype(int) - reference).max() <= 1


def test_nms_and_yolo_outputs():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], np.float32)
    assert list(nms(boxes, np.array([0.9, 0.8, 0.7]), iou=0.5)) == [0, 2]
    # raw output: (1, 4 + 2 classes, 3 anchors) in cx, cy, w, h; anchors 0 and 1 overlap but differ in class
    raw = np.array([[[5, 5.5, 25], [5, 5.5, 25], [10, 9, 10], [10, 9, 10],
                     [0.9, 0.1, 0.1], [0.1, 0.8, 0.6]]], np.float32)
    b, s, c = decode_yolo_raw(raw, conf=0.25, iou=0.5)
    assert list(c) == [0, 1, 1] and len(b) == 3            # class-aware: no suppression across classes
    assert len(decode_yolo_raw(raw, conf=0.25, iou=0.5, agnostic=True)[0]) == 2
    e2e = np.array([[[0, 0, 5, 5, 0.9, 1], [0, 0, 5, 5, 0.1, 0]]], np.float32)
    assert list(decode_yolo_end2end(e2e, conf=0.25)[2]) == [1]


class _ScriptedStep:
    """Fake decoder_step: the 'model' always predicts ``target[pos + 1]`` and its
    proposal heads guess ``target[pos + 2 + j]``, except at positions in ``wrong``."""

    def __init__(self, target, block_size, wrong=()):
        self.target, self.block_size, self.wrong, self.vocab = target, block_size, set(wrong), 16

    def run(self, inputs, keep=()):
        tokens, past = inputs["tokens"], inputs["past_k"]
        P, n = past.shape[3], tokens.shape[1]
        logits = np.zeros((1, n, self.vocab), np.float32)
        proposals = np.zeros((1, n, self.block_size - 1), np.int64)
        for i in range(n):
            pos = P + i
            logits[0, i, self.target[min(pos + 1, len(self.target) - 1)]] = 5.0
            for j in range(self.block_size - 1):
                k = min(pos + 2 + j, len(self.target) - 1)
                proposals[0, i, j] = 9 if pos in self.wrong else self.target[k]
        present = np.zeros(past.shape[:3] + (P + n, past.shape[4]), np.float32)
        return {"logits": logits, "present_k": present, "present_v": present, "proposal_ids": proposals}


def test_blockwise_decoding_matches_greedy():
    target = [1, 4, 5, 6, 7, 8, 10, 11, 12, 2]          # <sos> ... <eos>
    args = dict(layers=1, heads=1, head_dim=2, max_len=50)
    mask = np.zeros((1, 3), np.float32)
    greedy = GreedyDecoder(**args).decode(_ScriptedStep(target, 1), None, None, mask, 1, 2)
    for wrong in [(), (0, 3), (1, 2, 3, 4, 5, 6)]:
        got = BlockwiseDecoder(block_size=4, **args).decode(_ScriptedStep(target, 4, wrong), None, None, mask, 1, 2)
        assert got[0][0] == greedy[0][0] == target[:-1]
        np.testing.assert_allclose(got[0][1], greedy[0][1])
    # all guesses right: 9 tokens in ceil(9 / 4) = 3 forward passes
    decoder = BlockwiseDecoder(block_size=4, **args)
    decoder.decode(_ScriptedStep(target, 4), None, None, mask, 1, 2)
    assert (decoder.rounds, decoder.tokens) == (3, 9)

    capped = BlockwiseDecoder(block_size=4, **{**args, "max_len": 4})
    ids, _ = capped.decode(_ScriptedStep(target, 4), None, None, mask, 1, 2)[0]
    assert ids == target[:5]                           # stops at max_len tokens, like greedy


def test_variants_and_paths(tmp_path):
    root = write_model(tmp_path / "m", {"fp16": {"graphs": {"encoder": "encoder.fp16.onnx"},
                                                 "devices": ["cuda", "tensorrt"]}})
    config = hub.read_config(root)
    assert hub.pick_variant(config, "cuda") == "fp16" and hub.pick_variant(config, "cpu") is None
    assert hub.graph_files(config, "fp16") == {"encoder": "encoder.fp16.onnx"}
    for bad in ("../x", "/etc/passwd", ""):
        with pytest.raises(ValueError):
            hub.model_file(root, bad)


def test_model_files_may_be_symlinks(tmp_path):
    # The Hugging Face cache stores each file as a symlink to a shared blob.
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / "abc").write_bytes(b"graph")
    (tmp_path / "snap").mkdir()
    (tmp_path / "snap" / "model.onnx").symlink_to(tmp_path / "blobs" / "abc")
    assert hub.model_file(tmp_path / "snap", "model.onnx").read_bytes() == b"graph"


def test_fetch_downloads_only_what_the_device_runs(monkeypatch, tmp_path):
    import huggingface_hub
    write_model(tmp_path / "lines", {"fp16": {"graphs": {"encoder": "encoder.fp16.onnx"}, "devices": ["cuda"]}})
    calls = []

    def fake_snapshot(repo_id, revision=None, allow_patterns=None):
        calls.append((repo_id, revision, allow_patterns))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    assert hub.fetch("hf://Netra-Lab/Netra-OCR/lines@v1") == tmp_path / "lines"
    assert calls[-1] == ("Netra-Lab/Netra-OCR", "v1", ["lines/*"])
    assert hub.fetch("local/dir") == hub.Path("local/dir")

    calls.clear()
    hub.fetch("hf://Netra-Lab/Netra-OCR/lines", "cpu")
    assert calls[0][2] == ["lines/netra-model.json"]
    assert sorted(calls[1][2]) == ["lines/encoder.onnx", "lines/netra-model.json", "lines/vocab.json"]
    calls.clear()
    hub.fetch("hf://Netra-Lab/Netra-OCR/lines", "cuda")
    assert sorted(calls[1][2]) == ["lines/encoder.fp16.onnx", "lines/netra-model.json", "lines/vocab.json"]


@pytest.mark.skipif(not os.environ.get("NETRA_TEST_RECOGNIZER"),
                    reason="set NETRA_TEST_RECOGNIZER to a recognizer model folder")
def test_real_recognizer_runs():
    result = Recognizer(os.environ["NETRA_TEST_RECOGNIZER"], device="cpu")(Image.new("L", (300, 48), color=255))
    assert isinstance(result.text, str) and 0.0 <= result.confidence <= 1.0
