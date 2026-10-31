import pytest

from netra_ocr import models


def test_bundle_sources(monkeypatch, tmp_path):
    monkeypatch.delenv("NETRA_BUNDLES", raising=False)
    assert models.source("ar") == "hf://Netra-Lab/Netra-OCR/recognition@" + models.MODEL_REVISION
    assert models.source("blockwise") == "hf://Netra-Lab/Netra-OCR/recognition-blockwise@" + models.MODEL_REVISION
    for folder in ("lines", "layout"):
        (tmp_path / folder).mkdir()
    monkeypatch.setenv("NETRA_BUNDLES", str(tmp_path))
    assert models.source("lines") == str(tmp_path / "lines")
    assert models.source("layout") == str(tmp_path / "layout")
    with pytest.raises(ValueError):
        models.source("beam")


def test_partial_bundles_fall_back_to_hugging_face(monkeypatch, tmp_path):
    # A fine-tuned recognizer alone (training/export.sh): the detectors still come from HF.
    (tmp_path / "recognition").mkdir()
    monkeypatch.setenv("NETRA_BUNDLES", str(tmp_path))
    assert models.source("ar") == str(tmp_path / "recognition")
    assert models.source("lines") == "hf://Netra-Lab/Netra-OCR/lines@" + models.MODEL_REVISION


def test_describe_device(monkeypatch):
    monkeypatch.setattr(models, "device", lambda: "cpu")
    assert models.describe_device() == "CPU"
    monkeypatch.setattr(models, "device", lambda: "cuda:0")
    monkeypatch.setattr(models, "_gpu_model", lambda: "NVIDIA GeForce RTX 3060")
    assert models.describe_device() == "GPU (NVIDIA GeForce RTX 3060)"
