import pytest

from netra_ocr import models


def test_bundle_sources(monkeypatch):
    monkeypatch.delenv("NETRA_BUNDLES", raising=False)
    assert models.source("ar") == "hf://Netra-Lab/Netra-OCR/recognition"
    assert models.source("blockwise") == "hf://Netra-Lab/Netra-OCR/recognition-blockwise"
    monkeypatch.setenv("NETRA_BUNDLES", "/srv/bundles")
    assert models.source("lines") == "/srv/bundles/lines"
    assert models.source("layout") == "/srv/bundles/layout"
    with pytest.raises(ValueError):
        models.source("beam")


def test_describe_device(monkeypatch):
    monkeypatch.setattr(models, "device", lambda: "cpu")
    assert models.describe_device() == "CPU"
    monkeypatch.setattr(models, "device", lambda: "cuda:0")
    monkeypatch.setattr(models, "_gpu_model", lambda: "NVIDIA GeForce RTX 3060")
    assert models.describe_device() == "GPU (NVIDIA GeForce RTX 3060)"
