import pytest

from netra_ocr import runtime_engine


def test_engine_selection(monkeypatch):
    monkeypatch.delenv("NETRA_ENGINE", raising=False)
    assert runtime_engine.engine() == "torch"
    monkeypatch.setenv("NETRA_ENGINE", "runtime")
    assert runtime_engine.engine() == "runtime"
    monkeypatch.setenv("NETRA_ENGINE", "tensorflow")
    with pytest.raises(ValueError):
        runtime_engine.engine()


def test_runtime_engine_needs_bundles_dir(monkeypatch):
    pytest.importorskip("netra_runtime")
    monkeypatch.delenv("NETRA_BUNDLES", raising=False)
    runtime_engine.load.cache_clear()
    with pytest.raises(RuntimeError, match="NETRA_BUNDLES"):
        runtime_engine.load("ar")
