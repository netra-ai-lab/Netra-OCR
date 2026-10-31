"""The real models on the pages in test_images/, against known-good text.

Run with ``pytest --slow``; the models download on the first run. The golden
text is the CPU output (GPU arithmetic differs slightly). After a deliberate
change to the models or the pipeline, regenerate it and review the diff:

    NETRA_DEVICE=cpu python tests/test_real_models.py
"""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted((ROOT / "test_images").iterdir())
GOLDEN = Path(__file__).parent / "golden" / "pages_cpu.json"


def page_texts(decoder: str = "ar") -> dict[str, str]:
    from netra_ocr.exporters import export
    from netra_ocr.ocr_engine import KhmerOCRPipeline
    pipe = KhmerOCRPipeline(layout=True, decoder=decoder)
    return {page.name: export(pipe.process_document(str(page)), "txt")[0].decode() for page in PAGES}


@pytest.fixture(scope="module")
def cpu(monkeypatch_module):
    monkeypatch_module.setenv("NETRA_DEVICE", "cpu")
    from netra_ocr import models
    models._loaded.clear()
    yield
    models._loaded.clear()


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as patch:
        yield patch


@pytest.mark.slow
@pytest.mark.parametrize("decoder", ["ar", "blockwise"])
def test_pages_match_golden_text(cpu, decoder):
    expected = json.loads(GOLDEN.read_text())
    got = page_texts(decoder)
    assert sorted(got) == sorted(expected)
    for name in expected:
        assert got[name] == expected[name], f"{name} ({decoder}) changed"


if __name__ == "__main__":
    os.environ.setdefault("NETRA_DEVICE", "cpu")
    GOLDEN.write_text(json.dumps(page_texts(), ensure_ascii=False, indent=1) + "\n")
    print("wrote", GOLDEN)
