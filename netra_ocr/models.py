"""Netra OCR's models, run on ONNX Runtime by ``netra_ocr.inference`` (no PyTorch).

Each model is a folder of ONNX graphs plus its settings (``netra-model.json``).
By default they come from the Hugging Face repo ``Netra-Lab/Netra-OCR`` at commit
``MODEL_REVISION`` (one folder per model, downloaded once into the Hugging Face
cache). ``NETRA_BUNDLES`` points at a local folder
with the same layout instead:

    lines/                   YOLO text-line and logo detector
    layout/                  DocLayout-YOLO page layout (layout=True)
    recognition/             recognizer, decoder="ar"
    recognition-blockwise/   recognizer, decoder="blockwise"

``NETRA_DEVICE`` picks the device ("cpu", "cuda", ...; unset: GPU if usable).
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_REPO = "hf://Netra-Lab/Netra-OCR"
# The commit of the model files this package was tested with, so an upload to
# the repo's main branch doesn't change anyone's results until a release moves it.
MODEL_REVISION = "ada0c6014c147a15a6b4e11428e3fc36c5a52782"
BUNDLES = {
    "lines": "lines",
    "layout": "layout",
    "ar": "recognition",
    "blockwise": "recognition-blockwise",
}
DECODERS = ("ar", "blockwise")

_loaded: dict = {}


def source(key: str) -> str:
    """Where the model for ``key`` (see BUNDLES) is loaded from."""
    if key not in BUNDLES:
        raise ValueError(f"Unknown model '{key}'. Available: {list(BUNDLES)}")
    root = os.environ.get("NETRA_BUNDLES")
    return str(Path(root) / BUNDLES[key]) if root else f"{MODEL_REPO}/{BUNDLES[key]}@{MODEL_REVISION}"


def load(key: str):
    """The loaded recognizer or detector for ``key``, shared by every pipeline."""
    if key not in _loaded:
        from .inference import Detector, Recognizer

        model = Recognizer if key in DECODERS else Detector
        _loaded[key] = model(source(key), device=os.environ.get("NETRA_DEVICE") or "auto")
        logger.info("Loaded %s", _loaded[key])
    return _loaded[key]


def device() -> str:
    """The device the models run on: the loaded ones' (a GPU that fails to load
    falls back to "cpu"), else the one that would be picked."""
    if _loaded:
        return next(iter(_loaded.values())).device
    from .inference import resolve_device
    return resolve_device(os.environ.get("NETRA_DEVICE") or "auto")


def describe_device() -> str:
    """Human-readable device for logs and the UI, e.g. "GPU (NVIDIA GeForce RTX 3060)"."""
    name = device()
    if name.startswith(("cuda", "tensorrt")):
        return f"GPU ({_gpu_model()})" if _gpu_model() else "GPU"
    if name.startswith("openvino"):
        return "OpenVINO"
    return "CPU"


def _gpu_model() -> str | None:
    # The NVIDIA driver lists its GPUs here on Linux; ONNX Runtime has no API for the name.
    for info in sorted(Path("/proc/driver/nvidia/gpus").glob("*/information")):
        for line in info.read_text().splitlines():
            if line.startswith("Model:"):
                return line.split(":", 1)[1].strip()
    return None
