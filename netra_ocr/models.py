"""Netra OCR's models, run on Netra Runtime (ONNX Runtime; no PyTorch).

Each model is a Netra Runtime bundle. By default they come from the
Hugging Face repo ``Netra-Lab/Netra-OCR`` (one folder per bundle, downloaded
once into the Hugging Face cache). ``NETRA_BUNDLES`` points at a local folder
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
BUNDLES = {
    "lines": "lines",
    "layout": "layout",
    "ar": "recognition",
    "blockwise": "recognition-blockwise",
}
DECODERS = ("ar", "blockwise")

_loaded: dict = {}


def source(key: str) -> str:
    """Where the bundle for ``key`` (see BUNDLES) is loaded from."""
    if key not in BUNDLES:
        raise ValueError(f"Unknown model '{key}'. Available: {list(BUNDLES)}")
    root = os.environ.get("NETRA_BUNDLES")
    return str(Path(root) / BUNDLES[key]) if root else f"{MODEL_REPO}/{BUNDLES[key]}"


def load(key: str):
    """The loaded Netra Runtime task for ``key``, shared by every pipeline."""
    if key not in _loaded:
        import netra_runtime

        _loaded[key] = netra_runtime.load(source(key), device=os.environ.get("NETRA_DEVICE") or "auto")
        logger.info("Loaded %s", _loaded[key])
    return _loaded[key]


def device() -> str:
    """The device the models run on: the loaded ones' (a GPU that fails to load
    falls back to "cpu"), else the one that would be picked."""
    if _loaded:
        return next(iter(_loaded.values())).device
    from netra_runtime.backends import resolve_device
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
