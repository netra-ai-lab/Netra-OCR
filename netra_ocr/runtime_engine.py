"""Run Netra-OCR's models on Netra Runtime (ONNX Runtime) instead of PyTorch.

Selected with ``NETRA_ENGINE=runtime``; bundles are read from the folder in
``NETRA_BUNDLES``, which holds one sub-folder per bundle:

    khmer-text-lines/                  YOLO line/logo detector
    doclayout-docstructbench/          DocLayout-YOLO (layout=True)
    khmer-text-recognition/            recognizer, decoder="ar"
    khmer-text-recognition-blockwise/  recognizer, decoder="blockwise"

Only model inference moves: box padding, horizontal refinement, layout
dedupe and page structuring are the same code as the torch engine.
``NETRA_DEVICE`` picks the device for both engines (unset: GPU if usable).
Recognition is greedy (beam_width is ignored) and batched across lines.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .detectors.yolo import YoloDetector
from .layout.doclayout import DocLayoutYoloAnalyzer

logger = logging.getLogger(__name__)

BUNDLES = {
    "lines": "khmer-text-lines",
    "layout": "doclayout-docstructbench",
    "ar": "khmer-text-recognition",
    "blockwise": "khmer-text-recognition-blockwise",
}


def engine() -> str:
    name = os.environ.get("NETRA_ENGINE", "torch")
    if name not in ("torch", "runtime"):
        raise ValueError(f"NETRA_ENGINE must be 'torch' or 'runtime', not '{name}'")
    return name


@lru_cache(maxsize=None)
def load(key: str):
    """The loaded runtime task for ``key`` (see BUNDLES), shared by every pipeline."""
    import netra_runtime

    root = os.environ.get("NETRA_BUNDLES")
    if not root:
        raise RuntimeError("NETRA_ENGINE=runtime needs NETRA_BUNDLES (a folder of Netra Runtime bundles)")
    path = Path(root) / BUNDLES[key]
    task = netra_runtime.load(path, device=os.environ.get("NETRA_DEVICE") or "auto")
    logger.info("Loaded %s", task)
    return task


class RuntimeYoloDetector(YoloDetector):
    """``YoloDetector`` with the model call on Netra Runtime."""

    def __init__(self, conf: float = 0.25, iou: float = 0.7, pad: int = 2, refine: bool = True,
                 gap_ratio: float = 0.8):
        # The bundle's NMS IoU is fixed at export time (0.7, the same default).
        self._model = load("lines")
        self._conf = conf
        self._iou = iou
        self._pad = max(0, int(pad))
        self._refine = refine
        self._gap_ratio = gap_ratio

    def _predict(self, img_bgr):
        found = self._model(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), conf=self._conf)
        boxes = np.array([d.box for d in found], dtype=np.float32).reshape(-1, 4)
        return boxes, np.array([d.class_id for d in found])


class RuntimeLayoutAnalyzer(DocLayoutYoloAnalyzer):
    """``DocLayoutYoloAnalyzer`` with the model call on Netra Runtime."""

    def __init__(self, conf: float = 0.2):
        self._model = load("layout")
        self._conf = conf

    def _predict(self, img_bgr):
        found = self._model(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), conf=self._conf)
        return [(d.box, d.label, d.score) for d in found]


_warned_beam = False


def recognize_batch(crops: list, decoder: str = "ar", beam_width: int = 1, batch_size: int = 16) -> list[str]:
    global _warned_beam
    if beam_width > 1 and not _warned_beam:
        logger.warning("NETRA_ENGINE=runtime decodes greedily; beam_width=%d is ignored", beam_width)
        _warned_beam = True
    return [result.text for result in load(decoder).batch(crops, batch_size=batch_size)]
