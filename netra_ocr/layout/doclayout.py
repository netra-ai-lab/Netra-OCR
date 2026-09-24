"""DocLayout-YOLO page layout analyzer (https://github.com/opendatalab/DocLayout-YOLO)."""

import logging
import os
from typing import List

import numpy as np

from .base import BaseLayoutAnalyzer, LayoutRegion, area, intersection

logger = logging.getLogger(__name__)

DEFAULT_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
DEFAULT_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"


def default_weights_path() -> str:
    """Resolve the DocLayout-YOLO weights (NETRA_LAYOUT_WEIGHTS, else the HF Hub cache)."""
    override = os.environ.get("NETRA_LAYOUT_WEIGHTS")
    if override:
        return override
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=DEFAULT_REPO, filename=DEFAULT_FILENAME)


def dedupe_regions(regions: List[LayoutRegion], iou_threshold: float = 0.6,
                   contain_threshold: float = 0.85) -> List[LayoutRegion]:
    """Class-agnostic suppression of duplicate layout boxes.

    DocLayout-YOLO often emits the same area twice under different classes
    (e.g. "title" 0.47 and "abandon" 0.20), or a low-confidence box that
    swallows two confident ones. Keep regions in descending score order and
    drop any later region that overlaps a kept one too much or where one of
    the two is almost entirely contained in the other.
    """
    kept: List[LayoutRegion] = []
    for r in sorted(regions, key=lambda r: r.score, reverse=True):
        ra = area(r.bbox)
        if ra == 0:
            continue
        duplicate = False
        for k in kept:
            inter = intersection(r.bbox, k.bbox)
            if inter == 0:
                continue
            ka = area(k.bbox)
            if (inter / (ra + ka - inter) > iou_threshold
                    or inter / ra > contain_threshold
                    or inter / ka > contain_threshold):
                duplicate = True
                break
        if not duplicate:
            kept.append(r)
    return kept


class DocLayoutYoloAnalyzer(BaseLayoutAnalyzer):
    def __init__(self, weights: str | None = None, conf: float = 0.2,
                 imgsz: int = 1024, device: str | None = None):
        try:
            from doclayout_yolo import YOLOv10
        except ImportError as e:
            raise ImportError(
                "Layout analysis requires the 'layout' extra. Install it with: "
                "pip install \"netra-ocr[layout]\""
            ) from e
        weights = weights or default_weights_path()
        logger.info("Loading DocLayout-YOLO weights: %s", weights)
        self._model = YOLOv10(weights)
        self._names = self._model.names
        self._conf = conf
        self._imgsz = imgsz
        self._device = device or os.environ.get("NETRA_DEVICE") or None

    def analyze(self, img_bgr: np.ndarray) -> List[LayoutRegion]:
        h, w = img_bgr.shape[:2]
        kwargs = {"imgsz": self._imgsz, "conf": self._conf, "verbose": False}
        if self._device:
            kwargs["device"] = self._device
        result = self._model.predict(img_bgr, **kwargs)[0]
        regions = []
        for box, cls_id, score in zip(result.boxes.xyxy.cpu().numpy(),
                                      result.boxes.cls.cpu().numpy(),
                                      result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = (max(0, int(box[0])), max(0, int(box[1])),
                              min(w, int(box[2])), min(h, int(box[3])))
            regions.append(LayoutRegion(self._names[int(cls_id)], (x1, y1, x2, y2), float(score)))
        return dedupe_regions(regions)
