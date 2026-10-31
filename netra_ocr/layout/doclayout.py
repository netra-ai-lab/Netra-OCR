"""DocLayout-YOLO page layout analyzer (https://github.com/opendatalab/DocLayout-YOLO),
on ONNX Runtime."""

from typing import List

import numpy as np

from ..imaging import bgr_to_rgb
from .base import BaseLayoutAnalyzer, LayoutRegion, area, intersection


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
    """``bundle``: a model folder or ``hf://`` source; default: the ``layout``
    model of ``netra_ocr.models`` (DocStructBench, 1024 px)."""

    def __init__(self, bundle: str | None = None, conf: float = 0.2):
        from .. import models
        if bundle is None:
            self._model = models.load("layout")
        else:
            from ..inference import Detector
            self._model = Detector(bundle, device=models.device())
        self._conf = conf

    def _predict(self, img_bgr: np.ndarray):
        """Raw model output: ``[(x1 y1 x2 y2, class name, score)]`` in image pixels."""
        found = self._model(bgr_to_rgb(img_bgr), conf=self._conf)
        return [(d.box, d.label, d.score) for d in found]

    def analyze(self, img_bgr: np.ndarray) -> List[LayoutRegion]:
        h, w = img_bgr.shape[:2]
        regions = []
        for box, label, score in self._predict(img_bgr):
            x1, y1, x2, y2 = (max(0, int(box[0])), max(0, int(box[1])),
                              min(w, int(box[2])), min(h, int(box[3])))
            regions.append(LayoutRegion(label, (x1, y1, x2, y2), score))
        return dedupe_regions(regions)
