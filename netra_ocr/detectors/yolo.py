import cv2
import numpy as np
from PIL import Image
from .base import BaseTextDetector, DetectedLine, ImageInput, load_bgr
from .refine import expand_boxes_horizontally


class YoloDetector(BaseTextDetector):
    """Netra OCR's YOLO line detector (class 0: text, class 1: logo), on ONNX Runtime.

    ``bundle``: a model folder or ``hf://`` source; default: the ``lines``
    model of ``netra_ocr.models``. NMS IoU (0.7) is fixed in its settings.
    """

    def __init__(self, bundle: str | None = None, conf: float = 0.25, pad: int = 2,
                 refine: bool = True, gap_ratio: float = 0.8):
        from .. import models
        if bundle is None:
            self._model = models.load("lines")
        else:
            from ..inference import Detector
            self._model = Detector(bundle, device=models.device())
        self._conf = conf
        self._pad  = max(0, int(pad))
        self._refine = refine
        self._gap_ratio = gap_ratio

    def _predict(self, img_bgr):
        """Raw model output: ``(boxes (N, 4) x1 y1 x2 y2, class ids (N,))`` in image pixels."""
        found = self._model(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), conf=self._conf)
        boxes = np.array([d.box for d in found], dtype=np.float32).reshape(-1, 4)
        return boxes, np.array([d.class_id for d in found])

    def detect(self, image: ImageInput) -> list:
        img_bgr = load_bgr(image)
        img_h, img_w = img_bgr.shape[:2]

        # Pass 1: collect raw (box, label), applying the fixed pad + min-size filter.
        text_boxes, logo_boxes = [], []
        boxes, classes = self._predict(img_bgr)
        for box, cls_id in zip(boxes, classes):
            x1, y1, x2, y2 = (
                max(0, int(box[0]) - self._pad), max(0, int(box[1]) - self._pad),
                min(img_w, int(box[2]) + self._pad), min(img_h, int(box[3]) + self._pad),
            )
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            if int(cls_id) == 1:
                logo_boxes.append((x1, y1, x2, y2))
            else:
                text_boxes.append((x1, y1, x2, y2))

        # Pass 2: content-aware horizontal refinement of text boxes only.
        if self._refine and text_boxes:
            text_boxes = expand_boxes_horizontally(
                img_bgr, text_boxes, gap_ratio=self._gap_ratio, pad=self._pad,
            )

        # Pass 3: crop and build DetectedLine results.
        detected = []
        for boxes, label in ((text_boxes, "text"), (logo_boxes, "logo")):
            for (x1, y1, x2, y2) in boxes:
                crop_bgr = img_bgr[y1:y2, x1:x2]
                pil_crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                detected.append(DetectedLine(bbox=(x1, y1, x2, y2), crop=pil_crop, label=label))

        return detected
