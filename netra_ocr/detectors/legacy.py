import cv2
from PIL import Image
from .base import BaseTextDetector, DetectedLine, ImageInput, load_bgr


class LegacyDetector(BaseTextDetector):
    def __init__(self, pad: int | None = None, **kwargs):
        from netra_ocr.legacy_detector import ImageProcessingTextDetector
        # `pad` is applied per-box AFTER detection (see detect), not passed into
        # the detector's internal `padding`. The internal padding influences box
        # merging, so feeding `pad` there would change the number of boxes as the
        # value changes. Keeping detection on its stable auto-padding fixes that.
        self._pad = max(0, pad) if pad is not None else 0
        self._detector = ImageProcessingTextDetector(**kwargs)

    def detect(self, image: ImageInput) -> list:
        img_bgr = load_bgr(image)
        line_boxes = self._detector.detect_lines(img_bgr)
        h_img, w_img = img_bgr.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        lines = []
        for (x, y, w, h) in line_boxes:
            # Apply user padding per-box, clamped to image bounds. Box count is
            # already final here, so changing `pad` only resizes boxes.
            x1 = max(0, x - self._pad)
            y1 = max(0, y - self._pad)
            x2 = min(w_img, x + w + self._pad)
            y2 = min(h_img, y + h + self._pad)
            crop = pil_img.crop((x1, y1, x2, y2))
            lines.append(DetectedLine(bbox=(x1, y1, x2, y2), crop=crop, label="text"))
        return lines
