from PIL import Image

from ..imaging import bgr_to_rgb
from .base import BaseTextDetector, DetectedLine, ImageInput, load_bgr


class TesseractDetector(BaseTextDetector):
    def __init__(self):
        try:
            from .tesseract_lines import KhmerLineDetector
        except ImportError as e:
            raise ImportError(
                "The Tesseract detector requires the 'tesseract' extra (and the "
                "system Tesseract binary). Install it with: "
                "pip install \"netra-ocr[tesseract]\""
            ) from e
        print("Initializing KhmerLineDetector (Tesseract)...")
        self._detector = KhmerLineDetector()

    def detect(self, image: ImageInput) -> list:
        img_bgr = load_bgr(image)
        result = self._detector.detect(img_bgr)
        lines = []
        for lb, crop_np in zip(result.lines, result.line_crops):
            pil_crop = Image.fromarray(bgr_to_rgb(crop_np))
            bbox = (lb.x, lb.y, lb.x + lb.w, lb.y + lb.h)
            lines.append(DetectedLine(bbox=bbox, crop=pil_crop))
        return lines
