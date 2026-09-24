import cv2
from PIL import Image
from .base import BaseTextDetector, DetectedLine, ImageInput, load_bgr


class TesseractDetector(BaseTextDetector):
    def __init__(self):
        try:
            from netra_ocr.engine import KhmerLineDetector
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
            pil_crop = Image.fromarray(cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB))
            bbox = (lb.x, lb.y, lb.x + lb.w, lb.y + lb.h)
            lines.append(DetectedLine(bbox=bbox, crop=pil_crop))
        return lines
