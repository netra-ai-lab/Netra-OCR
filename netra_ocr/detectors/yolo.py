import os
import cv2
from PIL import Image
from .base import BaseTextDetector, DetectedLine, ImageInput, load_bgr
from .refine import expand_boxes_horizontally

_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "yolo26s_best", "best.pt"
)


class YoloDetector(BaseTextDetector):
    def __init__(self, weights: str = _WEIGHTS_PATH, conf: float = 0.25, iou: float = 0.7,
                 pad: int = 2, refine: bool = True, gap_ratio: float = 0.8):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "The YOLO detector requires the 'yolo' extra. Install it with: "
                "pip install \"netra-ocr[yolo]\""
            ) from e
        print(f"Loading YOLO weights: {weights}")
        self._model = YOLO(weights)
        self._conf = conf
        self._iou  = iou
        self._pad  = max(0, int(pad))
        self._refine = refine
        self._gap_ratio = gap_ratio

    def detect(self, image: ImageInput) -> list:
        img_bgr = load_bgr(image)
        # NETRA_DEVICE pins every model to one device (e.g. "cpu" on a GPU
        # host); unset, ultralytics picks the GPU when there is one.
        device = os.environ.get("NETRA_DEVICE")
        results = self._model.predict(
            source=img_bgr,
            conf=self._conf,
            iou=self._iou,
            imgsz=640,
            verbose=False,
            **({"device": device} if device else {}),
        )
        img_h, img_w = img_bgr.shape[:2]

        # Pass 1: collect raw (box, label), applying the fixed pad + min-size filter.
        text_boxes, logo_boxes = [], []
        for result in results:
            boxes   = result.boxes.xyxy.cpu().numpy()   # (N, 4) — x1 y1 x2 y2
            classes = result.boxes.cls.cpu().numpy()    # (N,)  integer class ids
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
