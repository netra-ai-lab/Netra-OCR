"""YOLO-family detectors: the text-line/logo detector (``lines``) and
DocLayout-YOLO (``layout``). One ONNX graph, ``images`` (1, 3, H, W) RGB in
[0, 1] -> ``output0``, either raw (``yolo-nms``: (1, 4 + classes, anchors), NMS
done here) or NMS-free (``yolo-end2end``: (1, max_det, 6) x1 y1 x2 y2 score
class). Settings come from the model folder's ``netra-model.json``.
"""

from dataclasses import dataclass

import numpy as np

from . import hub
from .recognizer import to_pil
from .session import device_kind


@dataclass
class Detection:
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in original image pixels
    score: float
    label: str
    class_id: int


def to_rgb_array(image) -> np.ndarray:
    """Path / PIL image / uint8 array (H x W gray or H x W x 3 RGB) -> H x W x 3 uint8 RGB."""
    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
        return np.ascontiguousarray(image, dtype=np.uint8)
    return np.asarray(to_pil(image).convert("RGB"))


def resize_bilinear(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resize with half-pixel centers, no antialiasing, float64 math, round half up.

    Written out rather than delegated to OpenCV: cv2's fixed-point
    INTER_LINEAR differs between its SIMD and scalar paths and across
    versions, while this exact arithmetic is reproduced by the browser version
    (web/), so Python and JavaScript see identical pixels.
    """
    src_h, src_w = image.shape[:2]

    def axis(dst, src):
        pos = np.clip((np.arange(dst) + 0.5) * (src / dst) - 0.5, 0, src - 1)
        lo = np.floor(pos).astype(np.int64)
        return lo, np.minimum(lo + 1, src - 1), pos - lo

    y0, y1, wy = axis(height, src_h)
    x0, x1, wx = axis(width, src_w)
    # Gather the needed rows/columns as uint8 first; only output-sized arrays become float64.
    rows0, rows1 = image[y0], image[y1]
    wx, wy = wx[None, :, None], wy[:, None, None]
    top = rows0[:, x0].astype(np.float64) * (1 - wx) + rows0[:, x1].astype(np.float64) * wx
    bottom = rows1[:, x0].astype(np.float64) * (1 - wx) + rows1[:, x1].astype(np.float64) * wx
    return np.floor(top * (1 - wy) + bottom * wy + 0.5).astype(np.uint8)


@dataclass
class LetterboxInfo:
    orig_h: int
    orig_w: int
    gain_x: float
    gain_y: float
    pad_x: int
    pad_y: int


def letterbox(image, imgsz: int, stride: int = 32, auto: bool = True, pad_value: int = 114,
              scaleup: bool = True, **_) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize keeping the aspect ratio so the long side is ``imgsz``, then pad
    with ``pad_value`` -- to a multiple of ``stride`` (``auto``, the minimum
    rectangle) or to ``imgsz`` x ``imgsz``. Returns ``(1, 3, H, W)`` float32 RGB in [0, 1]."""
    img = to_rgb_array(image)
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    if not scaleup:
        r = min(r, 1.0)
    new_w, new_h = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = imgsz - new_w, imgsz - new_h
    if auto:
        dw, dh = dw % stride, dh % stride
    dw, dh = dw / 2, dh / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    if (new_w, new_h) != (w, h):
        img = resize_bilinear(img, new_w, new_h)
    padded = np.full((new_h + top + bottom, new_w + left + right, 3), pad_value, np.uint8)
    padded[top:top + new_h, left:left + new_w] = img

    tensor = (padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    return tensor, LetterboxInfo(h, w, new_w / w, new_h / h, left, top)


def restore_boxes(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Map xyxy boxes from the letterboxed input back to original pixels (clipped)."""
    boxes = boxes.astype(np.float32).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - info.pad_x) / info.gain_x
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - info.pad_y) / info.gain_y
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, info.orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, info.orig_h)
    return boxes


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices in descending score order."""
    order = np.argsort(-scores, kind="stable")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        overlap = inter / (areas[i] + areas[rest] - inter)
        order = rest[overlap <= iou]
    return np.array(keep, dtype=np.int64)


def decode_yolo_raw(output: np.ndarray, conf: float, iou: float = 0.7, max_det: int = 300,
                    max_nms: int = 30000, agnostic: bool = False, **_):
    """Raw YOLO output ``(1, 4 + classes, anchors)`` (cx, cy, w, h, class scores)
    -> boxes, scores, class ids after best-class selection and class-aware NMS."""
    pred = output[0].T                                   # (anchors, 4 + classes)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(1)
    scores = scores_all[np.arange(len(pred)), cls]
    keep = scores > conf
    pred, cls, scores = pred[keep], cls[keep], scores[keep]
    if len(pred) > max_nms:
        top = np.argsort(-scores, kind="stable")[:max_nms]
        pred, cls, scores = pred[top], cls[top], scores[top]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    # Class-aware NMS by offsetting each class into its own coordinate range.
    offset = 0 if agnostic else cls[:, None].astype(np.float32) * 7680
    kept = nms(boxes + offset, scores, iou)[:max_det]
    return boxes[kept], scores[kept], cls[kept]


def decode_yolo_end2end(output: np.ndarray, conf: float, **_):
    """NMS-free YOLO output ``(1, max_det, 6)`` -> boxes, scores, class ids above ``conf``."""
    pred = output[0]
    pred = pred[pred[:, 4] > conf]
    return pred[:, :4], pred[:, 4], pred[:, 5].astype(np.int64)


_HEADS = {"yolo-nms": decode_yolo_raw, "yolo-end2end": decode_yolo_end2end}


class Detector:
    """Image -> list of ``Detection``. ``source``: a model folder or ``hf://`` path."""

    def __init__(self, source, device: str | None = None):
        _, config, self.variant, sessions, self.device = hub.load_sessions(source, device)
        self.name = f"{config['name']} {config['version']}"
        settings = config["components"]
        self.resize = dict(settings["preprocess"])
        # On GPUs always pad to the full square: ONNX Runtime's CUDA convolutions
        # redo their setup whenever the input shape differs from the previous call
        # (~60 ms vs ~6 ms per page for the line detector), so one fixed shape wins.
        if device_kind(self.device) in ("cuda", "tensorrt"):
            self.resize["auto"] = False
        self.head = dict(settings["head"])
        self._decode = _HEADS[self.head.pop("type")]
        self.labels = self.head.pop("labels")
        self.conf = self.head.pop("conf", 0.25)
        self.model = sessions["model"]

    def __repr__(self):
        return f"Detector({self.name}, device={self.device}, variant={self.variant or 'fp32'})"

    def __call__(self, image, conf: float | None = None) -> list[Detection]:
        tensor, info = letterbox(image, **self.resize)
        output = self.model.run({"images": tensor})["output0"]
        boxes, scores, classes = self._decode(output, conf=self.conf if conf is None else conf, **self.head)
        boxes = restore_boxes(boxes, info)
        return [Detection(tuple(float(v) for v in box), float(score), self.labels[int(c)], int(c))
                for box, score, c in zip(boxes, scores, classes)]
