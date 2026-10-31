"""The few image operations the pipeline needs, in numpy and Pillow.

Each gives the same result as the OpenCV call named in its docstring, so the
default install doesn't need OpenCV (only the optional legacy and Tesseract
detectors do). Images are uint8 arrays; colour arrays are BGR, as in OpenCV.
"""

import numpy as np
from PIL import Image, ImageOps


def read_bgr(path) -> np.ndarray:
    """``cv2.imread(path)``: a BGR image, turned upright by its EXIF orientation."""
    try:
        with Image.open(path) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except OSError as e:
        raise FileNotFoundError(f"Could not read image: {path}") from e
    return rgb_to_bgr(rgb)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """``cv2.cvtColor(image, cv2.COLOR_BGR2RGB)``."""
    return np.ascontiguousarray(image[..., ::-1])


rgb_to_bgr = bgr_to_rgb


def to_gray(bgr: np.ndarray) -> np.ndarray:
    """``cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)``: OpenCV 5's 15-bit fixed-point
    BT.601 weights (identical to it on all 2**24 colours)."""
    b, g, r = (bgr[..., i].astype(np.uint32) for i in range(3))
    return ((b * 3735 + g * 19235 + r * 9798 + (1 << 14)) >> 15).astype(np.uint8)


def otsu_threshold(gray: np.ndarray) -> int:
    """The threshold ``cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)`` picks.

    Same arithmetic as OpenCV's ``getThreshVal_Otsu_8u``, in float64, so ties
    resolve the same way. Pixels above it are foreground.
    """
    hist = np.bincount(gray.ravel(), minlength=256).tolist()
    scale = 1.0 / gray.size
    mu = sum(i * float(h) for i, h in enumerate(hist)) * scale
    eps = float(np.finfo(np.float32).eps)
    mu1 = q1 = 0.0
    best_sigma, best = 0.0, 0
    for i, h in enumerate(hist):
        p = h * scale
        mu1 *= q1
        q1 += p
        q2 = 1.0 - q1
        if min(q1, q2) < eps or max(q1, q2) > 1.0 - eps:
            continue
        mu1 = (mu1 + i * p) / q1
        mu2 = (mu - q1 * mu1) / q2
        sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2)
        if sigma > best_sigma:
            best_sigma, best = sigma, i
    return best


def connected_components(mask: np.ndarray) -> np.ndarray:
    """Bounding boxes and areas of the 8-connected components of ``mask``.

    Returns an ``(N, 5)`` int array of ``x, y, w, h, area`` rows, like the
    foreground rows of ``cv2.connectedComponentsWithStats(mask, connectivity=8)``
    (as a set; the row order differs).
    """
    mask = np.asarray(mask) != 0
    height, width = mask.shape
    # Horizontal runs of foreground pixels, in row-major order: [start, end).
    edges = np.diff(np.pad(mask, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    rows, starts = np.nonzero(edges == 1)
    _, ends = np.nonzero(edges == -1)
    n = len(rows)
    if n == 0:
        return np.zeros((0, 5), dtype=np.int64)

    # Runs on adjacent rows touch (8-connectivity) when their column ranges
    # overlap or meet diagonally. Encoding (row, column) as one sortable key,
    # the runs on the next row that touch run a form one contiguous range.
    stride = width + 2
    key_start = rows * stride + starts
    key_end = rows * stride + ends
    below = (rows + 1) * stride
    lo = np.searchsorted(key_end, below + starts, side="left")
    hi = np.searchsorted(key_start, below + ends, side="right")
    count = np.maximum(hi - lo, 0)
    a = np.repeat(np.arange(n), count)
    b = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count) + np.repeat(lo, count)

    # Every run takes the smallest label it is connected to.
    labels = np.arange(n)
    while len(a):
        low = np.minimum(labels[a], labels[b])
        before = labels.copy()
        np.minimum.at(labels, a, low)
        np.minimum.at(labels, b, low)
        while True:
            jumped = labels[labels]
            if np.array_equal(jumped, labels):
                break
            labels = jumped
        if np.array_equal(labels, before):
            break

    _, comp = np.unique(labels, return_inverse=True)
    m = comp.max() + 1
    x1 = np.full(m, width); np.minimum.at(x1, comp, starts)
    x2 = np.zeros(m, dtype=np.int64); np.maximum.at(x2, comp, ends)
    y1 = np.full(m, height); np.minimum.at(y1, comp, rows)
    y2 = np.zeros(m, dtype=np.int64); np.maximum.at(y2, comp, rows + 1)
    area = np.bincount(comp, weights=ends - starts, minlength=m).astype(np.int64)
    return np.stack([x1, y1, x2 - x1, y2 - y1, area], axis=1)
