"""Post-detection box refinement utilities.

These helpers run *after* a detector (e.g. YOLO) has produced bounding boxes.
They look at the actual ink in the image to fix boxes that under-shoot the
text line, without changing the detector's interface.

Only ``numpy`` is used (through ``netra_ocr.imaging``), so this stays cheap and dependency-light.
"""

from typing import List, Tuple

import numpy as np

from ..imaging import connected_components, otsu_threshold, to_gray

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def _binarize_ink(img_bgr: np.ndarray) -> np.ndarray:
    """Return a binary image where ink (text) is 255 and background is 0.

    Uses Otsu thresholding and auto-flips for inverted scans (light text on a
    dark page) by checking the foreground ratio.
    """
    gray = to_gray(img_bgr)
    binary = np.where(gray > otsu_threshold(gray), 0, 255).astype(np.uint8)
    # If more than half the image is "ink", Otsu picked the wrong polarity.
    if np.count_nonzero(binary) > 0.5 * binary.size:
        binary = 255 - binary
    return binary


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dedupe(boxes: List[Box], iou_threshold: float = 0.6) -> List[Box]:
    """Drop near-duplicate boxes, keeping the earlier (first-seen) one."""
    kept: List[Box] = []
    for box in boxes:
        if any(_iou(box, k) > iou_threshold for k in kept):
            continue
        kept.append(box)
    return kept


def expand_boxes_horizontally(
    img_bgr: np.ndarray,
    boxes: List[Box],
    gap_ratio: float = 0.8,
    min_area_ratio: float = 0.02,
    pad: int = 2,
    contain_ratio: float = 0.6,
    max_height_ratio: float = 1.6,
) -> List[Box]:
    """Grow each box left/right to cover the full text line using ink components.

    The vertical bounds of every box are preserved (horizontal-only refinement).

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR image the boxes were detected on.
    boxes : list of (x1, y1, x2, y2)
        Text boxes to refine (caller should exclude non-text such as logos).
    gap_ratio : float
        Maximum horizontal gap to bridge, as a fraction of the box height.
        Larger values cover wider word spacing; too large merges columns.
    min_area_ratio : float
        Ink components smaller than ``min_area_ratio * box_height**2`` are
        ignored as noise.
    pad : int
        Horizontal padding applied after expansion.
    contain_ratio : float
        Minimum fraction of a component's height that must fall inside the line
        band for it to count as line text. Rejects photos/graphics that extend
        well above/below the band (their contained fraction is low).
    max_height_ratio : float
        Components taller than ``max_height_ratio * line_height`` are skipped as
        non-text (e.g. an ID-card photo is taller than a text line).

    Returns
    -------
    list of (x1, y1, x2, y2)
        Refined boxes, de-duplicated, clamped to image bounds.
    """
    if not boxes:
        return []

    img_h, img_w = img_bgr.shape[:2]
    binary = _binarize_ink(img_bgr)

    # One connected-component pass for the whole image, reused for every box.
    # Their order doesn't matter: each box grows until no more components fit.
    comps = []  # (cx1, cy1, cx2, cy2, area)
    for cx, cy, cw, ch, area in connected_components(binary).tolist():
        comps.append((cx, cy, cx + cw, cy + ch, area))

    refined: List[Box] = []
    for (x1, y1, x2, y2) in boxes:
        line_h = max(1, y2 - y1)
        gap = max(4, int(gap_ratio * line_h))
        min_area = min_area_ratio * (line_h ** 2)

        # Candidate ink components that belong to this line.
        candidates = []
        for (cx1, cy1, cx2, cy2, area) in comps:
            if area < min_area:
                continue
            if (cx2 - cx1) > 0.9 * img_w:  # full-width rule / separator
                continue
            comp_h = cy2 - cy1
            if comp_h > max_height_ratio * line_h:  # too tall to be line text (photo/graphic)
                continue
            # Require the component to be *contained* in the line band: most of its
            # height must fall inside [y1, y2]. A photo extends far above/below the
            # band, so its contained fraction is low and it is rejected.
            overlap = min(cy2, y2) - max(cy1, y1)
            if comp_h > 0 and overlap / comp_h >= contain_ratio:
                candidates.append((cx1, cx2))

        # Iteratively union candidates within `gap` of the current x-extent.
        ext_x1, ext_x2 = x1, x2
        remaining = list(candidates)
        changed = True
        while changed:
            changed = False
            still: List[Tuple[int, int]] = []
            for (cx1, cx2) in remaining:
                # Distance from component to current extent (0 if overlapping).
                if cx1 > ext_x2:
                    dist = cx1 - ext_x2
                elif cx2 < ext_x1:
                    dist = ext_x1 - cx2
                else:
                    dist = 0
                if dist <= gap:
                    ext_x1 = min(ext_x1, cx1)
                    ext_x2 = max(ext_x2, cx2)
                    changed = True
                else:
                    still.append((cx1, cx2))
            remaining = still

        nx1 = max(0, ext_x1 - pad)
        nx2 = min(img_w, ext_x2 + pad)
        refined.append((nx1, y1, nx2, y2))

    return _dedupe(refined)
