"""Everything on the page that isn't recognized text, as transparent cut-outs.

Seals, signatures, logos, hand-written ticks and ruled lines are not read
by the models, but a rebuilt letter without its red seal or signature is
not the letter. The approach:

1. Find all marks: pixels that differ from the (locally estimated) paper.
2. Erase the recognized text: inside each line box, only pixels in the
   line's own ink colour. A red seal printed over a black date line keeps
   its red pixels.
3. Group what is left into clusters and drop specks and scan borders.
4. Cut each cluster out with an alpha channel taken from its contrast to
   the paper, so it can sit over the rebuilt text like ink does.

Logos the detector found get the same treatment (so the text under a seal
detected as a "logo" isn't duplicated), unless they're photos.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..imaging import connected_components
from .style import Ink, is_neutral

# A pixel is a mark when its colour is this far (max channel difference)
# from the local paper colour.
MARK_CONTRAST = 48
# A detected figure whose box is mostly marks is a photo, kept opaque.
PHOTO_FILL = 0.45


@dataclass
class Overlay:
    bbox: Tuple[int, int, int, int]
    image: Image.Image   # RGBA
    kind: str            # "stamp" | "signature" | "mark"


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        grown = out.copy()
        grown[1:] |= out[:-1]; grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]; grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _rules(marks: np.ndarray, min_len: int) -> np.ndarray:
    """Pixels on long, straight horizontal or vertical ink runs: ruled lines,
    underlines, table borders. Text has no runs that long."""
    out = np.zeros_like(marks)
    for axis in (1, 0):
        m = marks if axis == 1 else marks.T
        edges = np.diff(np.pad(m, ((0, 0), (1, 1))).astype(np.int8), axis=1)
        rows, starts = np.nonzero(edges == 1)
        _, ends = np.nonzero(edges == -1)
        long_runs = (ends - starts) >= min_len
        o = out if axis == 1 else out.T
        for r, a, b in zip(rows[long_runs], starts[long_runs], ends[long_runs]):
            o[r, a:b] = True
    return out


def _text_pixels(rgb: np.ndarray, marks: np.ndarray, box, colour,
                 ink: Optional[Ink] = None) -> Tuple[slice, slice, np.ndarray]:
    """Mask of the pixels inside ``box`` that belong to text in ``colour``.

    With the line's measured ink, the box is first narrowed to the rows a
    line of that size can reach (stacked vowels above, subscripts below the
    core band): detector boxes are padded and would otherwise swallow a
    rule or a signature stroke passing just above or below the text.
    """
    h, w = marks.shape
    x1, y1, x2, y2 = (int(v) for v in box)
    if ink is not None and ink.core > 0:
        reach = int(round(1.3 * ink.core)) + 1
        y1, y2 = max(y1, ink.top - reach), min(y2, ink.bottom + reach)
        x1, x2 = max(x1, ink.x1 - reach), min(x2, ink.x2 + reach)
    ys, xs = slice(max(0, y1), min(h, y2)), slice(max(0, x1), min(w, x2))
    px = rgb[ys, xs].astype(np.int16)
    chroma = px.max(axis=2) - px.min(axis=2)
    if colour is None or is_neutral(colour):
        same = chroma < 60
    else:
        ink = np.asarray(colour, dtype=np.float32)
        ink_dir = ink - ink.mean()
        ink_dir /= np.linalg.norm(ink_dir) + 1e-6
        d = px - px.mean(axis=2, keepdims=True)
        cos = (d * ink_dir).sum(axis=2) / (np.linalg.norm(d, axis=2) + 1e-6)
        same = (chroma < 45) | (cos > 0.8)
    return ys, xs, marks[ys, xs] & same


def _kind(px: np.ndarray, box) -> str:
    """Guess what a cut-out is from its colour and shape."""
    if len(px) == 0:
        return "mark"
    px = px.astype(np.int16)
    r, g, b = np.median(px, axis=0)
    w, h = box[2] - box[0], box[3] - box[1]
    if h <= max(4, 0.12 * w) or w <= max(4, 0.12 * h):
        return "mark"          # a ruled line or underline
    squareish = 0.55 < w / max(1, h) < 1.8
    if r - max(g, b) > 35 and squareish:
        return "stamp"
    if b - r > 25 or (w > 1.6 * h):
        return "signature"
    return "mark"


def _cutout(rgb: np.ndarray, bg: np.ndarray, contrast: np.ndarray, keep: np.ndarray,
            box) -> Image.Image:
    """RGBA crop: alpha from contrast to the paper, colour un-blended from it."""
    x1, y1, x2, y2 = box
    c = contrast[y1:y2, x1:x2]
    alpha = np.clip((c - MARK_CONTRAST * 0.5) / (MARK_CONTRAST * 1.5), 0, 1)
    alpha *= keep[y1:y2, x1:x2]
    src = rgb[y1:y2, x1:x2].astype(np.float32)
    paper = bg[y1:y2, x1:x2]
    # A half-transparent pixel on the scan is ink blended with paper;
    # recover the ink colour so the cut-out isn't washed out over white.
    colour = paper + (src - paper) / np.maximum(alpha, 0.35)[..., None]
    out = np.dstack([np.clip(colour, 0, 255), alpha * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _merge_boxes(components) -> List[Tuple[int, int, int, int]]:
    """Merge clusters whose boxes overlap: the ring of a seal and its
    centre, a signature's separate strokes. One picture each."""
    boxes = [[int(x), int(y), int(x + w), int(y + h)] for x, y, w, h, _ in components]
    merged = True
    while merged:
        merged = False
        out: List[List[int]] = []
        for b in boxes:
            for o in out:
                if b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3]:
                    o[0], o[1] = min(o[0], b[0]), min(o[1], b[1])
                    o[2], o[3] = max(o[2], b[2]), max(o[3], b[3])
                    merged = True
                    break
            else:
                out.append(list(b))
        boxes = out
    return [tuple(b) for b in boxes]


def extract(rgb: np.ndarray, bg: np.ndarray,
            text_lines: Sequence[Tuple[Sequence[int], Optional[Ink]]],
            figures: Sequence[Tuple[int, Sequence[int], bool]],
            tables: Sequence[Tuple[int, Sequence[int]]] = ()):
    """Find the page's graphics.

    ``text_lines``: ``(bbox, ink)`` of every recognized line (ink may be
    None). ``figures``: ``(block_index, bbox, is_logo)`` of figure blocks;
    marks inside them belong to them, not to free-standing overlays.

    ``tables``: ``(block_index, bbox)`` of table blocks; their grid lines
    are drawn by the writer as table borders, not as pictures.

    Returns ``(overlays, figure_cutouts, ruled)``: free-standing marks, an
    RGBA cut-out per logo block index (photos and other figures are left
    out: they stay opaque crops), and the set of table block indexes whose
    scan shows ruled lines.
    """
    h, w = rgb.shape[:2]
    contrast = np.abs(rgb.astype(np.float32) - bg).max(axis=2)
    marks = contrast > MARK_CONTRAST

    text = np.zeros_like(marks)
    for box, ink in text_lines:
        ys, xs, m = _text_pixels(rgb, marks, box, ink.colour if ink else None, ink)
        text[ys, xs] |= m
    cores = [ink.core for _, ink in text_lines if ink is not None]
    min_run = max(int(0.06 * w), 6 * int(np.median(cores)) if cores else 0, 12)
    rules = _rules(marks, min_run)
    text &= ~rules
    # Grow the erased text by a pixel: anti-aliased edges are lighter and
    # less saturated than stroke centres and would otherwise leave halos.
    text = _dilate(text, 1) & ~(marks & ~text & (contrast > 2 * MARK_CONTRAST))
    residual = marks & ~text

    figure_cutouts: Dict[int, Image.Image] = {}
    taken = np.zeros_like(marks)
    for index, box, is_logo in figures:
        x1, y1, x2, y2 = (max(0, int(box[0])), max(0, int(box[1])),
                          min(w, int(box[2])), min(h, int(box[3])))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        taken[y1:y2, x1:x2] = True
        if not is_logo or marks[y1:y2, x1:x2].mean() >= PHOTO_FILL:
            continue
        figure_cutouts[index] = _cutout(rgb, bg, contrast, residual, (x1, y1, x2, y2))

    ruled = set()
    for index, box in tables:
        x1, y1, x2, y2 = (max(0, int(box[0])), max(0, int(box[1])),
                          min(w, int(box[2])), min(h, int(box[3])))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        taken[y1:y2, x1:x2] = True
        if rules[y1:y2, x1:x2].any(axis=1).sum() >= 2:
            ruled.add(index)

    free = residual & ~taken
    r = max(2, round(w / 160))
    grown = _dilate(free, r)
    min_area = max(25, (w / 110) ** 2)
    overlays: List[Overlay] = []
    # Drop specks and scan borders first, then merge what's left: a border
    # touching a rule would otherwise swallow it.
    kept = []
    for x, y, cw, ch, _ in connected_components(grown):
        x1, y1, x2, y2 = int(x), int(y), int(x + cw), int(y + ch)
        area = int(free[y1:y2, x1:x2].sum())
        if area < min_area:
            continue
        edge = x1 <= r or y1 <= r or x2 >= w - r or y2 >= h - r
        if edge and (cw > 0.5 * w or ch > 0.5 * h or area < 4 * min_area):
            continue   # scan border, page shadow, fold
        if cw > 0.9 * w or ch > 0.9 * h:
            continue
        kept.append((x1, y1, cw, ch, area))
    keep = free & grown
    for x1, y1, x2, y2 in _merge_boxes(kept):
        img = _cutout(rgb, bg, contrast, keep, (x1, y1, x2, y2))
        kind = _kind(rgb[y1:y2, x1:x2][free[y1:y2, x1:x2]], (x1, y1, x2, y2))
        overlays.append(Overlay((x1, y1, x2, y2), img, kind))
    return overlays, figure_cutouts, ruled
