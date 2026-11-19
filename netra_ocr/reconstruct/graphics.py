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


def _rules(marks: np.ndarray, min_len: int, axes=(1, 0)) -> np.ndarray:
    """Pixels on long, straight horizontal (axis 1) or vertical (axis 0) ink
    runs: ruled lines, underlines, table borders, page frames. Text has no
    runs that long."""
    out = np.zeros_like(marks)
    for axis in axes:
        m = marks if axis == 1 else marks.T
        edges = np.diff(np.pad(m, ((0, 0), (1, 1))).astype(np.int8), axis=1)
        rows, starts = np.nonzero(edges == 1)
        _, ends = np.nonzero(edges == -1)
        long_runs = (ends - starts) >= min_len
        o = out if axis == 1 else out.T
        for r, a, b in zip(rows[long_runs], starts[long_runs], ends[long_runs]):
            o[r, a:b] = True
    return out


def _longest_runs(m: np.ndarray, bridge: int = 2) -> np.ndarray:
    """Per row of ``m``: the longest run of ink, bridging gaps of up to
    ``bridge`` px (a scanned line breaks up a little)."""
    out = np.zeros(m.shape[0], dtype=np.int32)
    for i, row in enumerate(m):
        on = np.nonzero(row)[0]
        if len(on) == 0:
            continue
        breaks = np.nonzero(np.diff(on) > bridge + 1)[0]
        starts = np.concatenate([[on[0]], on[breaks + 1]])
        ends = np.concatenate([on[breaks], [on[-1]]])
        out[i] = int((ends - starts).max()) + 1
    return out


def _underlines(marks: np.ndarray, text_lines) -> np.ndarray:
    """Underlines of recognized lines: a thin band of rows below the
    letters' core band, each one unbroken run across most of the line, with
    clear rows above and below it. They lie inside the line's box, so the
    text erase would take most of them and leave pieces; marked here, they
    are kept whole. (Subscripts and the next line's tall letters have
    gaps; they never make a run that long.)"""
    h, w = marks.shape
    out = np.zeros_like(marks)
    for box, ink in text_lines:
        if ink is None or ink.x2 - ink.x1 < 8:
            continue
        x1, x2 = max(0, ink.x1), min(w, ink.x2)
        # From the core band's last row: an underline needs a clear row
        # above it, and that row must be looked at, not assumed.
        y1, y2 = max(0, ink.bottom - 1), min(h, int(box[3]) + 2)
        if y2 - y1 < 3:
            continue
        frac = _longest_runs(marks[y1:y2, x1:x2]) / (x2 - x1)
        thick = max(3, int(round(0.25 * ink.core)))
        i = 1
        while i < len(frac):
            if frac[i] < 0.6:
                i += 1
                continue
            j = i
            while j + 1 < len(frac) and frac[j + 1] >= 0.6:
                j += 1
            above = frac[i - 1]
            below = frac[j + 1] if j + 1 < len(frac) else 0.0
            if j - i + 1 <= thick and above < 0.3 and below < 0.3:
                rows = slice(y1 + i, y1 + j + 1)
                out[rows, x1:x2] = marks[rows, x1:x2]
                break
            i = j + 1
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
        reach = int(round(max(1.3 * ink.core, 0.3 * (y2 - y1)))) + 1
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


def _belong_together(a, b) -> bool:
    """Two clusters' boxes overlap enough to be one picture: one mostly
    inside the other (a seal's ring and centre), or overlapping without the
    union being mostly empty (a signature's strokes). A long stroke whose
    box happens to reach over unrelated marks doesn't swallow them."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return False
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    if ix * iy >= 0.5 * min(area_a, area_b):
        return True
    union = (max(a[2], b[2]) - min(a[0], b[0])) * (max(a[3], b[3]) - min(a[1], b[1]))
    return union <= 1.5 * (area_a + area_b)


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
                if _belong_together(b, o):
                    o[0], o[1] = min(o[0], b[0]), min(o[1], b[1])
                    o[2], o[3] = max(o[2], b[2]), max(o[3], b[3])
                    merged = True
                    break
            else:
                out.append(list(b))
        boxes = out
    return [tuple(b) for b in boxes]


def _drop_paper_edges(overlays: List[Overlay], text_lines, w: int, h: int) -> List[Overlay]:
    """Drop the scanned sheet's own edges: the dark line or shadow where the
    paper ends, which isn't part of the document.

    Candidates are long, thin marks lying entirely outside the text (left of,
    right of, above or below every recognized line). A border printed on the
    page is a closed frame: three or four of its sides run (nearly) the full
    height or width of the text. Paper edges show one or two sides, often
    only partly (a sheet lying a little askew). Unless the candidates make
    such a frame, they go, with short slivers of the same shadow beside them.
    """
    boxes = [tuple(int(v) for v in box) for box, _ in text_lines]
    if not boxes or not overlays:
        return overlays
    cx1, cy1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    cx2, cy2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    tol_x, tol_y = 0.01 * w, 0.01 * h
    thin = max(6.0, 0.03 * min(w, h))

    sides = {}
    for o in overlays:
        x1, y1, x2, y2 = o.bbox
        bw, bh = x2 - x1, y2 - y1
        if bw <= thin and bh >= 0.15 * h:
            side = "left" if x2 <= cx1 + tol_x else "right" if x1 >= cx2 - tol_x else None
            complete = bh >= 0.8 * (cy2 - cy1)
        elif bh <= thin and bw >= 0.15 * w:
            side = "top" if y2 <= cy1 + tol_y else "bottom" if y1 >= cy2 - tol_y else None
            complete = bw >= 0.8 * (cx2 - cx1)
        else:
            continue
        if side:
            sides.setdefault(side, []).append((o, complete))
    if not sides or sum(any(c for _, c in v) for v in sides.values()) >= 3:
        return overlays                                  # nothing, or a printed frame

    edges = [o for v in sides.values() for o, _ in v]
    gone = {id(o) for o in edges}

    def sliver(o):
        """A short piece of the same shadow: elongated, beside a dropped edge, outside the text."""
        x1, y1, x2, y2 = o.bbox
        bw, bh = x2 - x1, y2 - y1
        if max(bw, bh) < 2.5 * min(bw, bh):
            return False
        for e in edges:
            ex1, ey1, ex2, ey2 = e.bbox
            if ex2 - ex1 <= thin:                        # a vertical edge
                if abs((x1 + x2) / 2 - (ex1 + ex2) / 2) <= 0.03 * w and (x2 <= cx1 + tol_x or x1 >= cx2 - tol_x):
                    return True
            elif abs((y1 + y2) / 2 - (ey1 + ey2) / 2) <= 0.03 * h and (y2 <= cy1 + tol_y or y1 >= cy2 - tol_y):
                return True
        return False

    return [o for o in overlays if id(o) not in gone and not sliver(o)]


def extract(rgb: np.ndarray, bg: np.ndarray,
            text_lines: Sequence[Tuple[Sequence[int], Optional[Ink]]],
            figures: Sequence[Tuple[int, Sequence[int], bool]],
            tables: Sequence[dict] = ()):
    """Find the page's graphics.

    ``text_lines``: ``(bbox, ink)`` of every recognized line (ink may be
    None). ``figures``: ``(block_index, bbox, is_logo)`` of figure blocks;
    marks inside them belong to them, not to free-standing overlays.

    ``tables``: the ``grid`` (row and column edges) of ruled tables; their
    lines are drawn by the writer as table borders, not as pictures. What
    else is in a table (barcodes, a seal) stays.

    Returns ``(overlays, figure_cutouts)``: free-standing marks, and an
    RGBA cut-out per logo block index (photos and other figures are left
    out: they stay opaque crops).
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
    rules = _rules(marks, min_run) | _underlines(marks, text_lines)
    text &= ~rules
    # Grow the erased text by a pixel: anti-aliased edges are lighter and
    # less saturated than stroke centres and would otherwise leave halos.
    text = _dilate(text, 1) & ~(marks & ~text & (contrast > 2 * MARK_CONTRAST)) & ~rules
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

    # Grid lines of ruled tables: Word draws them as borders.
    for grid in tables:
        for y in grid["rows"]:
            residual[max(0, y - 4):y + 5, max(0, grid["cols"][0] - 4):grid["cols"][-1] + 5] = False
        for x in grid["cols"]:
            residual[max(0, grid["rows"][0] - 4):grid["rows"][-1] + 5, max(0, x - 4):x + 5] = False

    free = residual & ~taken
    r = max(2, round(w / 160))
    min_area = max(25, (w / 110) ** 2)
    overlays: List[Overlay] = []

    line_boxes = [tuple(int(v) for v in box) for box, _ in text_lines]

    def marker(x1, y1, x2, y2) -> bool:
        """A small mark just left of where a text line starts, level with
        it: a list dash or bullet the line detector left out."""
        cy = (y1 + y2) / 2
        for lx1, ly1, lx2, ly2 in line_boxes:
            lh = ly2 - ly1
            if (ly1 <= cy <= ly2 and x2 <= lx1 + 2 and lx1 - x2 <= 2.5 * lh
                    and x2 - x1 <= 0.8 * lh and y2 - y1 <= 0.5 * lh):
                return True
        return False

    def text_residue(x1, y1, x2, y2) -> bool:
        """Leftovers of a recognized line: faint or off-colour edges of its
        letters (tall Muol strokes, JPEG-tinted blue) that the colour-matched
        erase missed. They sit within the line's box; kept, they'd be bits
        of the scan's text floating on the rebuilt, editable text. Not an
        underline below the letters, a seal's red crossing black text, or a
        list dash beside the line."""
        sub = free[y1:y2, x1:x2]
        ys, xs = np.nonzero(sub)
        if len(ys) == 0:
            return False
        tx1, tx2, ty1, ty2 = x1 + xs.min(), x1 + xs.max() + 1, y1 + ys.min(), y1 + ys.max() + 1
        r_, g_, b_ = np.median(rgb[y1:y2, x1:x2][sub].astype(np.int16), axis=0)
        red = r_ - max(g_, b_) > 35
        for (lx1, ly1, lx2, ly2), ink in text_lines:
            bh = ly2 - ly1
            if not (ly1 - 0.25 * bh <= ty1 and ty2 <= ly2 + 0.25 * bh
                    and lx1 - 0.5 * bh <= tx1 and tx2 <= lx2 + 0.5 * bh):
                continue
            core_bottom = ink.bottom if ink is not None else ly2 - 0.3 * bh
            thick = max(3, int(round(0.25 * (ink.core if ink is not None else 0.4 * bh))))
            if (ty1 >= core_bottom and ty2 - ty1 <= thick and tx2 - tx1 >= 3 * (ty2 - ty1)
                    and _longest_runs(sub).max() >= 0.6 * (tx2 - tx1)):
                continue                     # an underline: thin and straight
            if red and not (ink is not None and ink.colour[0] - max(ink.colour[1:]) > 35):
                continue                     # a seal over the text
            return True
        return False

    # Long straight rules (underlines, a page frame, a signature line) are
    # strips of their own: merged with what they pass by, a page frame
    # would make one page-sized picture of everything inside it.
    long_run = max(min_run, int(0.12 * min(w, h)))
    strips = np.zeros_like(free)
    for axis in (1, 0):
        rule = _dilate(_rules(free, long_run, (axis,)), 1) & free
        strips |= rule
        for x, y, cw, ch, area in connected_components(rule):
            x1, y1, x2, y2 = int(x), int(y), int(x + cw), int(y + ch)
            if area < min_area:
                continue
            edge = x1 <= r or y1 <= r or x2 >= w - r or y2 >= h - r
            if edge or cw > 0.97 * w or ch > 0.97 * h:
                continue   # scan border, page shadow, fold
            # A long stroke of the letters themselves -- but no letter makes a
            # straight stroke across much of the page: that's a rule, even when
            # the recognizer read it (and faint text on it) as a line.
            if cw < 0.4 * w and ch < 0.4 * h and text_residue(x1, y1, x2, y2):
                continue
            x1, y1, x2, y2 = max(0, x1 - 1), max(0, y1 - 1), min(w, x2 + 1), min(h, y2 + 1)
            overlays.append(Overlay((x1, y1, x2, y2), _cutout(rgb, bg, contrast, rule, (x1, y1, x2, y2)),
                                    "mark"))
    free &= ~strips
    grown = _dilate(free, r)
    # Drop specks and scan borders first, then merge what's left: a border
    # touching a rule would otherwise swallow it.
    kept = []
    for x, y, cw, ch, _ in connected_components(grown):
        x1, y1, x2, y2 = int(x), int(y), int(x + cw), int(y + ch)
        area = int(free[y1:y2, x1:x2].sum())
        is_marker = area >= 4 and marker(x1, y1, x2, y2)
        if area < min_area and not is_marker:
            continue
        if not is_marker and text_residue(x1, y1, x2, y2):
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
    return _drop_paper_edges(overlays, text_lines, w, h), figure_cutouts
