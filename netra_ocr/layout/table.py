"""Reconstruct a table's row/column grid from the text lines inside it.

The line detector finds text lines, not cell borders, so the grid is inferred
from geometry alone: lines that overlap vertically form a row, and x-ranges
that line up across rows form a column. It's a heuristic -- merged cells and
cells whose text wraps onto several lines come out as extra rows -- which is
why the result is editable in the UI rather than presented as final.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base import Box

Line = Tuple[Box, str]


def _rows(lines: Sequence[Line], min_v_overlap: float = 0.5) -> List[List[Line]]:
    rows: List[List[Line]] = []
    spans: List[Tuple[int, int]] = []
    for line in sorted(lines, key=lambda l: (l[0][1] + l[0][3]) / 2):
        b = line[0]
        h = max(1, b[3] - b[1])
        if spans:
            top, bot = spans[-1]
            ov = min(bot, b[3]) - max(top, b[1])
            if ov >= min_v_overlap * min(h, max(1, bot - top)):
                rows[-1].append(line)
                spans[-1] = (min(top, b[1]), max(bot, b[3]))
                continue
        rows.append([line])
        spans.append((b[1], b[3]))
    return rows


def _columns(lines: Sequence[Line], gap: int) -> List[Tuple[int, int]]:
    """Union the x-ranges of all lines; each disjoint run is one column."""
    cols: List[List[int]] = []
    for b, _ in sorted(lines, key=lambda l: l[0][0]):
        if cols and b[0] <= cols[-1][1] + gap:
            cols[-1][1] = max(cols[-1][1], b[2])
        else:
            cols.append([b[0], b[2]])
    return [(c[0], c[1]) for c in cols]


def build_grid(lines: Sequence[Line]) -> Optional[List[List[str]]]:
    """Return ``rows x cols`` cell texts, or ``None`` if no real grid is found.

    A grid needs at least 2 rows and 2 columns; anything less (for example the
    detector returned whole table rows as single lines) is reported as
    ``None`` so the caller can fall back to a single-cell table plus the image.
    """
    if len(lines) < 4:
        return None
    heights = sorted(b[3] - b[1] for b, _ in lines)
    gap = max(2, heights[len(heights) // 2] // 3)

    rows = _rows(lines)
    cols = _columns(lines, gap)
    if len(rows) < 2 or len(cols) < 2:
        return None

    grid = [["" for _ in cols] for _ in rows]
    for ri, row in enumerate(rows):
        for b, text in sorted(row, key=lambda l: l[0][0]):
            cx = (b[0] + b[2]) / 2
            ci = min(range(len(cols)),
                     key=lambda c: 0 if cols[c][0] <= cx <= cols[c][1]
                     else min(abs(cx - cols[c][0]), abs(cx - cols[c][1])))
            grid[ri][ci] = f"{grid[ri][ci]} {text}".strip() if grid[ri][ci] else text
    return grid


# ── ruled tables: read the drawn grid ────────────────────────────────────

def _ridges(gray: np.ndarray, axis: int, reach: int = 3, step: float = 3.0, smooth: int = 9) -> np.ndarray:
    """Where ``gray`` has a thin dark line along ``axis`` (0: horizontal,
    1: vertical): pixels a little darker than the pixels ``reach`` away on
    both sides, after smoothing along the line. Scanned grid lines are
    often only a few grey levels darker than the paper, far too faint for
    an ink threshold, but they are straight and long."""
    g = gray.astype(np.float32)
    k = np.ones(smooth, dtype=np.float32) / smooth
    if axis == 0:
        s = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, g)
        a, b = np.roll(s, reach, 0), np.roll(s, -reach, 0)
    else:
        s = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, g)
        a, b = np.roll(s, reach, 1), np.roll(s, -reach, 1)
    r = ((a - s) >= step) & ((b - s) >= step)
    if axis == 0:
        r[:reach] = r[-reach:] = False
        r[:, :smooth] = r[:, -smooth:] = False
    else:
        r[:, :reach] = r[:, -reach:] = False
        r[:smooth] = r[-smooth:] = False
    return r


def _peaks(cover: np.ndarray, threshold: float) -> List[int]:
    """Centres of runs of ``cover >= threshold`` (a line is 1-4 px thick)."""
    out: List[List[int]] = []
    for i in np.nonzero(cover >= threshold)[0]:
        if out and i - out[-1][-1] <= 2:
            out[-1].append(int(i))
        else:
            out.append([int(i)])
    return [int(round(sum(g) / len(g))) for g in out]


def ruled_grid(gray: np.ndarray, box: Box, lines: Sequence[Line]
               ) -> Optional[Tuple[List[int], List[int]]]:
    """Row and column edges of a table drawn with grid lines, in page pixels.

    Horizontal rules must cross most of the table. A vertical rule must run
    through nearly every row, or through many rows standing alone (barcode
    bars and text stems are tall and dark too, but come in clusters). Columns whose rule is too faint to see
    are taken from the gaps between the text columns. Returns ``None`` when
    the table isn't ruled (fewer than two horizontal rules).
    """
    h, w = gray.shape
    pad = 12
    x1, y1 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
    x2, y2 = min(w, int(box[2]) + pad), min(h, int(box[3]) + pad)
    crop = gray[y1:y2, x1:x2]
    if crop.shape[0] < 20 or crop.shape[1] < 40:
        return None
    hr = _ridges(crop, 0)
    ys = _peaks(hr.mean(axis=1), 0.45)
    if len(ys) < 2:
        return None

    vr = _ridges(crop, 1)
    xs = []
    bands = list(zip(ys, ys[1:]))
    candidates = _peaks(vr.mean(axis=0), 0.2)
    for k, x in enumerate(candidates):
        lo, hi = max(0, x - 2), x + 3
        present = [vr[a + 3:b - 2, lo:hi].any(axis=1).mean() >= 0.9 for a, b in bands if b - a > 6]
        if not present:
            continue
        share = sum(present) / len(present)
        # A rule runs through (nearly) every row; a faint one may fade in
        # places, but it stands alone. Barcode bars line up from row to row
        # when codes share a prefix, but they come in dense clusters.
        alone = all(abs(x - o) > 12 for j, o in enumerate(candidates) if j != k)
        if share >= 0.85 or (alone and share >= 0.4):
            xs.append(x)

    ys = [y + y1 for y in ys]
    xs = [x + x1 for x in xs]
    text = [(b, t) for b, t in lines if t.strip()]
    text_left = min((b[0] for b, _ in text), default=int(box[0]))
    text_right = max((b[2] for b, _ in text), default=int(box[2]))
    # No rule left of the text (a table without outer borders): the
    # table's own edge is the first column edge; likewise on the right.
    if not xs or xs[0] > text_left:
        xs.insert(0, min(text_left, int(box[0])))
    if xs[-1] < text_right:
        xs.append(max(text_right, int(box[2])))

    # Text columns between rules too faint to find: split at the gaps.
    if text:
        heights = sorted(b[3] - b[1] for b, _ in text)
        gap = max(2, heights[len(heights) // 2] // 3)
        cols = _columns(text, gap)
        for a, b in zip(cols, cols[1:]):
            mid = (a[1] + b[0]) // 2
            if all(abs(mid - x) > max(8, (b[0] - a[1]) // 2) for x in xs):
                xs.append(mid)
        xs.sort()
        # An edge through the middle of a text line isn't a column edge.
        xs = [x for i, x in enumerate(xs)
              if i in (0, len(xs) - 1) or not any(b[0] + 6 < x < b[2] - 6 for b, _ in text)]
    xs = _merge_edges(xs, 10)
    ys = _drop_slivers(_merge_edges(ys, 6))
    if len(xs) < 2 or len(ys) < 2:
        return None
    return ys, xs


def _merge_edges(edges: Sequence[int], within: int) -> List[int]:
    out: List[int] = []
    for e in sorted(edges):
        if out and e - out[-1] <= within:
            out[-1] = (out[-1] + e) // 2
        else:
            out.append(e)
    return out


def _drop_slivers(ys: List[int]) -> List[int]:
    """Double rules (a line under the header drawn twice) make rows far
    thinner than any row of text; fold them into the row above."""
    if len(ys) < 4:
        return ys
    heights = sorted(b - a for a, b in zip(ys, ys[1:]))
    typical = heights[len(heights) // 2]
    out = [ys[0]]
    for y in ys[1:]:
        if y - out[-1] < 0.4 * typical and len(out) > 1:
            out[-1] = y          # the sliver joins the row above
        else:
            out.append(y)
    return out


def cells_from_grid(lines: Sequence[Line], ys: Sequence[int], xs: Sequence[int]) -> List[List[str]]:
    """Cell texts of a ruled grid: each line goes to the cell holding its centre."""
    rows = len(ys) - 1
    cols = len(xs) - 1
    found: List[List[List[Line]]] = [[[] for _ in range(cols)] for _ in range(rows)]

    def index(edges, v):
        for i in range(len(edges) - 1):
            if v < edges[i + 1]:
                return max(0, i)
        return len(edges) - 2

    for b, t in lines:
        if not t.strip():
            continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if cy < ys[0] - 4 or cy > ys[-1] + 4:
            continue
        found[index(ys, cy)][index(xs, cx)].append((b, t))
    out = []
    for r in found:
        row = []
        for cell in r:
            cell = sorted(cell, key=lambda l: ((l[0][1] + l[0][3]) / 2, l[0][0]))
            row.append("\n".join(t for _, t in cell))
        out.append(row)
    return out
