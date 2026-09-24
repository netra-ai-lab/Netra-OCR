"""Reconstruct a table's row/column grid from the text lines inside it.

The line detector finds text lines, not cell borders, so the grid is inferred
from geometry alone: lines that overlap vertically form a row, and x-ranges
that line up across rows form a column. It's a heuristic -- merged cells and
cells whose text wraps onto several lines come out as extra rows -- which is
why the result is editable in the UI rather than presented as final.
"""

from typing import List, Optional, Sequence, Tuple

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
