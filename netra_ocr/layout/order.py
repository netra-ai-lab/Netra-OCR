"""Reading order and line-to-region assignment for layout analysis."""

from typing import Callable, Dict, List, Sequence, TypeVar

from .base import Box, LayoutRegion, area, intersection

T = TypeVar("T")


def _cut_groups(items: List[T], key: Callable[[T], Box], axis: int, tolerance: float) -> List[List[T]]:
    """Split ``items`` along ``axis`` (0 = x, 1 = y) wherever there's an empty gap.

    Each box is shrunk by ``tolerance`` of its own extent on that axis first, so
    boxes that merely touch or overlap by a few pixels (common with detector
    padding) don't block a cut.
    """
    spans = []
    for it in items:
        b = key(it)
        lo, hi = b[axis], b[axis + 2]
        shrink = (hi - lo) * tolerance
        spans.append((lo + shrink, hi - shrink, it))
    spans.sort(key=lambda s: s[0])

    groups: List[List[T]] = []
    current: List[T] = []
    reach = None
    for lo, hi, it in spans:
        if current and lo > reach:
            groups.append(current)
            current = []
            reach = None
        current.append(it)
        reach = hi if reach is None else max(reach, hi)
    if current:
        groups.append(current)
    return groups


def xy_cut_order(items: Sequence[T], key: Callable[[T], Box], tolerance: float = 0.1) -> List[T]:
    """Order boxes in reading order with a recursive XY-cut.

    Cut the page into horizontal bands wherever there's a full-width vertical
    gap (top to bottom), then cut each band into columns (left to right), and
    recurse. A full-width title therefore separates the column blocks above and
    below it, and two-column text is read column by column. When no cut is
    possible, fall back to top-to-bottom, then left-to-right.
    """
    items = list(items)
    if len(items) <= 1:
        return items

    bands = _cut_groups(items, key, axis=1, tolerance=tolerance)
    if len(bands) > 1:
        return [it for band in bands for it in xy_cut_order(band, key, tolerance)]

    columns = _cut_groups(items, key, axis=0, tolerance=tolerance)
    if len(columns) > 1:
        return [it for col in columns for it in xy_cut_order(col, key, tolerance)]

    return sorted(items, key=lambda it: (key(it)[1], key(it)[0]))


def assign_lines(line_boxes: Sequence[Box], regions: Sequence[LayoutRegion],
                 min_overlap: float = 0.5) -> Dict[int, List[int]]:
    """Map region index -> indices of the lines that belong to it.

    A line belongs to the region that covers the largest fraction of the line
    (at least ``min_overlap``); ties go to the smaller region, so a caption
    inside a figure's box is claimed by the caption. Unassigned lines are
    returned under key ``-1``.
    """
    out: Dict[int, List[int]] = {i: [] for i in range(len(regions))}
    out[-1] = []
    for li, lb in enumerate(line_boxes):
        la = area(lb)
        best, best_key = -1, None
        for ri, r in enumerate(regions):
            if la == 0:
                break
            frac = intersection(lb, r.bbox) / la
            if frac < min_overlap:
                continue
            k = (round(frac, 2), -area(r.bbox))
            if best_key is None or k > best_key:
                best, best_key = ri, k
        out[best].append(li)
    return out


def group_orphan_lines(line_boxes: Sequence[Box], indices: Sequence[int],
                       gap_ratio: float = 1.0, min_x_overlap: float = 0.3) -> List[List[int]]:
    """Group lines no layout region claimed into paragraph-like clusters.

    Consecutive lines (top to bottom) join the same cluster when they overlap
    horizontally and the vertical gap is within ``gap_ratio`` line heights.
    """
    clusters: List[List[int]] = []
    for i in sorted(indices, key=lambda i: (line_boxes[i][1], line_boxes[i][0])):
        b = line_boxes[i]
        placed = False
        for c in clusters:
            last = line_boxes[c[-1]]
            h = max(1, min(b[3] - b[1], last[3] - last[1]))
            gap = b[1] - last[3]
            x_ov = min(b[2], last[2]) - max(b[0], last[0])
            if gap <= gap_ratio * h and x_ov >= min_x_overlap * min(b[2] - b[0], last[2] - last[0]):
                c.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def union_box(boxes: Sequence[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
