"""Turn one page's detections (layout regions + text lines) into document blocks.

Two phases, so recognition can run once per page as a single batch:

1. ``plan_page`` decides the page's units in reading order (each unit is a
   layout region with the text lines it owns, or a region that has to be
   recognized as a whole because the line detector missed its text).
2. ``build_blocks`` turns the units plus the recognized texts into blocks,
   splitting text regions into paragraphs and list items.

Everything here is geometry heuristics tuned on scanned Khmer letters and
notices (``test_images/``): letterheads, mottos, numbered lists, stamps and
signatures. The layout model's labels are treated as hints, not ground truth.
"""

import re
from dataclasses import dataclass, field
from statistics import median
from typing import List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from .document import AssetStore
from .layout.base import IMAGE_LABELS, TABLE_LABELS, TEXT_LABELS, Box, LayoutRegion, area, intersection
from .layout.order import assign_lines, group_orphan_lines, union_box, xy_cut_order
from .layout.table import build_grid

CAPTION_LABELS = {"figure_caption", "table_caption", "table_footnote", "formula_caption"}

# 'abandon' regions only become running headers/footers when they sit in the
# page margins. DocLayout-YOLO also labels letterheads, mottos and signature
# blocks 'abandon'; those are part of the document body.
HEADER_ZONE = 0.08   # region bottom within the top 8% of the page
FOOTER_ZONE = 0.88   # region top below 88% of the page

# A figure whose ink is almost all inside detected text lines and logos is a
# block of text (a letterhead, a dated signature line), not a picture.
FIGURE_RESIDUAL_INK = 0.03

# Bullets (including "o" bullets the recognizer reads as "០"/"00"), "1." / "1)"
# and "ក)" numbering. "(1)" is left out: it's common mid-sentence enumeration.
LIST_MARKER = re.compile(r"^\s*(?:[-–—•●○◦▪■*·]|[o០0]{1,2}\s|[0-9០-៩]{1,2}[.)]|[a-zក-អ]\))")


@dataclass
class Unit:
    label: str                      # a DocLayout-YOLO label, or "logo"
    bbox: Box
    lines: List[int] = field(default_factory=list)   # indices into the page's text lines
    recognize_whole: bool = False   # OCR the region crop itself instead of its lines


def _is_khmer(ch: str) -> bool:
    return "ក" <= ch <= "៿" or "᧠" <= ch <= "᧿"


def has_content(text: str) -> bool:
    """False for recognizer output from ornaments and stamps ("@-@.@", "-",
    "@-@ 7 @.@ 5"): no letters, and digits are a minority of the symbols."""
    if any(ch.isalpha() for ch in text):
        return True
    chars = [ch for ch in text if not ch.isspace()]
    digits = sum(ch.isdigit() for ch in chars)
    return digits > 0 and digits >= 0.5 * len(chars)


def _same_row(a: Box, b: Box) -> bool:
    # Khmer line boxes are tall (stacked subscripts and vowels above and
    # below), so boxes of adjacent rows can overlap by half their height.
    ov = min(a[3], b[3]) - max(a[1], b[1])
    return ov >= 0.65 * max(1, min(a[3] - a[1], b[3] - b[1]))


def _height(b: Box) -> int:
    return b[3] - b[1]


def dedupe_lines(line_boxes: Sequence[Box], contain: float = 0.8) -> List[int]:
    """Indices of the line boxes to keep, in their original order.

    The line detector sometimes returns a whole row and a fragment of the
    same row; recognizing both repeats the fragment's text. A box mostly
    (``contain``) inside a larger kept box is dropped.
    """
    order = sorted(range(len(line_boxes)), key=lambda i: -area(line_boxes[i]))
    kept: List[int] = []
    for i in order:
        b = line_boxes[i]
        a = max(1, area(b))
        if not any(intersection(b, line_boxes[k]) >= contain * a for k in kept):
            kept.append(i)
    return sorted(kept)


def consolidate_lines(img_bgr: np.ndarray, boxes: Sequence[Box], crops: Sequence[Image.Image],
                      min_x_overlap: float = 0.25) -> tuple[List[Box], List[Image.Image]]:
    """Clean up the line detector's output before recognition.

    Drops boxes contained in a larger one (``dedupe_lines``) and fuses
    fragments of one row that overlap each other horizontally -- recognizing
    them separately reads the shared words twice. A fused row is re-cropped
    from the page.
    """
    keep = dedupe_lines(boxes)
    boxes = [boxes[i] for i in keep]
    crops = [crops[i] for i in keep]
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            x_ov = min(a[2], b[2]) - max(a[0], b[0])
            y_ov = min(a[3], b[3]) - max(a[1], b[1])
            # Stricter than _same_row: tall Khmer line boxes of adjacent rows
            # overlap a lot vertically, and fusing two rows garbles both.
            if (y_ov >= 0.7 * min(_height(a), _height(b))
                    and x_ov > min_x_overlap * min(a[2] - a[0], b[2] - b[0])):
                parent[find(j)] = find(i)
    groups: dict = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    out_boxes, out_crops = [], []
    for members in sorted(groups.values()):
        if len(members) == 1:
            out_boxes.append(boxes[members[0]])
            out_crops.append(crops[members[0]])
        else:
            box = union_box([boxes[i] for i in members])
            out_boxes.append(box)
            out_crops.append(_crop(img_bgr, box))
    return out_boxes, out_crops


def ink_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of dark (ink) pixels, thresholded for the whole page."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray < min(thr, 200)


def join_lines(boxes: Sequence[Box], texts: Sequence[str], keep_breaks: bool) -> str:
    """Join a unit's lines (already in reading order) into one block of text.

    Fragments on the same visual row are joined with a space. Across rows,
    ``keep_breaks`` keeps a newline (headings, letterhead lines, machine
    readable zones); otherwise wrapped lines are rejoined -- with no space
    between two Khmer characters, since Khmer doesn't put spaces between words.
    """
    out = ""
    prev_box = None
    prev_text = ""
    for box, text in zip(boxes, texts):
        text = text.strip()
        if not text:
            continue
        if out:
            if prev_box is not None and _same_row(prev_box, box):
                sep = " "
            elif keep_breaks or "<<" in text or "<<" in prev_text:
                sep = "\n"
            elif _is_khmer(out[-1]) and _is_khmer(text[0]):
                sep = ""
            else:
                sep = " "
            out += sep
        out += text
        prev_box = box
        prev_text = text
    return out


def _rows(indices: Sequence[int], line_boxes: Sequence[Box]) -> List[List[int]]:
    """Group line indices (in reading order) into visual rows."""
    rows: List[List[int]] = []
    row_box: Optional[Box] = None
    for i in indices:
        b = line_boxes[i]
        if rows and _same_row(row_box, b):
            rows[-1].append(i)
            row_box = union_box([row_box, b])
        else:
            rows.append([i])
            row_box = b
    return rows


def _row_boxes(boxes: Sequence[Box]) -> List[Box]:
    """Merge fragments on the same visual row (boxes in reading order)."""
    rows: List[Box] = []
    for b in boxes:
        if rows and _same_row(rows[-1], b):
            rows[-1] = union_box([rows[-1], b])
        else:
            rows.append(b)
    return rows


def text_extent(line_boxes: Sequence[Box], page_width: int) -> tuple[int, int]:
    """(left, right) of the page's text column.

    Taken from full-length lines only, so short marginal text (letterhead
    corners, footer addresses) doesn't widen the column.
    """
    if not line_boxes:
        return 0, page_width
    widths = sorted(b[2] - b[0] for b in line_boxes)
    p90 = widths[min(len(widths) - 1, int(0.9 * len(widths)))]
    long = [b for b in line_boxes if b[2] - b[0] >= 0.6 * p90]
    return min(b[0] for b in long), max(b[2] for b in long)


def infer_align(bbox: Box, boxes: Sequence[Box], content: tuple[int, int],
                page_width: Optional[int] = None) -> str:
    """Guess alignment from where a block's rows sit in the text column.

    Centred: every row has (about) the same margin on both sides -- of the
    text column or of the page, since scans often sit off-centre -- and isn't
    full width. Right: every row ends at the right margin and starts well
    past the left one. Everything else is left-aligned, including justified
    paragraphs with a short last row or an indented first row.
    """
    left, right = content
    width = max(1, right - left)
    tol = 0.05 * width
    rows = _row_boxes(boxes) or [bbox]

    def centred(r, lo, hi):
        w = max(1, hi - lo)
        li, ri = r[0] - lo, hi - r[2]
        # A full-width row has equal margins too; it isn't centred text.
        return abs(li - ri) <= 0.05 * w and li > 0.08 * w and r[2] - r[0] < 0.8 * width

    if all(centred(r, left, right) for r in rows):
        return "center"
    if page_width and all(centred(r, 0, page_width) for r in rows):
        return "center"
    if all(right - r[2] <= tol and r[0] - left > 0.25 * width for r in rows):
        return "right"
    return "left"


def _figure_is_text(region: LayoutRegion, line_boxes: Sequence[Box], logo_boxes: Sequence[Box],
                    ink: np.ndarray) -> bool:
    """True when a 'figure' is really text (plus logos): nearly all its ink is
    inside detected lines and logos."""
    inside = [b for b in line_boxes if intersection(b, region.bbox) >= 0.5 * max(1, area(b))]
    if not inside:
        return False
    x1, y1, x2, y2 = region.bbox
    mask = ink[y1:y2, x1:x2].copy()
    if mask.size == 0:
        return False
    for b in list(inside) + [b for b in logo_boxes if intersection(b, region.bbox) > 0]:
        bx1, by1 = max(0, b[0] - 3 - x1), max(0, b[1] - 3 - y1)
        bx2, by2 = max(0, b[2] + 3 - x1), max(0, b[3] + 3 - y1)
        mask[by1:by2, bx1:bx2] = False
    return mask.mean() < FIGURE_RESIDUAL_INK


def _page_zone(bbox: Box, page_h: int) -> Optional[str]:
    if bbox[3] <= HEADER_ZONE * page_h:
        return "page_header"
    if bbox[1] >= FOOTER_ZONE * page_h:
        return "page_footer"
    return None


def _merge_continuations(units: List[Unit], line_boxes: Sequence[Box]) -> List[Unit]:
    """Rejoin a paragraph the layout model cut into two regions.

    DocLayout-YOLO often boxes an indented first (or a lone last) line
    separately. Two consecutive plain-text units, one of them a single row,
    are one paragraph when the first one's last row runs to the right margin,
    the second starts at the left margin, and the gap between them is no
    bigger than the line spacing inside the paragraph (a subtitle or a
    one-line paragraph above has extra space below it).
    """
    out: List[Unit] = []
    for u in units:
        prev = out[-1] if out else None
        if (prev is not None and prev.label == u.label == "plain text"
                and prev.lines and u.lines and not prev.recognize_whole and not u.recognize_whole):
            head = union_box([line_boxes[i] for i in _rows(prev.lines, line_boxes)[0]])
            last = union_box([line_boxes[i] for i in _rows(prev.lines, line_boxes)[-1]])
            first = union_box([line_boxes[i] for i in _rows(u.lines, line_boxes)[0]])
            left = min(prev.bbox[0], u.bbox[0])
            right = max(prev.bbox[2], u.bbox[2])
            w = max(1, right - left)
            lh = median([_height(line_boxes[i]) for i in prev.lines + u.lines])
            x_ov = min(prev.bbox[2], u.bbox[2]) - max(prev.bbox[0], u.bbox[0])
            rows_p, rows_u = _rows(prev.lines, line_boxes), _rows(u.lines, line_boxes)
            multi = rows_u if len(rows_u) > 1 else rows_p
            row_b = [union_box([line_boxes[i] for i in r]) for r in multi]
            inner_gaps = [b[1] - a[3] for a, b in zip(row_b, row_b[1:])]
            single = min(len(rows_p), len(rows_u)) == 1 and inner_gaps
            if (single and first[1] - last[3] <= median(inner_gaps) + 0.2 * lh
                    and head[0] <= left + 0.08 * w          # indented at most, not centred
                    and x_ov >= 0.6 * min(prev.bbox[2] - prev.bbox[0], u.bbox[2] - u.bbox[0])
                    and last[2] >= right - 0.12 * w
                    and first[0] <= left + max(0.02 * w, 0.5 * lh)):
                prev.lines = prev.lines + u.lines
                prev.bbox = union_box([prev.bbox, u.bbox])
                continue
        out.append(u)
    return out


def _density(ink: np.ndarray, b: Box) -> float:
    region = ink[b[1]:b[3], b[0]:b[2]]
    return float(region.mean()) if region.size else 0.0


def _promote_titles(units: List[Unit], line_boxes: Sequence[Box], content: tuple[int, int],
                    page_width: int, ink: Optional[np.ndarray]) -> None:
    """Short, centred text in large or bold type becomes a title.

    Catches titles the layout model labelled as plain text or missed (then
    they arrive here as orphan lines). Bold shows up as ink density.
    """
    if not line_boxes:
        return
    body_h = median(_height(b) for b in line_boxes)
    body_ink = median(_density(ink, b) for b in line_boxes) if ink is not None else None
    left, right = content
    for u in units:
        if u.label != "plain text" or not u.lines or u.recognize_whole:
            continue
        if len(_rows(u.lines, line_boxes)) > 2:
            continue
        boxes = [line_boxes[i] for i in u.lines]
        if infer_align(u.bbox, boxes, content, page_width) != "center":
            continue
        large = median(_height(b) for b in boxes) >= 1.3 * body_h
        bold = (body_ink is not None and body_ink > 0
                and median(_density(ink, b) for b in boxes) >= 1.35 * body_ink)
        if large or bold:
            u.label = "title"


def _merge_titles(units: List[Unit], line_boxes: Sequence[Box]) -> List[Unit]:
    """Stacked title lines ("Notice" / "on" / "the subject") are one heading."""
    out: List[Unit] = []
    for u in units:
        prev = out[-1] if out else None
        if (prev is not None and prev.label == u.label == "title" and prev.lines and u.lines
                and not prev.recognize_whole and not u.recognize_whole):
            lh = median([_height(line_boxes[i]) for i in prev.lines + u.lines])
            gap = min(line_boxes[i][1] for i in u.lines) - max(line_boxes[i][3] for i in prev.lines)
            x_ov = min(prev.bbox[2], u.bbox[2]) - max(prev.bbox[0], u.bbox[0])
            if gap <= 0.8 * lh and x_ov > 0:
                prev.lines = prev.lines + u.lines
                prev.bbox = union_box([prev.bbox, u.bbox])
                continue
        out.append(u)
    return out


def plan_page(page_size, line_boxes: Sequence[Box], logo_boxes: Sequence[Box],
              regions: Sequence[LayoutRegion], include_headers_footers: bool = True,
              ink: Optional[np.ndarray] = None) -> List[Unit]:
    """Group a page's lines under layout regions and return units in reading order.

    ``ink`` (see ``ink_mask``) lets figures that are really text be dissolved
    into text; without it figures are always kept as pictures.
    """
    w, h = page_size
    regions = list(regions)

    # Letterheads/signatures labelled 'abandon' outside the margins are body text.
    regions = [LayoutRegion("plain text", r.bbox, r.score)
               if r.label == "abandon" and _page_zone(r.bbox, h) is None else r
               for r in regions]
    if ink is not None:
        regions = [r for r in regions
                   if not (r.label == "figure" and _figure_is_text(r, line_boxes, logo_boxes, ink))]

    assignment = assign_lines(line_boxes, regions)
    line_h = median([_height(b) for b in line_boxes]) if line_boxes else None

    units: List[Unit] = []
    for ri, region in enumerate(regions):
        owned = assignment[ri]
        label = region.label
        if label in IMAGE_LABELS:
            # Text the line detector found inside a real figure (labels,
            # stamps) is part of the picture; it's kept as the image.
            units.append(Unit(label, region.bbox))
        elif label in TABLE_LABELS:
            units.append(Unit(label, region.bbox, owned))
        elif label in TEXT_LABELS:
            if label == "abandon" and not include_headers_footers:
                continue
            x1, y1, x2, y2 = region.bbox
            rw, rh = x2 - x1, y2 - y1
            single_line = rh > 0 and rw / rh >= 2.5 and (line_h is None or rh <= 2 * line_h)
            if owned:
                boxes = [line_boxes[i] for i in owned]
                one_row = len(_row_boxes(sorted(boxes, key=lambda b: b[0]))) == 1
                covered = sum(b[2] - b[0] for b in boxes)
                if single_line and one_row and covered < 0.7 * rw:
                    # The detector caught only part of a one-line region
                    # (typical for small footer text): read the whole region.
                    units.append(Unit(label, region.bbox, recognize_whole=True))
                else:
                    units.append(Unit(label, union_box(boxes), owned))
            elif single_line and label != "abandon":
                # The line detector missed this region entirely; read it as one
                # line. Taller empty regions are usually decoration.
                units.append(Unit(label, region.bbox, recognize_whole=True))

    for cluster in group_orphan_lines(line_boxes, assignment[-1]):
        units.append(Unit("plain text", union_box([line_boxes[i] for i in cluster]), cluster))

    figure_boxes = [r.bbox for r in regions if r.label in IMAGE_LABELS]
    for lb in logo_boxes:
        if any(intersection(lb, fb) > 0.5 * area(lb) for fb in figure_boxes):
            continue
        units.append(Unit("logo", lb))

    ordered = xy_cut_order(units, key=lambda u: u.bbox)
    for u in ordered:
        u.lines = xy_cut_order(u.lines, key=lambda i: line_boxes[i])
    _promote_titles(ordered, line_boxes, text_extent(line_boxes, w), w, ink)
    ordered = _merge_continuations(ordered, line_boxes)
    return _merge_titles(ordered, line_boxes)


def _crop(img_bgr: np.ndarray, box: Box) -> Image.Image:
    x1, y1, x2, y2 = box
    return Image.fromarray(cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))


def region_crops(img_bgr: np.ndarray, units: Sequence[Unit]) -> List[Image.Image]:
    """Crops to recognize for the ``recognize_whole`` units, in unit order."""
    return [_crop(img_bgr, u.bbox) for u in units if u.recognize_whole]


def split_paragraphs(indices: Sequence[int], line_boxes: Sequence[Box],
                     line_texts: Sequence[str],
                     content_width: Optional[int] = None) -> List[tuple[List[int], bool]]:
    """Split a text region's lines into paragraphs.

    Returns ``[(line indices, keep_breaks)]``. A row that stops well short of
    the region's right edge ends its paragraph (last line of a paragraph, a
    one-line list item, an address line), and a row starting with a list
    marker starts a new one. Rows centred on each other with different
    widths (a letterhead, a dated signature), and blocks much narrower than
    the text column (a motto, a sender's address), stay one block with line
    breaks: their rows are separate lines, not wrapped text.
    """
    rows = _rows(indices, line_boxes)
    if len(rows) <= 1:
        return [(list(indices), False)]
    boxes = [union_box([line_boxes[i] for i in r]) for r in rows]
    left = min(b[0] for b in boxes)
    right = max(b[2] for b in boxes)
    width = max(1, right - left)
    lh = median(_height(b) for b in boxes)

    centre = (left + right) / 2
    if content_width and width < 0.4 * content_width:
        return [(list(indices), True)]
    if (all(abs((b[0] + b[2]) / 2 - centre) <= 0.05 * width for b in boxes)
            and max(b[0] for b in boxes) - left > 0.05 * width):
        return [(list(indices), True)]

    paras: List[List[int]] = [list(rows[0])]
    for prev, row, box in zip(boxes, rows[1:], boxes[1:]):
        short = prev[2] < right - max(0.15 * width, 2 * lh)
        marker = bool(LIST_MARKER.match(line_texts[row[0]]))
        if short or marker:
            paras.append(list(row))
        else:
            paras[-1].extend(row)
    return [(p, False) for p in paras]


def build_blocks(page_index: int, img_bgr: np.ndarray, units: Sequence[Unit],
                 line_boxes: Sequence[Box], line_texts: Sequence[str],
                 whole_texts: Sequence[str], assets: AssetStore) -> List[dict]:
    h, w = img_bgr.shape[:2]
    content = text_extent(line_boxes, w)
    whole_iter = iter(whole_texts)
    blocks: List[dict] = []

    # Heading level from relative text size: the page's tallest title lines
    # are H1, noticeably smaller titles are H2.
    def unit_line_height(u: Unit) -> float:
        if u.lines:
            return median([_height(line_boxes[i]) for i in u.lines])
        return _height(u.bbox)
    title_heights = [unit_line_height(u) for u in units if u.label == "title"]
    max_title_h = max(title_heights) if title_heights else 0

    def add(block: dict) -> None:
        block["id"] = f"p{page_index}-b{len(blocks)}"
        blocks.append(block)

    def text_block(u: Unit, btype: str, boxes, texts, keep_breaks: bool,
                   align: Optional[str] = None, **extra) -> None:
        text = join_lines(boxes, texts, keep_breaks=keep_breaks)
        if not text.strip():
            return
        bbox = union_box(boxes) if boxes else u.bbox
        add({"page": page_index, "bbox": [int(v) for v in bbox], "label": u.label,
             "type": btype, **extra, "text": text,
             "align": align or infer_align(bbox, boxes, content, w),
             "lines": [{"text": t, "bbox": [int(v) for v in b]} for b, t in zip(boxes, texts)]})

    for n, u in enumerate(units):
        base = {"page": page_index, "bbox": [int(v) for v in u.bbox], "label": u.label}
        if u.label in IMAGE_LABELS or u.label == "logo":
            asset = assets.put(f"p{page_index}-img{n}", _crop(img_bgr, u.bbox))
            btype = "formula" if u.label == "isolate_formula" else "figure"
            add({**base, "type": btype, "image": asset})
            continue

        if u.recognize_whole:
            idx, boxes, texts = [], [u.bbox], [next(whole_iter)]
        else:
            idx = [i for i in u.lines if has_content(line_texts[i])]
            boxes = [line_boxes[i] for i in idx]
            texts = [line_texts[i] for i in idx]
        if not has_content(" ".join(texts)):
            continue

        if u.label in TABLE_LABELS:
            asset = assets.put(f"p{page_index}-table{n}", _crop(img_bgr, u.bbox))
            grid = build_grid(list(zip(boxes, texts)))
            rows = grid if grid is not None else [[join_lines(boxes, texts, keep_breaks=True)]]
            add({**base, "type": "table", "rows": rows, "image": asset,
                 "lines": [{"text": t, "bbox": [int(v) for v in b]} for b, t in zip(boxes, texts)]})
        elif u.label == "title":
            level = 1 if unit_line_height(u) >= 0.85 * max_title_h else 2
            text_block(u, "heading", boxes, texts, True, level=level)
        elif u.label in CAPTION_LABELS:
            text_block(u, "caption", boxes, texts, False)
        elif u.label == "abandon":
            text_block(u, _page_zone(u.bbox, h) or "page_header", boxes, texts, True)
        elif u.recognize_whole:
            text_block(u, "paragraph", boxes, texts, False)
        else:
            paras = split_paragraphs(idx, line_boxes, line_texts, content[1] - content[0])
            # Paragraphs split out of one region share its alignment: a short
            # list item isn't centred just because it happens to sit mid-page.
            align = infer_align(u.bbox, boxes, content, w) if len(paras) > 1 else None
            for para, keep_breaks in paras:
                text_block(u, "paragraph", [line_boxes[i] for i in para],
                           [line_texts[i] for i in para], keep_breaks, align=align)
    return blocks


def plain_blocks(page_index: int, img_bgr: np.ndarray, line_boxes: Sequence[Box],
                 line_texts: Sequence[str], logo_boxes: Sequence[Box],
                 assets: AssetStore) -> List[dict]:
    """Layout-free structure: one paragraph per detected line, logos as figures.

    Same top-to-bottom order the classic pipeline has always used.
    """
    items = [("text", b, t) for b, t in zip(line_boxes, line_texts) if has_content(t)]
    items += [("logo", b, None) for b in logo_boxes]
    items.sort(key=lambda it: (it[1][1], it[1][0]))
    blocks = []
    for n, (kind, box, text) in enumerate(items):
        base = {"id": f"p{page_index}-b{n}", "page": page_index, "bbox": [int(v) for v in box]}
        if kind == "logo":
            asset = assets.put(f"p{page_index}-img{n}", _crop(img_bgr, box))
            blocks.append({**base, "type": "figure", "image": asset, "label": "logo"})
        else:
            blocks.append({**base, "type": "paragraph", "text": text, "label": "line",
                           "lines": [{"text": text, "bbox": base["bbox"]}]})
    return blocks
