"""Turn an enriched page into a layout plan any writer can follow.

The plan describes the page the way a word processor builds one: text
flows top to bottom with exact spacing, side-by-side blocks (a ministry
letterhead opposite the kingdom motto) become columns of one row, and
pictures (logos, seals, signatures) float at fixed page positions. All
lengths are in points.

Vertical placement is reproduced with exact line pitch and exact space
before each paragraph, measured from the scan, so each line lands where
it was without absolute positioning, and the text still flows if edited.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..document import TEXT_TYPES
from . import fonts as F

DEFAULT_STYLE = {"font": F.DEFAULT_FAMILY, "size": 11.0, "bold": False, "color": "000000"}
SCALE_RANGE = (70, 130)        # Word character scale limits we allow, %
FIT_MARGIN = 0.97              # keep each line a little narrower than its room
                               # (Latin text is set in Arial, whose widths differ a little)
# A justified line is stretched to the column by widening its spaces, and
# Khmer lines have few spaces: left to justification alone, the slack piles
# into one or two wide gaps. This share of it goes into the letters instead
# (no further than JUSTIFY_MAX_SCALE); measured on 13 test pages, Word's gaps
# then come out as wide as the scan's.
JUSTIFY_SHARE = 0.5
JUSTIFY_MAX_SCALE = 110


@dataclass
class Row:
    text: str
    x1: float                  # pt, page coordinates
    x2: float
    centre: float              # vertical centre of the core band, pt
    scale: int = 100           # character scale, %


@dataclass
class Para:
    """One paragraph: a text block with its rows (source lines)."""
    block: dict
    rows: List[Row]
    style: dict
    align: str
    pitch: float               # exact line pitch, pt
    top: float                 # top of the first line's slot, pt (page coordinates)
    bottom: float
    keep_lines: bool = True    # rows end in line breaks (False: reflow the block text)
    text: str = ""             # used when not keep_lines
    space_before: float = 0.0
    left: float = 0.0          # indents, pt, relative to the container
    right: float = 0.0
    first: float = 0.0


@dataclass
class Cell:
    lines: List[str]
    align: str = "left"        # left | center | right
    scale: int = 100           # character scale for every line, %
    valign: str = "center"     # top | center | bottom


@dataclass
class Table:
    block: dict
    cells: List[List[Cell]]
    col_widths: List[float]
    row_heights: List[float]   # exact, pt: a table never grows taller than on the scan
    style: dict
    header_style: Optional[dict]
    top: float
    bottom: float
    left: float                # pt, page x of the table's left edge
    borders: bool = False
    space_before: float = 0.0
    padding: float = 2.0       # cell padding left and right, pt


@dataclass
class Column:
    x1: float
    x2: float
    items: List[Para] = field(default_factory=list)


@dataclass
class Band:
    """A horizontal slice of the page: one paragraph or table in the flow,
    or several columns side by side."""
    top: float
    bottom: float
    item: Optional[object] = None          # Para | Table
    columns: List[Column] = field(default_factory=list)
    space_before: float = 0.0


@dataclass
class Float:
    image: str                 # asset id
    x: float                   # pt, page coordinates
    y: float
    width: float
    height: float
    kind: str = "figure"
    block_id: Optional[str] = None   # the figure block it shows, if any


@dataclass
class PagePlan:
    index: int
    width: float
    height: float
    margins: tuple             # (left, right, top, bottom), pt
    bands: List[Band]
    floats: List[Float]


# ── helpers ──────────────────────────────────────────────────────────────

def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _rows(lines: Sequence[dict], to_pt, pitch_hint: float) -> List[Row]:
    """Group a block's lines into visual rows (pieces of one row side by
    side join with a space), in reading order."""
    items = []
    for line in lines:
        text = line.get("text", "")
        if not text.strip():
            continue
        box = line.get("ink") or line.get("bbox")
        if not box:
            continue
        if line.get("ink"):
            centre = (box[1] + box[3]) / 2
        else:
            centre = (box[1] + box[3]) / 2
        items.append((centre, box[0], box[2], text, line.get("scale")))
    items.sort()
    rows: List[list] = []
    for it in items:
        if rows and abs(it[0] - rows[-1][-1][0]) < 0.35 * pitch_hint:
            rows[-1].append(it)
        else:
            rows.append([it])
    out = []
    for r in rows:
        r.sort(key=lambda it: it[1])
        scales = [it[4] for it in r if it[4]]
        out.append(Row(text=" ".join(it[3].strip() for it in r),
                       x1=to_pt(min(it[1] for it in r)), x2=to_pt(max(it[2] for it in r)),
                       centre=to_pt(float(np.mean([it[0] for it in r]))),
                       scale=int(np.median(scales)) if scales else 100))
    return out


def natural_pitch(style: dict) -> float:
    """A font's own line height at a size (pt): ascent + descent."""
    fam = F.FAMILIES.get(style["font"], F.FAMILIES[F.DEFAULT_FAMILY])
    font = F._font(fam.file(style.get("bold", False)), F.REF_PX)
    asc, desc = font.getmetrics()
    return (asc + desc) / F.REF_PX * style["size"]


def _style(block: dict) -> dict:
    st = dict(DEFAULT_STYLE)
    st.update(block.get("style") or {})
    if st["font"] not in F.FAMILIES:
        st["font"] = F.DEFAULT_FAMILY
    return st


def _justified(rows: List[Row], tol: float) -> bool:
    if len(rows) < 3:
        return False
    right = max(r.x2 for r in rows)
    body = rows[:-1]
    return sum(right - r.x2 <= tol for r in body) >= 0.75 * len(body)


# ── building ─────────────────────────────────────────────────────────────

def _para(block: dict, to_pt, page_w_pt: float) -> Optional[Para]:
    st = _style(block)
    hint = st["size"] * 1.6
    rows = _rows(block.get("lines") or [], to_pt, hint)
    text = block.get("text", "")
    if not rows:
        if not text.strip() or not block.get("bbox"):
            return None
        x1, y1, x2, y2 = (to_pt(v) for v in block["bbox"])
        rows = [Row(text=t, x1=x1, x2=x2, centre=(y1 + y2) / 2) for t in text.split("\n")[:1]]
    centres = [r.centre for r in rows]
    if len(rows) >= 2:
        pitch = float(np.median(np.diff(centres)))
    else:
        pitch = natural_pitch(st)
    pitch = max(pitch, st["size"] * 1.1)
    # An edited block no longer matches its source lines: reflow its text.
    keep = _squash(text) == _squash("".join(r.text for r in rows))
    align = block.get("align", "left")
    tol = 0.015 * page_w_pt
    if align == "left" and _justified(rows, tol):
        align = "justify"
    elif len(rows) >= 2 and align != "justify":
        # Letterhead lines are each centred on one axis (a ministry name
        # over its department, the motto under "Kingdom of Cambodia").
        centres = [(r.x1 + r.x2) / 2 for r in rows]
        lefts = [r.x1 for r in rows]
        rights = [r.x2 for r in rows]
        if (max(centres) - min(centres) <= 1.5 * tol
                and max(lefts) - min(lefts) > 2 * tol and max(rights) - min(rights) > 2 * tol):
            align = "center"
    return Para(block=block, rows=rows, style=st, align=align, pitch=pitch,
                top=rows[0].centre - pitch / 2, bottom=rows[-1].centre + pitch / 2,
                keep_lines=keep, text=text)


def _cell_align(line_box, cell_x1: float, cell_x2: float) -> str:
    """Where a line sits in its cell: centred, right or left aligned."""
    w = max(1.0, cell_x2 - cell_x1)
    gap_l, gap_r = line_box[0] - cell_x1, cell_x2 - line_box[2]
    # Compare the gaps, not the centres: long text in a wide cell has its
    # centre near the cell's even when it starts at the left edge.
    if abs(gap_l - gap_r) <= max(3.0, 0.25 * (gap_l + gap_r)):
        return "center"
    if gap_r < gap_l and gap_r < 0.1 * w:
        return "right"
    return "left"


def _table(block: dict, to_pt) -> Optional[Table]:
    from ..layout.table import _columns
    st = _style(block)
    header = dict(st, **block["header_style"]) if block.get("header_style") else None
    lines = [l for l in block.get("lines") or [] if l.get("bbox")]
    rows = block.get("rows") or [[""]]
    nrows, ncols = len(rows), max(len(r) for r in rows)

    grid = block.get("grid")
    if grid and len(grid["rows"]) - 1 == nrows and len(grid["cols"]) - 1 == ncols:
        ys, xs = list(grid["rows"]), list(grid["cols"])          # the drawn grid, exactly
    else:
        header = None
        if block.get("bbox"):
            x1, y1, x2, y2 = block["bbox"]
        elif lines:
            x1 = min(l["bbox"][0] for l in lines); x2 = max(l["bbox"][2] for l in lines)
            y1 = min(l["bbox"][1] for l in lines); y2 = max(l["bbox"][3] for l in lines)
        else:
            return None
        xs = [x1 + (x2 - x1) * i / ncols for i in range(ncols + 1)]
        if lines and ncols > 1:
            heights = sorted(l["bbox"][3] - l["bbox"][1] for l in lines)
            gap = max(2, heights[len(heights) // 2] // 3)
            cols = _columns([(tuple(l["bbox"]), l.get("text", "")) for l in lines], gap)
            if len(cols) == ncols:
                xs = [x1] + [(cols[i][1] + cols[i + 1][0]) / 2 for i in range(ncols - 1)] + [x2]
        ys = [y1 + (y2 - y1) * i / nrows for i in range(nrows + 1)]

    # Alignment of each cell from where its text sits on the scan (a
    # barcode's number sits at the bottom of its cell, under the bars).
    aligns = [["left"] * ncols for _ in range(nrows)]
    valigns = [["center"] * ncols for _ in range(nrows)]
    spans: dict = {}
    for l in lines:
        box = l.get("ink") or l["bbox"]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        r = next((i for i in range(nrows) if cy < ys[i + 1]), nrows - 1)
        c = next((i for i in range(ncols) if cx < xs[i + 1]), ncols - 1)
        aligns[r][c] = _cell_align(box, xs[c], xs[c + 1])
        lo, hi = spans.get((r, c), (box[1], box[3]))
        spans[(r, c)] = (min(lo, box[1]), max(hi, box[3]))
    for (r, c), (lo, hi) in spans.items():
        h = ys[r + 1] - ys[r]
        offset = ((lo + hi) / 2 - (ys[r] + ys[r + 1]) / 2) / max(1, h)
        valigns[r][c] = "bottom" if offset > 0.18 else ("top" if offset < -0.18 else "center")

    widths = [max(6.0, to_pt(xs[i + 1] - xs[i])) for i in range(ncols)]
    heights = [max(4.0, to_pt(ys[i + 1] - ys[i])) for i in range(nrows)]
    borders = bool(block.get("ruled"))
    pad = 2.0 if borders else 0.0
    cells = []
    for ri, row in enumerate(rows):
        cst = header if (ri == 0 and header) else st
        out = []
        for ci in range(ncols):
            text = row[ci] if ci < len(row) else ""
            parts = text.split("\n") if text else [""]
            room = widths[ci] - 2 * pad
            natural = max((F.text_width_pt(t, cst["font"], cst["bold"], cst["size"]) for t in parts),
                          default=0.0)
            # Shrink a line that wouldn't fit its cell rather than let it
            # wrap: a wrapped cell makes the row, and the page, taller.
            scale = 100 if natural <= 0 else int(np.clip(100 * FIT_MARGIN * room / natural, 50, 100))
            out.append(Cell(parts, aligns[ri][ci], scale, valigns[ri][ci]))
        cells.append(out)
    return Table(block=block, cells=cells, col_widths=widths, row_heights=heights, style=st,
                 header_style=header, top=to_pt(ys[0]), bottom=to_pt(ys[-1]), left=to_pt(xs[0]),
                 borders=borders, padding=pad)


def _fit(para: Para, room: float) -> None:
    """Set each row's character scale so it is as wide as on the scan, and
    never wider than its room (a wrapped line would push the page down)."""
    st = para.style
    for i, r in enumerate(para.rows):
        natural = F.text_width_pt(r.text, st["font"], st["bold"], st["size"])
        if natural <= 0:
            continue
        want = 100 * (r.x2 - r.x1) / natural
        if para.align == "justify" and i < len(para.rows) - 1 and want > 100:
            # justification stretches it the rest of the way
            want = min(100 + JUSTIFY_SHARE * (want - 100), JUSTIFY_MAX_SCALE)
        avail = room - para.left - max(0.0, para.first if i == 0 else 0.0) - para.right
        if i == 0 and para.first < 0:
            avail -= para.first
        cap = 100 * FIT_MARGIN * avail / natural if avail > 0 else want
        r.scale = int(np.clip(round(min(want, cap)), *SCALE_RANGE))


def _indents(para: Para, left: float, right: float) -> None:
    """Indents relative to a container spanning [left, right] (pt)."""
    rows = para.rows
    x1 = min(r.x1 for r in rows)
    x2 = max(r.x2 for r in rows)
    if para.align == "center":
        c = (x1 + x2) / 2
        d = (c - left) - (right - c)
        para.left, para.right = max(0.0, d), max(0.0, -d)
    elif para.align == "right":
        para.left, para.right = 0.0, max(0.0, right - x2)
    else:
        rest = min(r.x1 for r in rows[1:]) if len(rows) > 1 else rows[0].x1
        para.left = max(0.0, rest - left)
        para.first = rows[0].x1 - rest if len(rows) > 1 else 0.0
        if para.left + para.first < 0:
            para.first = -para.left
        if para.align == "justify" and len(rows) > 1:
            para.right = max(0.0, right - max(r.x2 for r in rows[:-1]))
    _fit(para, right - left)


def _part(p: Para, rows: List[Row]) -> Para:
    """A paragraph made of some of ``p``'s rows."""
    return Para(block=p.block, rows=rows, style=p.style, align=p.align, pitch=p.pitch,
                top=rows[0].centre - p.pitch / 2, bottom=rows[-1].centre + p.pitch / 2,
                keep_lines=True, text="\n".join(r.text for r in rows))


def _split_beside(items: list) -> list:
    """Split a tall paragraph where a shorter block beside it ends.

    Layout analysis sometimes merges a letterhead column with the titles
    under it (the kingdom motto and the title block, say), while the
    ministry's letterhead sits beside only the top rows. Kept whole, the two
    can't be columns (the titles are wider than the motto) and would be
    stacked; split, the top rows pair with the letterhead and the rest
    flows on below.
    """
    changed = True
    while changed:
        changed = False
        for a in items:
            if not isinstance(a, Para) or len(a.rows) < 2 or not a.keep_lines:
                continue
            for b in items:
                if b is a:
                    continue
                b_top, b_bot = _core(b)
                if b_bot <= a.rows[0].centre - a.pitch / 2 or b_top >= a.rows[0].centre + a.pitch / 2 + (
                        a.rows[-1].centre - a.rows[0].centre):
                    continue                       # no vertical overlap
                k = sum(1 for r in a.rows if r.centre - a.pitch * 0.3 < b_bot)
                if 0 < k < len(a.rows) and b_top >= a.rows[0].centre - a.pitch:
                    items.remove(a)
                    items.extend([_part(a, a.rows[:k]), _part(a, a.rows[k:])])
                    changed = True
                    break
            if changed:
                break
    return items


def _split_irregular(p: Para, tol: float) -> List[Para]:
    """Split a left or justified paragraph where its rows stop lining up.

    Word gives every line of a paragraph after the first the same left edge,
    and stretches every justified line that ends in a line break to the
    right edge. A row the scan set elsewhere -- centred under the others,
    indented, or a justified row that stops short -- would come out moved
    and stretched, with wide gaps between its words. Such a row gets a
    paragraph of its own (where it is, and as a paragraph's last line it is
    never stretched); the rows after it start a new one.
    """
    rows = p.rows
    if p.align not in ("left", "justify") or not p.keep_lines or len(rows) < 2:
        return [p]
    left = float(np.median([r.x1 for r in rows[1:]]))
    right = max(r.x2 for r in rows[:-1])
    off = [abs(r.x1 - left) > 2 * tol for r in rows]
    parts = [[rows[0]]]
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        short = p.align == "justify" and prev.x2 < right - 3 * tol
        if off[i] or (off[i - 1] and i - 1 > 0) or short:
            parts.append([])
        parts[-1].append(rows[i])
    if len(parts) == 1:
        return [p]
    return [_part(p, rs) for rs in parts]


def _band_columns(items: list, gap: float) -> List[list]:
    """Split a band's items into side-by-side columns by x gaps."""
    spans = sorted(items, key=lambda it: _x_extent(it)[0])
    cols: List[list] = []
    right = None
    for it in spans:
        a, b = _x_extent(it)
        if cols and a <= right + gap:
            cols[-1].append(it)
            right = max(right, b)
        else:
            cols.append([it])
            right = b
    return cols


def _x_extent(item):
    if isinstance(item, Table):
        return item.left, item.left + sum(item.col_widths)
    return min(r.x1 for r in item.rows), max(r.x2 for r in item.rows)


def _core(item):
    """Vertical extent used to decide what sits side by side: the line
    slots shrunk a little, since padded boxes of stacked blocks overlap."""
    if isinstance(item, Table):
        return item.top, item.bottom
    return item.rows[0].centre - item.pitch * 0.3, item.rows[-1].centre + item.pitch * 0.3


def build(page: dict, blocks: Sequence[dict]) -> PagePlan:
    """Plan one enriched page. ``blocks`` are that page's blocks."""
    dpi = page.get("dpi") or 96.0
    size = page.get("size_pt") or [page["width"] * 72 / dpi, page["height"] * 72 / dpi]
    page_w, page_h = float(size[0]), float(size[1])
    # The scan is centred horizontally on the sheet when it was fitted by height.
    ox = max(0.0, (page_w - page["width"] * 72 / dpi) / 2)

    def to_pt(v):
        return float(v) * 72 / dpi

    def shift(x):
        return x + ox

    items, floats = [], []
    # Paragraphs added in the editor have no place on the scan: each goes
    # right after the block before it (or first on the page), in the flow.
    added: List[tuple] = []            # (previous block or None, block)
    prev = None
    for b in blocks:
        if b["type"] in TEXT_TYPES:
            p = _para(b, to_pt, page_w)
            if p:
                items.append(p)
            elif b.get("text", "").strip() and not b.get("bbox") and not b.get("lines"):
                added.append((prev, b))
        elif b["type"] == "table":
            t = _table(b, to_pt)
            if t:
                items.append(t)
        elif b.get("image") and b.get("bbox"):
            x1, y1, x2, y2 = (to_pt(v) for v in b["bbox"])
            floats.append(Float(b["image"], shift(x1), y1, x2 - x1, y2 - y1, "figure", b.get("id")))
        prev = b
    for ov in page.get("overlays") or []:
        x1, y1, x2, y2 = (to_pt(v) for v in ov["bbox"])
        floats.append(Float(ov["image"], shift(x1), y1, x2 - x1, y2 - y1, ov.get("kind", "mark")))

    for it in items:
        if isinstance(it, Para):
            for r in it.rows:
                r.x1, r.x2 = shift(r.x1), shift(r.x2)
        else:
            it.left = shift(it.left)

    if not items and not added:
        return PagePlan(page.get("index", 0), page_w, page_h, (36.0, 36.0, 36.0, 36.0), [], floats)
    if not items:
        margin = 0.1 * page_w
        bands = []
        for _, b in added:
            bands.append(_added_band(b, page, margin, page_w - margin))
        return PagePlan(page.get("index", 0), page_w, page_h, (margin, margin, margin, 36.0), bands, floats)

    # Content box: the sheet's margins are where the text is on the scan.
    left = min(_x_extent(it)[0] for it in items)
    right = max(_x_extent(it)[1] for it in items)
    left = max(6.0, left)
    right = min(page_w - 6.0, right)

    items = _split_beside(items)
    tol = 0.015 * page_w
    items = [part for it in items
             for part in (_split_irregular(it, tol) if isinstance(it, Para) else [it])]

    # Bands: items whose cores overlap vertically sit side by side.
    items.sort(key=lambda it: _core(it)[0])
    groups: List[list] = []
    bottom = None
    for it in items:
        top_i, bot_i = _core(it)
        if groups and top_i < bottom:
            groups[-1].append(it)
            bottom = max(bottom, bot_i)
        else:
            groups.append([it])
            bottom = bot_i

    bands: List[Band] = []
    gap = 0.015 * page_w
    for g in groups:
        cols = _band_columns(g, gap) if len(g) > 1 else [g]
        if len(cols) == 1:
            # One column: its items simply follow one another in the flow.
            for it in sorted(g, key=lambda it: it.top):
                if isinstance(it, Para):
                    _indents(it, left, right)
                bands.append(Band(top=it.top, bottom=it.bottom, item=it))
            continue
        tables = [it for it in g if isinstance(it, Table)]
        if tables:
            # A table beside text is rare and hard to rebuild faithfully as
            # nested tables; stack them instead.
            for it in sorted(g, key=lambda it: it.top):
                if isinstance(it, Para):
                    _indents(it, left, right)
                bands.append(Band(top=it.top, bottom=it.bottom, item=it))
            continue
        extents = [(min(_x_extent(it)[0] for it in c), max(_x_extent(it)[1] for it in c)) for c in cols]
        edges = [left] + [(extents[i][1] + extents[i + 1][0]) / 2 for i in range(len(cols) - 1)] + [right]
        columns = []
        for i, c in enumerate(cols):
            col = Column(edges[i], edges[i + 1], sorted(c, key=lambda it: it.top))
            for it in col.items:
                _indents(it, col.x1, col.x2)
            columns.append(col)
        bands.append(Band(top=min(it.top for it in g), bottom=max(it.bottom for it in g),
                          columns=columns))

    # Exact spacing: each band starts where it started on the scan, after
    # whatever the previous band actually took up.
    bands.sort(key=lambda b: b.top)
    for before, b in added:
        at = _band_after(bands, before)
        band = _added_band(b, page, left, right)
        band.top = band.bottom = None            # follows the flow, wherever that is
        bands.insert(at, band)
    top_margin = max(0.0, next(b.top for b in bands if b.top is not None))
    cursor = top_margin
    for band in bands:
        if band.top is None:                     # an added paragraph: no gap of its own
            band.space_before = 0.0
            band.item.space_before = 0.0
            cursor += band.item.bottom - band.item.top
            continue
        band.space_before = max(0.0, band.top - cursor)
        start = cursor + band.space_before
        if band.item is not None:
            band.item.space_before = 0.0
            height = band.item.bottom - band.item.top
        else:
            height = 0.0
            for col in band.columns:
                y = start
                for it in col.items:
                    it.space_before = max(0.0, it.top - y)
                    y += it.space_before + (it.bottom - it.top)
                height = max(height, y - start)
        cursor = start + height

    bottom_margin = float(np.clip((page_h - cursor) / 2, 0.0, 36.0))
    margins = (left, page_w - right, top_margin, bottom_margin)
    return PagePlan(page.get("index", 0), page_w, page_h, margins, bands, floats)


def _added_band(block: dict, page: dict, left: float, right: float) -> Band:
    """A paragraph typed in the editor: the page's body face and size,
    reflowed across the text column."""
    st = dict(DEFAULT_STYLE)
    st["font"] = page.get("body_font") if page.get("body_font") in F.FAMILIES else F.DEFAULT_FAMILY
    st["size"] = float(page.get("body_size") or DEFAULT_STYLE["size"])
    st.update(block.get("style") or {})
    pitch = natural_pitch(st)
    text = block.get("text", "")
    row = Row(text=text.split("\n")[0], x1=left, x2=right, centre=pitch / 2)
    para = Para(block=block, rows=[row], style=st, align=block.get("align", "left"), pitch=pitch,
                top=0.0, bottom=pitch, keep_lines=False, text=text)
    return Band(top=0.0, bottom=pitch, item=para)


def _band_after(bands: List[Band], block: Optional[dict]) -> int:
    """Index just after the band holding ``block`` (0 when there is none)."""
    if block is None:
        return 0
    for i, band in enumerate(bands):
        items = [band.item] if band.item is not None else [it for c in band.columns for it in c.items]
        if any(it.block is block for it in items):
            return i + 1
    return len(bands)


# ── serialisation (the web editor draws the same plan the writer follows) ──

def _para_dict(p: Para) -> dict:
    return {"kind": "para", "id": p.block.get("id"), "style": p.style, "align": p.align,
            "pitch": p.pitch, "keep_lines": p.keep_lines, "text": p.text,
            "rows": [{"text": r.text, "scale": r.scale} for r in p.rows],
            "left": p.left, "right": p.right, "first": p.first, "space_before": p.space_before}


def _table_dict(t: Table) -> dict:
    return {"kind": "table", "id": t.block.get("id"), "style": t.style, "header_style": t.header_style,
            "col_widths": t.col_widths, "row_heights": t.row_heights, "left": t.left,
            "borders": t.borders, "padding": t.padding, "space_before": t.space_before,
            "pitch": natural_pitch(t.style),
            "header_pitch": natural_pitch(t.header_style) if t.header_style else None,
            "cells": [[{"lines": c.lines, "align": c.align, "scale": c.scale, "valign": c.valign} for c in row]
                      for row in t.cells]}


def to_dict(plan: PagePlan) -> dict:
    """A page plan as plain JSON, in points."""
    bands = []
    for band in plan.bands:
        item = None
        if band.item is not None:
            item = _para_dict(band.item) if isinstance(band.item, Para) else _table_dict(band.item)
        bands.append({"space_before": band.space_before, "item": item,
                      "columns": [{"x1": c.x1, "x2": c.x2, "items": [_para_dict(it) for it in c.items]}
                                  for c in band.columns]})
    return {"index": plan.index, "width": plan.width, "height": plan.height, "margins": list(plan.margins),
            "bands": bands,
            "floats": [{"image": f.image, "x": f.x, "y": f.y, "width": f.width, "height": f.height,
                        "kind": f.kind, "block_id": f.block_id} for f in plan.floats]}


def build_document(doc) -> List[PagePlan]:
    by_page = {}
    for b in doc.blocks:
        by_page.setdefault(b["page"], []).append(b)
    return [build(p, by_page.get(p["index"], [])) for p in doc.pages]
