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
                               # (Latin text is set in Arial/Times, whose widths differ a little)


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
class Table:
    block: dict
    rows: List[List[str]]
    col_widths: List[float]
    style: dict
    pitch: float
    top: float
    bottom: float
    left: float                # pt, page x of the table's left edge
    borders: bool = False
    space_before: float = 0.0


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


def _table(block: dict, to_pt) -> Optional[Table]:
    from ..layout.table import _columns
    st = _style(block)
    lines = [l for l in block.get("lines") or [] if l.get("bbox")]
    rows = block.get("rows") or [[""]]
    ncols = max(len(r) for r in rows)
    if block.get("bbox"):
        x1, y1, x2, y2 = (to_pt(v) for v in block["bbox"])
    elif lines:
        x1 = to_pt(min(l["bbox"][0] for l in lines)); x2 = to_pt(max(l["bbox"][2] for l in lines))
        y1 = to_pt(min(l["bbox"][1] for l in lines)); y2 = to_pt(max(l["bbox"][3] for l in lines))
    else:
        return None
    widths = [(x2 - x1) / ncols] * ncols
    if lines and ncols > 1:
        heights = sorted(l["bbox"][3] - l["bbox"][1] for l in lines)
        gap = max(2, heights[len(heights) // 2] // 3)
        cols = _columns([(tuple(l["bbox"]), l.get("text", "")) for l in lines], gap)
        if len(cols) == ncols:
            edges = [x1] + [to_pt((cols[i][1] + cols[i + 1][0]) / 2) for i in range(ncols - 1)] + [x2]
            widths = [max(12.0, edges[i + 1] - edges[i]) for i in range(ncols)]
    pitch = max(natural_pitch(st), (y2 - y1) / max(1, len(rows)))
    return Table(block=block, rows=rows, col_widths=widths, style=st,
                 pitch=pitch, top=y1, bottom=y2, left=x1, borders=bool(block.get("ruled", True)))


def _fit(para: Para, room: float) -> None:
    """Set each row's character scale so it is as wide as on the scan, and
    never wider than its room (a wrapped line would push the page down)."""
    st = para.style
    for i, r in enumerate(para.rows):
        natural = F.text_width_pt(r.text, st["font"], st["bold"], st["size"])
        if natural <= 0:
            continue
        want = 100 * (r.x2 - r.x1) / natural
        if para.align == "justify" and i < len(para.rows) - 1:
            want = min(want, 100)   # justification stretches it the rest of the way
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
    for b in blocks:
        if b["type"] in TEXT_TYPES:
            p = _para(b, to_pt, page_w)
            if p:
                items.append(p)
        elif b["type"] == "table":
            t = _table(b, to_pt)
            if t:
                items.append(t)
        elif b.get("image") and b.get("bbox"):
            x1, y1, x2, y2 = (to_pt(v) for v in b["bbox"])
            floats.append(Float(b["image"], shift(x1), y1, x2 - x1, y2 - y1, "figure"))
    for ov in page.get("overlays") or []:
        x1, y1, x2, y2 = (to_pt(v) for v in ov["bbox"])
        floats.append(Float(ov["image"], shift(x1), y1, x2 - x1, y2 - y1, ov.get("kind", "mark")))

    for it in items:
        if isinstance(it, Para):
            for r in it.rows:
                r.x1, r.x2 = shift(r.x1), shift(r.x2)
        else:
            it.left = shift(it.left)

    if not items:
        return PagePlan(page.get("index", 0), page_w, page_h, (36.0, 36.0, 36.0, 36.0), [], floats)

    # Content box: the sheet's margins are where the text is on the scan.
    left = min(_x_extent(it)[0] for it in items)
    right = max(_x_extent(it)[1] for it in items)
    left = max(6.0, left)
    right = min(page_w - 6.0, right)

    items = _split_beside(items)

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
    top_margin = max(0.0, bands[0].top)
    cursor = top_margin
    for band in bands:
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


def build_document(doc) -> List[PagePlan]:
    by_page = {}
    for b in doc.blocks:
        by_page.setdefault(b["page"], []).append(b)
    return [build(p, by_page.get(p["index"], [])) for p in doc.pages]
