"""Draw a page plan as an image: what the rebuilt page should look like.

It follows the plan with the same arithmetic the Word writer asks Word to
do (exact line pitch, space before, indents, character scale, floating
pictures), so it is a fast way to judge a reconstruction against its scan
without a word processor. It is not a Word renderer: line breaking,
kerning and justification are approximations.
"""

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from . import fonts as F
from .plan import Band, Para, PagePlan, Table
from .style import measure


def _hex(c: str):
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _draw_row(canvas: Image.Image, text: str, style: dict, scale: int, x: float,
              centre_y: float, px_per_pt: float, align: str, width: Optional[float] = None):
    size_px = max(4, int(round(style["size"] * px_per_pt)))
    img = F.render(text, style["font"], bool(style.get("bold")), size_px)
    ink = measure(img, threshold=128, weight_centre=False)
    if ink is None:
        return
    crop = img[:, ink.x1:ink.x2]
    w = max(1, int(round(crop.shape[1] * scale / 100)))
    if width is not None and align == "justify":
        w = max(w, int(round(width)))
    crop = Image.fromarray(crop).resize((w, crop.shape[0]), Image.LANCZOS)
    y = int(round(centre_y - ink.centre))
    alpha = Image.fromarray(255 - np.asarray(crop))
    fill = Image.new("RGB", crop.size, _hex(style.get("color", "000000")))
    canvas.paste(fill, (int(round(x)), y), alpha)


def _draw_para(canvas, p: Para, left: float, right: float, top: float, k: float) -> float:
    """Draw a paragraph whose first slot starts at ``top`` (pt). Returns its height."""
    st = p.style
    for i, r in enumerate(p.rows):
        natural = F.text_width_pt(r.text, st["font"], st["bold"], st["size"]) * r.scale / 100
        lo = left + p.left + (p.first if i == 0 else 0)
        hi = right - p.right
        if p.align == "center":
            x = (lo + hi) / 2 - natural / 2
        elif p.align == "right":
            x = hi - natural
        else:
            x = lo
        last = i == len(p.rows) - 1
        _draw_row(canvas, r.text, st, r.scale, x * k, (top + p.pitch * (i + 0.5)) * k, k,
                  "justify" if p.align == "justify" and not last else "left",
                  width=(hi - lo) * k)
    return p.pitch * len(p.rows)


def _draw_table(canvas, t: Table, top: float, k: float) -> float:
    draw = ImageDraw.Draw(canvas)
    y = top
    for ri, row in enumerate(t.cells):
        h = t.row_heights[ri]
        style = t.header_style if (ri == 0 and t.header_style) else t.style
        x = t.left
        for ci, cell in enumerate(row):
            w = t.col_widths[ci]
            if t.borders:
                draw.rectangle([x * k, y * k, (x + w) * k, (y + h) * k], outline=(128, 128, 128))
            n = max(1, len(cell.lines))
            for li, line in enumerate(cell.lines):
                if not line.strip():
                    continue
                width = F.text_width_pt(line, style["font"], style["bold"], style["size"]) * cell.scale / 100
                lo, hi = x + t.padding, x + w - t.padding
                lx = (lo + hi - width) / 2 if cell.align == "center" else (hi - width if cell.align == "right" else lo)
                step = min(h / n, style["size"] * 1.6)
                mid = {"top": y + step * n / 2, "bottom": y + h - step * n / 2}.get(cell.valign, y + h / 2)
                cy = mid + (li - (n - 1) / 2) * step
                _draw_row(canvas, line, style, cell.scale, lx * k, cy * k, k, "left")
            x += w
        y += h
    return y - top


def render(plan: PagePlan, assets, px_per_pt: float = 1.5) -> Image.Image:
    k = px_per_pt
    canvas = Image.new("RGB", (int(plan.width * k), int(plan.height * k)), "white")
    left, right_margin, top, _ = plan.margins
    right = plan.width - right_margin
    cursor = top
    for band in plan.bands:
        start = cursor + band.space_before
        if band.item is not None:
            it = band.item
            if isinstance(it, Para):
                h = _draw_para(canvas, it, left, right, start, k)
            else:
                h = _draw_table(canvas, it, start, k)
        else:
            h = 0.0
            for col in band.columns:
                y = start
                for it in col.items:
                    y += it.space_before
                    y += _draw_para(canvas, it, col.x1, col.x2, y, k)
                h = max(h, y - start)
        cursor = start + h

    for f in plan.floats:
        img = assets.get(f.image) if assets is not None else None
        if img is None:
            continue
        img = img.convert("RGBA").resize((max(1, int(f.width * k)), max(1, int(f.height * k))))
        canvas.paste(img, (int(f.x * k), int(f.y * k)), img)
    return canvas
