"""Rebuild a page as it looked: fonts, sizes, colours, placement, graphics.

The OCR pipeline produces *what* a page says and its structure. This
package measures *how it looked* from the scan, so exporters can rebuild
the original page rather than a re-flowed version of it::

    from netra_ocr import reconstruct
    reconstruct.enrich_page(page, blocks, img_bgr, assets)   # in the pipeline
    reconstruct.enrich_document(doc)                       # an existing document

It adds, all optional in the document schema:

* ``page["dpi"]``, ``page["size_pt"]``: the physical page.
* ``page["overlays"]``: seals, signatures and other marks the models don't
  read, as transparent cut-outs ``{"image", "bbox", "kind"}``.
* ``block["style"]``: ``{"font", "size", "bold", "color"}`` per text or
  table block (``font`` is a family key from ``fonts.FAMILIES``).
* ``line["ink"]`` (tight ink box: core band and horizontal extent) and
  ``line["scale"]`` (scan width / font width at the block's size, %).
* ``block["ruled"]`` on tables: the scan shows grid lines.
* ``page["body_font"]``, ``page["body_size"]``: the page's body text face
  and size, for writers that need a default.

Logo blocks' images are replaced by transparent cut-outs of the same id.
Nothing here changes recognized text.
"""

from collections import defaultdict
from typing import List, Optional

import numpy as np
from PIL import Image

from ..imaging import bgr_to_rgb, rgb_to_bgr, to_gray
from . import fonts as F
from . import graphics, page as P, style as S

STYLED_TYPES = {"heading", "paragraph", "caption", "page_header", "page_footer", "table"}


def is_enriched(page: dict) -> bool:
    return bool(page.get("size_pt"))


def _round_size(pt: float) -> float:
    return float(np.clip(round(pt * 2) / 2, 5.0, 72.0))


BODY_TYPES = {"paragraph", "caption", "page_header", "page_footer", "table"}


def _part(block: dict, line: dict) -> str:
    """"header" for lines in a ruled table's first row, else "body"."""
    grid = block.get("grid")
    if block["type"] != "table" or not grid or len(grid["rows"]) < 3:
        return "body"
    box = line.get("bbox")
    return "header" if box and (box[1] + box[3]) / 2 < grid["rows"][1] else "body"
MAX_FIT_LINES = 5


# Formal Khmer titles and letterheads are set in Muol far more often than in
# bold body type, and on small scans the two are hard to tell apart.
MUOL_PRIOR = {"heading": 0.65}


def _khmer_share(text: str) -> float:
    letters = [ch for ch in text if not ch.isspace()]
    return sum(F._is_khmer(ch) for ch in letters) / len(letters) if letters else 0.0


def _vote(block_fits, candidates, biases, btype: str = "paragraph"):
    """Pick one (family, bold) for a block: the lowest total score, weighting
    each line by its width (long lines carry more evidence)."""
    totals = defaultdict(float)
    counts = defaultdict(int)
    for ink, fits in block_fits:
        for key in candidates:
            if key in fits:
                weight = MUOL_PRIOR.get(btype, 1.0) if key[0] == "muol" else 1.0
                totals[key] += F.score(fits[key], *biases[key[0]]) * ink.width * weight
                counts[key] += 1
    n = max(counts.values()) if counts else 0
    totals = {k: v for k, v in totals.items() if counts[k] == n}
    return min(totals, key=totals.get) if totals else None


def _block_size_px(items, key) -> float:
    """A block's type size (em, scan px).

    Width is measured over hundreds of pixels, the core band's height over
    a handful, so the size that makes the text as wide as the scan is far
    more precise. Justified lines are stretched, though: in a paragraph of
    several lines the narrower (less stretched) estimates are the honest
    ones. The height-based estimate guards against a wrong font or text.
    """
    by_height = [fits[key].size_px for *_, fits in items if key in fits]
    by_width = [fits[key].size_px * fits[key].width_ratio for _, line, _, fits in items
                if key in fits and len(line.get("text", "").strip()) >= 6]
    if not by_height:
        # No shaping engine (or no text): a Khmer line's core band is
        # about 0.45 em.
        return float(np.median([ink.core for _, _, ink, _ in items])) / 0.45
    h = float(np.median(by_height))
    if not by_width:
        return h
    wsize = float(np.percentile(by_width, 25) if len(by_width) >= 3 else np.median(by_width))
    return wsize if 0.55 * h <= wsize <= 1.5 * h else h


def enrich_page(page: dict, blocks: List[dict], img_bgr: np.ndarray, assets,
                dpi: Optional[float] = None, graphics_layer: bool = True) -> None:
    """Measure one page's look in place. ``blocks`` are that page's blocks."""
    h, w = img_bgr.shape[:2]
    rgb = bgr_to_rgb(img_bgr)
    gray = to_gray(img_bgr)
    page_dpi, size_pt, _ = P.page_geometry(w, h, dpi or page.get("dpi"))
    page["dpi"] = round(page_dpi, 3)
    page["size_pt"] = [round(v, 2) for v in size_pt]
    thr = P.ink_threshold(gray)

    # 1. Measure every line, and fit every candidate font to the longest
    # few lines of each block (more lines add time, not evidence).
    measured = []   # (block, line, ink, fits)
    for b in blocks:
        if b["type"] not in STYLED_TYPES:
            continue
        inks = []
        for line in b.get("lines") or []:
            if not line.get("bbox"):
                continue
            ink = S.measure_line(rgb, gray, line["bbox"], thr, latin=S.is_latin(line.get("text", "")))
            if ink is None:
                continue
            line["ink"] = [ink.x1, ink.top, ink.x2, ink.bottom]
            inks.append((line, ink))
        longest = {id(l) for l, _ in sorted(inks, key=lambda li: -li[1].width)[:MAX_FIT_LINES]}
        for line, ink in inks:
            fit = F.HAS_SHAPING and id(line) in longest
            fits = F.fit_all(ink, line.get("text", "")) if fit else {}
            measured.append((b, line, ink, fits))

    # Scans thicken strokes (blur, ink spread, JPEG). Most body text is
    # regular weight, so how much heavier a body line looks than the
    # lighter of the regular body faces is the page's blur, not boldness.
    # One estimate for the whole page: estimating it per face would cancel
    # out the very differences in weight that tell the faces apart.
    body_items = [m for m in measured if m[0]["type"] in BODY_TYPES] or measured

    def blur(faces):
        pair = []
        for attr in ("stroke_ratio", "density_ratio"):
            vals = []
            for *_, f in body_items:
                per_face = [getattr(f[(fam, False)], attr) for fam in faces if (fam, False) in f]
                if per_face:
                    vals.append(min(per_face))
            pair.append(float(np.clip(np.median(vals), 1.0, 1.6)) if vals else 1.0)
        return {fam: tuple(pair) for fam in F.FAMILIES}

    # Provisional blur, against the lighter of the two body faces, to pick
    # the body face; then the blur against that face, to judge weights.
    biases = blur(("sans", "serif"))

    # A document has one body face; vote for it over the body text lines.
    body_votes = {}
    for fam in ("sans", "serif"):
        total = 0.0
        for _, _, ink, fits in body_items:
            options = [F.score(fits[(fam, bold)], *biases[fam])
                       for bold in (False, True) if (fam, bold) in fits]
            total += min(options) * ink.width if options else 0.0
        body_votes[fam] = total
    body = min(body_votes, key=body_votes.get) if any(body_votes.values()) else F.DEFAULT_FAMILY
    page["body_font"] = body
    biases = blur((body,))

    # 2. One font, weight, size and colour per block -- and per header row
    # of a ruled table, which is often set in another face (Muol, bold).
    by_block = defaultdict(list)
    for item in measured:
        by_block[(id(item[0]), _part(item[0], item[1]))].append(item)
    colours, styled = [], []
    for (_, part), items in by_block.items():
        b = items[0][0]
        candidates = [(body, False), (body, True)]
        # Muol is a Khmer face; its Latin letters are a display serif.
        if _khmer_share(" ".join(line.get("text", "") for _, line, _, _ in items)) >= 0.5:
            candidates.append(("muol", False))
        key = _vote([(ink, fits) for _, _, ink, fits in items], candidates, biases,
                    b["type"]) or (body, False)
        size = _round_size(_block_size_px(items, key) * 72 / page_dpi)
        colours.append(S.text_colour([ink.colour for *_, ink, _ in items]))
        styled.append((b, items, {"font": key[0], "size": size, "bold": bool(key[1])}, part))

    # Body text is one size on a real page; small measured differences are
    # noise. Snap body blocks near the page's body size onto it.
    weights = [(st["size"], len(items)) for b, items, st, _ in styled if b["type"] in BODY_TYPES]
    if weights:
        sizes = np.repeat([s for s, _ in weights], [n for _, n in weights])
        body_size = float(np.median(sizes))
        for b, _, st, _ in styled:
            if b["type"] in BODY_TYPES and abs(st["size"] / body_size - 1) <= 0.15:
                st["size"] = _round_size(body_size)
        page["body_size"] = _round_size(body_size)

    for b, items, st, _ in styled:
        for _, line, ink, _ in items:
            text = line.get("text", "")
            natural = F.text_width_pt(text, st["font"], st["bold"], st["size"]) if text.strip() else 0
            if natural > 0:
                line["scale"] = int(round(100 * (ink.width * 72 / page_dpi) / natural))
    for (b, _, style, part), colour in zip(styled, S.snap_colours(colours)):
        style["color"] = colour
        b["header_style" if part == "header" else "style"] = style

    # 3. Graphics: marks the models don't read, and transparent logos.
    if not graphics_layer:
        return
    bg = P.local_background(rgb)
    text_lines = [(line["bbox"], ink) for _, line, ink, _ in measured]
    # Lines that couldn't be measured are still text: erase them as black.
    seen = {id(line) for _, line, _, _ in measured}
    for b in blocks:
        for line in b.get("lines") or []:
            if id(line) not in seen and line.get("bbox") and line.get("text", "").strip():
                text_lines.append((line["bbox"], None))
    figures = [(i, b["bbox"], b.get("label") == "logo") for i, b in enumerate(blocks)
               if b["type"] in ("figure", "formula") and b.get("bbox")]
    # Ruled tables' grid lines are drawn by the writer as table borders.
    grids = [b["grid"] for b in blocks if b["type"] == "table" and b.get("grid")]
    overlays, cutouts = graphics.extract(rgb, bg, text_lines, figures, grids)
    for i, img in cutouts.items():
        assets.put(blocks[i]["image"], img)
    for b in blocks:
        if b["type"] == "table":
            b["ruled"] = bool(b.get("grid"))
    page_index = page.get("index", 0)
    page["overlays"] = []
    for k, ov in enumerate(overlays):
        asset = assets.put(f"p{page_index}-gfx{k}", ov.image)
        page["overlays"].append({"image": asset, "bbox": list(ov.bbox), "kind": ov.kind})


def enrich_document(doc, force: bool = False) -> None:
    """Enrich every page of a ``Document`` from its stored page images.

    Used for documents produced without ``reconstruct=True``. Page renders
    are stored as JPEG, so measurements are a little less exact than in
    the pipeline, which sees the original pixels.
    """
    by_page = defaultdict(list)
    for b in doc.blocks:
        by_page[b["page"]].append(b)
    for page in doc.pages:
        if is_enriched(page) and not force:
            continue
        img = doc.assets.get(page.get("image")) if page.get("image") else None
        if img is None:
            w, h = page.get("width") or 1, page.get("height") or 1
            dpi, size_pt, _ = P.page_geometry(w, h, page.get("dpi"))
            page["dpi"], page["size_pt"] = round(dpi, 3), size_pt
            continue
        arr = rgb_to_bgr(np.asarray(img.convert("RGB")))
        enrich_page(page, by_page[page["index"]], arr, doc.assets)


__all__ = ["enrich_page", "enrich_document", "is_enriched"]
