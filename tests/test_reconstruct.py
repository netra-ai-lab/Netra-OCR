"""Rebuilding the original page: measurement, graphics, layout plan, Word output.

A synthetic A4 letter at 150 dpi is drawn with known fonts, sizes and
colours (a blue Muol title, a side-by-side letterhead, black body text,
and a red seal printed across a text line), then measured back.
"""

import io
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from netra_ocr.document import AssetStore, Document
from netra_ocr.exporters import export
from netra_ocr.imaging import rgb_to_bgr
from netra_ocr.reconstruct import enrich_page, fonts as F, page as P, plan as PL

pytestmark = pytest.mark.skipif(not F.HAS_SHAPING, reason="Pillow without libraqm can't shape Khmer")

DPI = 150
W, H = 1240, 1754                        # A4 at 150 dpi
BLUE, RED = (20, 50, 170), (210, 30, 30)
BODY = [
    "ក្រសួងសុខាភិបាលនៃព្រះរាជាណាចក្រកម្ពុជា សូមជម្រាបជូនសាធារណជន",
    "ទាំងអស់ឱ្យបានជ្រាប និងថា យោងតាមរបាយការណ៍ពីមន្ទីរពេទ្យ",
    "មិត្តភាពខ្មែរ-សូវៀត បានឱ្យដឹងថា អ្នកជំងឺបានជាសះស្បើយ",
    "ហើយបានចេញពីមន្ទីរពេទ្យនៅថ្ងៃទី២២ ខែមីនា ឆ្នាំ២០២១",
]


def _px(pt):
    return int(round(pt * DPI / 72))


def _draw(draw, xy, text, family, bold, size_pt, fill):
    """Draw text with its left edge at x and its baseline at y; returns the
    padded line box a detector would give."""
    khmer, latin = F.run_fonts(family, bold)
    x, y = xy
    x0 = x
    for run, is_khmer in F.script_runs(text):
        font = F._font(khmer if is_khmer else latin, _px(size_pt))
        draw.text((x, y), run, font=font, fill=fill, anchor="ls")
        x += font.getlength(run)
    em = _px(size_pt)
    return [int(x0) - 6, int(y - 1.25 * em), int(x) + 6, int(y + 0.55 * em)]


def make_letter():
    img = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(img)
    lines = {}
    # Letterhead: ministry (left) opposite the kingdom motto (right).
    lines["left"] = [("ក្រសួងសុខាភិបាល", _draw(d, (120, 200), "ក្រសួងសុខាភិបាល", "muol", False, 13, BLUE))]
    lines["right"] = [("ព្រះរាជាណាចក្រកម្ពុជា", _draw(d, (760, 200), "ព្រះរាជាណាចក្រកម្ពុជា", "muol", False, 13, BLUE))]
    # Title.
    lines["title"] = [("សេចក្តីប្រកាសព័ត៌មាន", _draw(d, (430, 420), "សេចក្តីប្រកាសព័ត៌មាន", "muol", False, 18, BLUE))]
    # Body, 12 pt, pitch 26 pt.
    lines["body"] = []
    for i, text in enumerate(BODY):
        y = 560 + i * _px(26)
        lines["body"].append((text, _draw(d, (150, y), text, "sans", False, 12, (20, 20, 20))))
    # A red seal (thick ring) printed over the last body line.
    cx, cy, r = 760, 650, 110
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=RED, width=10)
    d.ellipse([cx - 45, cy - 45, cx + 45, cy + 45], fill=RED)
    return img, lines


def make_doc():
    img, lines = make_letter()
    assets = AssetStore()
    assets.put("page-0", img)

    def block(bid, btype, items, align="left", **extra):
        boxes = [b for _, b in items]
        bbox = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes)]
        return {"id": bid, "type": btype, "page": 0, "bbox": bbox, "align": align,
                "text": "\n".join(t for t, _ in items), **extra,
                "lines": [{"text": t, "bbox": b} for t, b in items]}

    blocks = [
        block("left", "paragraph", lines["left"]),
        block("right", "paragraph", lines["right"], align="right"),
        block("title", "heading", lines["title"], align="center", level=1),
        block("body", "paragraph", lines["body"]),
    ]
    page = {"index": 0, "width": W, "height": H, "image": "page-0"}
    return Document(pages=[page], blocks=blocks, assets=assets), img


@pytest.fixture(scope="module")
def enriched():
    doc, img = make_doc()
    enrich_page(doc.pages[0], doc.blocks, rgb_to_bgr(np.asarray(img)), doc.assets)
    return doc


def by_id(doc, bid):
    return next(b for b in doc.blocks if b["id"] == bid)


def test_page_geometry():
    dpi, size, name = P.page_geometry(W, H)
    assert name == "A4" and abs(dpi - DPI) < 1
    assert size == [595.28, 841.89]
    # A photo of a letter isn't cropped exactly to the sheet; it still fits.
    dpi, size, name = P.page_geometry(526, 770)
    assert name == "A4" and 770 * 72 / dpi <= size[1] + 0.01
    # A known render resolution (PDF) gives the exact size.
    assert P.page_geometry(1654, 2339, dpi=200)[1] == [595.44, 842.04]


def test_measures_font_size_and_colour(enriched):
    doc = enriched
    page = doc.pages[0]
    assert abs(page["dpi"] - DPI) < 1 and page["size_pt"] == [595.28, 841.89]

    body = by_id(doc, "body")["style"]
    assert body["font"] in ("sans", "serif") and not body["bold"]
    assert abs(body["size"] - 12) <= 1
    assert body["color"] == "000000"

    title = by_id(doc, "title")["style"]
    assert title["font"] == "muol"
    assert abs(title["size"] - 18) <= 1.5
    r, g, b = (int(title["color"][i:i + 2], 16) for i in (0, 2, 4))
    assert b > r + 60 and b > g + 40                 # blue

    for line in by_id(doc, "body")["lines"]:
        assert 85 <= line["scale"] <= 115            # width matches the font at that size
        assert line["ink"][1] > line["bbox"][1]      # tight ink box inside the padded one


def test_seal_becomes_transparent_overlay(enriched):
    doc = enriched
    overlays = doc.pages[0]["overlays"]
    stamps = [o for o in overlays if o["kind"] == "stamp"]
    assert len(stamps) == 1
    x1, y1, x2, y2 = stamps[0]["bbox"]
    assert x1 <= 650 + 5 and x2 >= 870 - 5 and y1 <= 540 + 5 and y2 >= 760 - 5
    img = np.asarray(doc.assets.get(stamps[0]["image"]))
    assert img.shape[2] == 4
    alpha = img[..., 3]
    # Opaque on the red ring, including where it crosses the last body line,
    # transparent in the paper between ring and centre.
    ring_on_text = img[650 - y1 - 3:650 - y1 + 3, 870 - 5 - x1 - 3:870 - 5 - x1][..., 3]
    assert ring_on_text.max() > 200
    assert alpha[650 - y1, 760 + 75 - x1] < 30
    red = img[alpha > 200][:, :3].astype(int)
    assert np.median(red[:, 0] - red[:, 1]) > 100
    # No overlay is just leftover body text.
    assert all(o["kind"] != "mark" or (o["bbox"][2] - o["bbox"][0]) < 200 for o in overlays)


def test_plan_puts_letterhead_side_by_side(enriched):
    plan = PL.build(enriched.pages[0], enriched.blocks)
    side = [b for b in plan.bands if b.columns]
    assert len(side) == 1 and len(side[0].columns) == 2
    flow = [b.item for b in plan.bands if b.item is not None]
    body = next(p for p in flow if p.block["id"] == "body")
    assert len(body.rows) == 4 and abs(body.pitch - 26) < 1.5
    assert body.keep_lines
    assert any(f.kind == "stamp" for f in plan.floats)


def test_edited_block_reflows(enriched):
    doc = enriched
    b = by_id(doc, "body")
    saved = b["text"]
    b["text"] = "edited text"
    try:
        plan = PL.build(doc.pages[0], doc.blocks)
        body = next(x.item for x in plan.bands if x.item is not None and x.item.block is b)
        assert not body.keep_lines and body.text == "edited text"
    finally:
        b["text"] = saved


def test_docx_rebuilds_the_page(enriched):
    from docx import Document as Word

    data = export(enriched, "docx")[0]
    word = Word(io.BytesIO(data))
    sec = word.sections[0]
    assert abs(sec.page_width.pt - 595.3) < 1 and abs(sec.page_height.pt - 841.9) < 1

    xml = word.element.xml
    assert xml.count("<wp:anchor") >= 1                         # the seal floats on the page
    # Standard Khmer fonts by default, in every slot (Word draws Khmer with
    # the ascii/hAnsi font): Khmer OS Muol Light for the Muol title, Khmer OS
    # Siemreap for the body.
    for name in ("Khmer OS Muol Light", "Khmer OS Siemreap"):
        assert f'w:ascii="{name}" w:hAnsi="{name}" w:eastAsia="{name}" w:cs="{name}"' in xml
    assert "Noto" not in xml and 'w:cs="Moul"' not in xml
    assert 'w:lineRule="exact"' in xml
    texts = [p.text for p in word.paragraphs]
    assert "\n".join(BODY) in texts                             # source line breaks kept
    assert len(word.tables) == 1                                # the side-by-side letterhead
    assert [c.text for c in word.tables[0].rows[0].cells] == ["ក្រសួងសុខាភិបាល", "ព្រះរាជាណាចក្រកម្ពុជា"]

    # Nothing is embedded: the fonts are on every Khmer-language PC.
    z = zipfile.ZipFile(io.BytesIO(data))
    assert not any(n.startswith("word/fonts/") for n in z.namelist())


def test_embed_fonts_mode(enriched):
    """fonts="embed": the measuring fonts are embedded, obfuscated per
    ECMA-376 17.8.1, and recoverable."""
    from docx.oxml.ns import qn
    from lxml import etree

    data = export(enriched, "docx", fonts="embed")[0]
    z = zipfile.ZipFile(io.BytesIO(data))
    assert b'w:cs="Moul"' in z.read("word/document.xml")
    assert b"<w:embedTrueTypeFonts" in z.read("word/settings.xml")
    table = etree.fromstring(z.read("word/fontTable.xml"))
    rels = etree.fromstring(z.read("word/_rels/fontTable.xml.rels"))
    targets = {r.get("Id"): r.get("Target") for r in rels}
    embedded = {}
    for font in table.findall(qn("w:font")):
        for emb in font:
            if emb.tag in (qn("w:embedRegular"), qn("w:embedBold")):
                embedded[(font.get(qn("w:name")), emb.tag)] = (emb.get(qn("w:fontKey")),
                                                              targets[emb.get(qn("r:id"))])
    assert ("Moul", qn("w:embedRegular")) in embedded
    key_guid, target = embedded[("Moul", qn("w:embedRegular"))]
    blob = bytearray(z.read("word/" + target))
    key = bytes.fromhex(key_guid.strip("{}").replace("-", ""))[::-1]
    for i in range(32):
        blob[i] ^= key[i % 16]
    with open(F.FAMILIES["muol"].file(False), "rb") as fh:
        assert bytes(blob) == fh.read()


def _run_fonts(enriched, text, **kw):
    """{run text: set of font names in its four rFonts slots} for ``text``
    exported as the body block."""
    from docx import Document as Word
    from docx.oxml.ns import qn

    b = by_id(enriched, "body")
    saved = b["text"]
    b["text"] = text
    try:
        word = Word(io.BytesIO(export(enriched, "docx", **kw)[0]))
    finally:
        b["text"] = saved
    runs = {}
    for r in word.element.body.iter(qn("w:r")):
        t, fonts = r.find(qn("w:t")), r.find(qn("w:rPr") + "/" + qn("w:rFonts"))
        if t is not None and fonts is not None and t.text.strip():
            runs[t.text.strip()] = {fonts.get(qn(a)) for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs")}
    return runs


def test_runs_name_one_font_in_every_slot(enriched):
    """Word draws Khmer with the ascii/hAnsi font, not the complex-script
    one: a Khmer run with Arial there comes out in DaunPenh, far smaller."""
    runs = _run_fonts(enriched, "ក្រសួង ABC 123")
    assert runs["ក្រសួង"] == {"Khmer OS Siemreap"}
    assert runs["ABC 123"] == {"Khmer OS Siemreap"}
    runs = _run_fonts(enriched, "ក្រសួង ABC 123", fonts="embed")
    assert runs["ក្រសួង"] == {F.FAMILIES[by_id(enriched, "body")["style"]["font"]].name}
    assert runs["ABC 123"] == {"Arial"}


def test_standard_khmer_fonts():
    """Titles set in Muol become Khmer OS Muol Light; everything else Khmer OS Siemreap."""
    assert F.FAMILIES["muol"].system_name == "Khmer OS Muol Light"
    assert F.FAMILIES["sans"].system_name == F.FAMILIES["serif"].system_name == "Khmer OS Siemreap"


def _body_para(enriched, **kw):
    from docx import Document as Word
    word = Word(io.BytesIO(export(enriched, "docx", **kw)[0]))
    return word, next(p for p in word.paragraphs if p.text.startswith(BODY[0]))


def test_sizes_are_converted_to_the_khmer_os_fonts(enriched):
    """The measured size is Noto's size for the scan's line width; Word gets
    Khmer OS Siemreap's size for that same width (Siemreap is ~8% wider per
    point), with the character scale left as measured -- natural letters,
    as tall as on the scan when it was typed in Siemreap. ``fonts="embed"``
    keeps the measured size."""
    from docx.oxml.ns import qn

    def body_run(**kw):
        rpr = _body_para(enriched, **kw)[1].runs[0]._r.rPr
        w = rpr.find(qn("w:w"))
        return int(rpr.find(qn("w:sz")).get(qn("w:val"))) / 2, int(w.get(qn("w:val"))) if w is not None else 100

    body = by_id(enriched, "body")["style"]
    k, s = F.FAMILIES[body["font"]].system_size_and_width(body["bold"])
    size, scale = body_run()
    assert abs(size - body["size"] * k * s) <= 0.25 and size < body["size"]
    size_e, scale_e = body_run(fonts="embed")
    assert size_e == body["size"]
    # Same line width; the scale only absorbs rounding the size to half points.
    assert abs(size * scale - size_e * scale_e * k * s) <= 0.02 * size_e * scale_e
    assert abs(scale - scale_e) <= 5


def test_lines_are_raised_to_where_word_draws_them(enriched):
    """Word puts the baseline of an exact-height line at 80% of the line, the
    plan puts the core band in the middle: each paragraph is raised by the
    difference, the first one through the page's top margin."""
    from netra_ocr.exporters.docx_layout import SPACER_PT, WORD_BASELINE, _Writer

    plan = PL.build_document(enriched)[0]
    band = next(b for b in plan.bands if b.item is not None and b.item.block["id"] == "body")
    body = band.item
    writer = _Writer(enriched)
    size = writer._word_size(body.style)
    drop = writer._drop(body.style, body.pitch)
    fam = F.FAMILIES[body.style["font"]]
    assert drop == pytest.approx((WORD_BASELINE - 0.5) * body.pitch - fam.system_core_mid * size)
    assert 2.0 < drop < 6.0                               # 12 pt text on a 26 pt pitch: ~4 pt

    # The page starts higher than the plan's first line by the first band's
    # rise (the letterhead row here), never by more.
    word, para = _body_para(enriched)
    margin = word.sections[0].top_margin.pt
    assert plan.margins[2] - SPACER_PT - 8.0 < margin < plan.margins[2] - SPACER_PT
    # The body's own gap above it shrinks by its drop (less what was already
    # gained above it), so its lines land on the scan's.
    assert para.paragraph_format.space_before.pt < band.space_before


def _para(rows, align="justify"):
    rows = [PL.Row(text=f"row {i}", x1=x1, x2=x2, centre=100 + 20 * i) for i, (x1, x2) in enumerate(rows)]
    return PL.Para(block={"id": "p"}, rows=rows, style={"font": "sans", "bold": False, "size": 12.0},
                   align=align, pitch=20, top=90, bottom=90 + 20 * len(rows))


def test_rows_that_stop_lining_up_get_a_paragraph_of_their_own():
    """Word would move a centred row to the paragraph's left edge, and stretch
    a justified row that stops short to the right edge, with wide gaps."""
    full = (72, 520)
    whole = _para([full, full, full, (72, 300)])
    assert PL._split_irregular(whole, 9) == [whole]

    short = PL._split_irregular(_para([full, (72, 300), full, (72, 200)]), 9)
    assert [len(p.rows) for p in short] == [2, 2]          # the short row ends its paragraph

    centred = PL._split_irregular(_para([full, full, (200, 390), full, full]), 9)
    assert [len(p.rows) for p in centred] == [2, 1, 2]
    assert centred[1].rows[0].x1 == 200


def test_size_is_capped_by_the_line_boxes():
    """A block measured far larger than its line boxes allow (heavy bold print,
    a word too short to check against its width) gets the size they imply."""
    from netra_ocr.reconstruct import _cap_sizes

    def block(btype, size, height, text="ក" * 20, n=3):
        lines = [{"bbox": [0, 0, 100, height], "text": text} for _ in range(n)]
        return ({"type": btype}, [(None, ln, None, None) for ln in lines], {"size": size}, None)

    dpi = 144                                             # 2 px per pt
    body = [block("paragraph", 12.0, 36) for _ in range(3)]  # boxes 1.5 em
    big = block("paragraph", 30.0, 36)                    # boxes say 12 pt
    tiny = block("heading", 6.0, 54, text="និង", n=1)       # a short word: 18 pt
    fine = block("heading", 16.0, 48)                     # within reach: kept
    _cap_sizes(body + [big, tiny, fine], dpi)
    assert [b[2]["size"] for b in body] == [12.0] * 3
    assert big[2]["size"] == 12.0
    assert tiny[2]["size"] == 18.0
    assert fine[2]["size"] == 16.0


def test_rows_are_never_widened_to_match_the_khmer_os_font(enriched, monkeypatch):
    """Rows are narrowed when Siemreap sets them wider than planned (Word would
    wrap them) and left alone when it sets them narrower."""
    from netra_ocr.exporters.docx_layout import _Writer

    style = {"font": "sans", "bold": False, "size": 12.0}
    text = "កម្ពុជា"
    planned = F.text_width_pt(text, "sans", False, 12.0)
    writer = _Writer(enriched)
    # At the converted size a calibrated Siemreap is as wide as planned.
    monkeypatch.setattr(F, "system_width_pt", lambda *a: planned * 1.04)
    assert writer._width_fix(text, style) == pytest.approx(1 / 1.04)
    monkeypatch.setattr(F, "system_width_pt", lambda *a: planned * 0.96)
    assert writer._width_fix(text, style) == 1.0
    monkeypatch.setattr(F, "system_width_pt", lambda *a: None)   # font not installed
    assert writer._width_fix(text, style) == 1.0
    assert _Writer(enriched, fonts="embed")._width_fix(text, style) == 1.0


def test_appearance_survives_validation(enriched, tmp_path):
    enriched.save(str(tmp_path))
    again = Document.load(str(tmp_path))
    page = again.pages[0]
    assert page["overlays"] and page["size_pt"] and page["dpi"]
    assert again.assets.get(page["overlays"][0]["image"]).mode == "RGBA"
    body = next(b for b in again.blocks if b["id"] == "body")
    assert body["style"] == by_id(enriched, "body")["style"]
    assert body["lines"][0]["ink"] and body["lines"][0]["scale"]

    bad = again.to_dict()
    bad["blocks"][0]["style"] = {"font": "comic sans"}
    with pytest.raises(Exception):
        Document.from_dict(bad, again.assets)


def test_export_without_page_images():
    """Old documents without a stored page render still export."""
    doc = Document.from_dict({
        "pages": [{"width": 600, "height": 800}],
        "blocks": [{"id": "a", "type": "paragraph", "text": "សួស្តី\nHello", "page": 0,
                    "bbox": [50, 50, 300, 90]}],
    })
    from docx import Document as Word
    word = Word(io.BytesIO(export(doc, "docx")[0]))
    assert any("សួស្តី" in p.text for p in word.paragraphs)


# ── ruled tables ─────────────────────────────────────────────────────────

def make_ruled_table():
    """A faint-grey ruled table (4 columns x 5 rows) with barcodes in the
    second column (sharing a prefix, so their bars line up) and dates."""
    rng = np.random.default_rng(0)
    img = Image.new("L", (1000, 420), 252)
    d = ImageDraw.Draw(img)
    xs, ys = [40, 140, 460, 700, 960], [30, 100, 170, 240, 310, 380]
    for y in ys:
        d.line([(xs[0], y), (xs[-1], y)], fill=238, width=2)    # 14 grey levels: barely visible
    for x in xs:
        d.line([(x, ys[0]), (x, ys[-1])], fill=238, width=2)
    prefix = rng.integers(0, 2, 40)
    lines = []
    font = F._font(F.FAMILIES["sans"].file(False), 24)
    for r in range(1, 5):
        y = ys[r]
        bits = np.concatenate([prefix, rng.integers(0, 2, 30)])
        for i, bit in enumerate(bits):                           # the barcode
            if bit:
                d.rectangle([180 + 3 * i, y + 8, 180 + 3 * i + 1, y + 44], fill=10)
        lines.append(((176, y + 6, 396, y + 46), "II Nilliania"))   # what OCR makes of it
        text = f"0{r}-2022"
        d.text((520, y + 46), text, font=font, fill=20, anchor="ls")
        lines.append(((514, y + 22, 520 + int(font.getlength(text)) + 6, y + 54), text))
        d.text((60, y + 46), str(r), font=font, fill=20, anchor="ls")
        lines.append(((54, y + 22, 80, y + 54), str(r)))
    d.text((520, 72), "Date", font=font, fill=20, anchor="ls")
    lines.append(((514, 48, 580, 80), "Date"))
    return np.asarray(img), lines, xs, ys


def test_ruled_grid_reads_faint_lines_and_ignores_barcodes():
    from netra_ocr.layout.table import cells_from_grid, ruled_grid
    from netra_ocr.structure import is_barcode

    gray, lines, xs, ys = make_ruled_table()
    barcodes = [b for b, t in lines if is_barcode(gray, b)]
    assert len(barcodes) == 4 and all(t == "II Nilliania" for b, t in lines if b in barcodes)
    text = [(b, t) for b, t in lines if b not in barcodes]

    found = ruled_grid(gray, (35, 25, 965, 385), text)
    assert found is not None
    fy, fx = found
    assert len(fy) == len(ys) and all(abs(a - b) <= 3 for a, b in zip(fy, ys))
    assert len(fx) == len(xs) and all(abs(a - b) <= 3 for a, b in zip(fx, xs))
    cells = cells_from_grid(text, fy, fx)
    assert cells[0][2] == "Date" and cells[2][2] == "02-2022" and cells[4][0] == "4"
    assert cells[1][1] == ""                                   # barcode cell: no noise text


def test_ruled_table_keeps_its_grid_in_word():
    from docx import Document as Word
    from netra_ocr.layout.table import cells_from_grid, ruled_grid
    from netra_ocr.structure import is_barcode

    gray, lines, xs, ys = make_ruled_table()
    text = [(b, t) for b, t in lines if not is_barcode(gray, b)]
    fy, fx = ruled_grid(gray, (35, 25, 965, 385), text)
    assets = AssetStore()
    assets.put("page-0", Image.fromarray(gray).convert("RGB"))
    block = {"id": "t", "type": "table", "page": 0, "bbox": [35, 25, 965, 385],
             "rows": cells_from_grid(text, fy, fx), "grid": {"rows": fy, "cols": fx}, "ruled": True,
             "lines": [{"text": t, "bbox": list(b)} for b, t in text]}
    doc = Document.from_dict({"pages": [{"width": 1000, "height": 420, "image": "page-0"}],
                              "blocks": [block]}, assets)
    from netra_ocr.reconstruct import enrich_document
    enrich_document(doc)
    plan = PL.build_document(doc)[0]
    table = next(b.item for b in plan.bands if isinstance(b.item, PL.Table))
    dpi = doc.pages[0]["dpi"]
    assert [round(h, 1) for h in table.row_heights] == [round((b - a) * 72 / dpi, 1) for a, b in zip(fy, fy[1:])]
    assert table.cells[2][2].align == "center" and table.cells[1][0].align == "left"

    word = Word(io.BytesIO(export(doc, "docx")[0]))
    assert len(word.tables) == 1
    t = word.tables[0]
    assert len(t.rows) == 5 and len(t.columns) == 4
    assert t.cell(2, 2).text == "02-2022"
    xml = t._tbl.xml
    assert 'w:hRule="exact"' in xml and 'w:val="single"' in xml   # exact rows, drawn borders


def make_crossed_table():
    """A dark ruled table (3 columns x 3 rows) whose rows the line detector
    read as single lines straight across the column rules, the way it does
    on small scans ("1 | Policy specialist | 1")."""
    img = Image.new("RGB", (800, 220), "white")
    d = ImageDraw.Draw(img)
    xs, ys = [40, 120, 600, 760], [30, 80, 130, 180]
    for y in ys:
        d.line([(xs[0], y), (xs[-1], y)], fill=(40, 40, 40), width=2)
    for x in xs:
        d.line([(x, ys[0]), (x, ys[-1])], fill=(40, 40, 40), width=2)
    font = F._font(F.FAMILIES["sans"].file(False), 22)
    boxes = []
    for r in range(3):
        base = ys[r] + 36
        for x, text in ((70, str(r + 1)), (130, "Digital Policy Specialist"), (660, "1")):
            d.text((x, base), text, font=font, fill="black", anchor="ls")
        # Starts a little above the row's top rule, crosses both inner rules.
        boxes.append((60, ys[r] - 2, 690, ys[r + 1] - 4))
    return img, boxes, xs, ys


def test_lines_across_column_rules_are_cut_into_cells():
    from netra_ocr.layout.table import drawn_rules, split_at_rules
    from netra_ocr.structure import Unit, ink_mask, split_table_lines

    img, boxes, xs, ys = make_crossed_table()
    bgr = rgb_to_bgr(np.asarray(img))
    gray = np.asarray(img.convert("L"))
    fy, strong, _ = drawn_rules(gray, (40, 30, 760, 180))
    assert len(fy) == 4 and all(abs(a - b) <= 2 for a, b in zip(fy, ys))
    assert len(strong) == 4 and all(abs(a - b) <= 2 for a, b in zip(strong, xs))

    pieces = split_at_rules(boxes, ink_mask(bgr), fy, strong)
    for r, row in enumerate(pieces):
        assert len(row) == 3
        for (x1, y1, x2, y2), (a, b) in zip(row, zip(xs, xs[1:])):
            assert a < x1 < x2 < b                      # inside its column, clear of the rules
            assert ys[r] < y1 < y2 <= ys[r + 1]         # and of the row's rules

    crops = [img.crop(b) for b in boxes]
    units = [Unit("plain text", (0, 0, 30, 10), []), Unit("table", (40, 30, 760, 180), [0, 1, 2])]
    new_boxes, new_crops, new_units = split_table_lines(bgr, units, boxes, crops)
    assert len(new_boxes) == len(new_crops) == 9
    assert new_units[1].lines == list(range(9)) and new_units[0].lines == []


def test_cell_alignment_compares_the_gaps():
    # Long text from the left edge of a wide cell: its centre is near the
    # cell's, but it is left aligned.
    assert PL._cell_align((110, 0, 398, 0), 104, 443) == "left"
    assert PL._cell_align((80, 0, 90, 0), 64, 104) == "center"
    assert PL._cell_align((380, 0, 438, 0), 104, 443) == "right"


def test_page_frame_does_not_make_a_page_sized_picture():
    from netra_ocr.reconstruct import graphics

    img = Image.new("RGB", (800, 1000), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 740, 940], outline=(30, 30, 30), width=2)          # the page frame
    d.ellipse([480, 700, 640, 860], outline=(200, 30, 40), width=8)         # a seal
    d.line([(150, 300), (190, 330), (230, 290)], fill=(20, 40, 160), width=4)   # a tick
    rgb = np.asarray(img)
    overlays, _ = graphics.extract(rgb, P.local_background(rgb), [], [])
    page = 800 * 1000
    assert overlays and all((o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]) < 0.1 * page
                            for o in overlays)
    assert any(o.kind == "stamp" for o in overlays)


def test_figure_around_text_keeps_only_its_picture():
    from netra_ocr.layout.base import LayoutRegion
    from netra_ocr.structure import plan_page

    ink = np.zeros((600, 500), bool)
    lines = [(20, 100 + 30 * i, 300, 120 + 30 * i) for i in range(5)]
    for x1, y1, x2, y2 in lines:
        ink[y1 + 5:y2 - 5, x1:x2:4] = True                  # text strokes
    ink[110:220, 340:460] = True                            # a seal beside the text
    units = plan_page((500, 600), lines, [], [LayoutRegion("figure", (10, 90, 480, 250))], ink=ink)
    figures = [u for u in units if u.label == "figure"]
    assert len(figures) == 1 and figures[0].bbox[0] >= 320       # the seal alone
    assert sorted(i for u in units if u.label != "figure" for i in u.lines) == list(range(5))


def test_list_dash_left_of_a_line_is_kept():
    from netra_ocr.reconstruct import graphics

    img = Image.new("RGB", (1000, 400), "white")
    d = ImageDraw.Draw(img)
    d.line([(180, 120), (188, 120)], fill="black", width=2)     # "-" the detector left out
    d.line([(600, 300), (608, 300)], fill="black", width=2)     # a speck in the margin
    rgb = np.asarray(img)
    overlays, _ = graphics.extract(rgb, P.local_background(rgb), [((210, 100, 900, 140), None)], [])
    assert len(overlays) == 1 and overlays[0].bbox[0] < 180 < overlays[0].bbox[2]


def test_a_rule_through_the_middle_of_a_line_does_not_cut_it():
    from netra_ocr.layout.table import split_at_rules

    ink = np.zeros((200, 200), bool)
    ink[50:110, 60:100] = True
    # A "rule" found through the middle of a text line is the text itself.
    assert split_at_rules([(60, 40, 100, 120)], ink, [30, 80, 180], []) == [None]
    # One just inside its top edge is the row's rule: the box is trimmed below it.
    (piece,), = split_at_rules([(60, 40, 100, 120)], ink, [45, 180], [])
    assert piece[1] > 45


def test_leftovers_of_recognized_text_are_not_pictures():
    from netra_ocr.reconstruct import graphics
    from netra_ocr.reconstruct.style import Ink

    img = Image.new("RGB", (800, 400), "white")
    d = ImageDraw.Draw(img)
    box, ink = (100, 90, 600, 140), Ink(110, 590, 105, 125, 0.2, 0.3, (20, 60, 150), 20.0)
    d.rectangle([110, 105, 590, 125], fill=(20, 60, 150))          # the line's letters
    d.rectangle([200, 96, 300, 100], fill=(120, 150, 210))         # off-colour tops of tall strokes
    d.line([(110, 133), (560, 133)], fill=(20, 60, 150), width=2)  # an underline
    d.ellipse([400, 60, 560, 220], outline=(200, 30, 40), width=8)  # a seal across the line
    rgb = np.asarray(img)
    overlays, _ = graphics.extract(rgb, P.local_background(rgb), [(box, ink)], [])
    boxes = [o.bbox for o in overlays]
    assert not any(b[1] <= 98 <= b[3] and b[0] <= 250 <= b[2] for b in boxes)    # tops: erased
    underline = [b for b in boxes if b[1] <= 133 <= b[3] and b[3] - b[1] < 8]
    assert underline and min(b[0] for b in underline) <= 112 and max(b[2] for b in underline) >= 558
    assert any(o.kind == "stamp" for o in overlays)                              # seal: kept


def test_paragraph_added_in_the_editor_follows_its_block():
    from netra_ocr.reconstruct import enrich_document

    assets = AssetStore()
    assets.put("page-0", Image.new("RGB", (1240, 1754), "white"))
    blocks = [
        {"id": "a", "type": "paragraph", "page": 0, "text": "first", "bbox": [100, 100, 600, 140],
         "lines": [{"text": "first", "bbox": [100, 100, 600, 140]}]},
        {"id": "new", "type": "paragraph", "page": 0, "text": "typed in the editor", "bbox": None},
        {"id": "b", "type": "paragraph", "page": 0, "text": "second", "bbox": [100, 400, 600, 440],
         "lines": [{"text": "second", "bbox": [100, 400, 600, 440]}]},
    ]
    doc = Document.from_dict({"pages": [{"width": 1240, "height": 1754, "image": "page-0"}], "blocks": blocks}, assets)
    enrich_document(doc)
    plan = PL.to_dict(PL.build_document(doc)[0])
    import json
    json.dumps(plan)                                       # plain JSON for the editor
    order = [b["item"]["id"] for b in plan["bands"] if b["item"]]
    assert order == ["a", "new", "b"]
    added = next(b["item"] for b in plan["bands"] if b["item"] and b["item"]["id"] == "new")
    assert added["keep_lines"] is False and added["text"] == "typed in the editor"
    assert "typed in the editor" in zipfile.ZipFile(io.BytesIO(export(doc, "docx")[0])).read("word/document.xml").decode()


def test_paper_edges_go_but_a_printed_frame_stays():
    from netra_ocr.reconstruct import graphics

    def page(frame):
        img = Image.new("RGB", (800, 1100), "white")
        d = ImageDraw.Draw(img)
        lines = []
        for i in range(12):                                    # the text
            y = 150 + 60 * i
            for x in range(120, 680, 7):                       # letters: strokes, not one bar
                d.line([(x, y + 8), (x + 2, y + 22)], fill=(30, 30, 30), width=2)
            lines.append(((110, y, 690, y + 30), None))
        d.line([(120, 900), (680, 900)], fill=(30, 30, 30), width=2)   # a rule inside the text: stays
        lines.append(((110, 920, 690, 950), None))
        if frame:
            d.rectangle([60, 60, 740, 1040], outline=(20, 20, 20), width=3)
        else:                                                  # where the scanned sheet ends
            d.rectangle([752, 40, 770, 1060], fill=(90, 90, 90))           # right shadow, full height
            d.rectangle([30, 1062, 770, 1074], fill=(70, 70, 70))          # bottom shadow
            d.line([(40, 600), (40, 900)], fill=(60, 60, 60), width=4)     # left edge, partly
        rgb = np.asarray(img)
        return graphics.extract(rgb, P.local_background(rgb), lines, [])[0]

    edges = page(frame=False)
    assert [o for o in edges if o.bbox[1] <= 900 <= o.bbox[3] and o.bbox[2] - o.bbox[0] > 500]   # the rule
    assert not [o for o in edges if o.bbox[0] > 700 or o.bbox[1] > 1050 or o.bbox[2] < 100]
    frame = page(frame=True)
    assert len([o for o in frame if max(o.bbox[2] - o.bbox[0], o.bbox[3] - o.bbox[1]) > 500]) == 5   # 4 sides + rule
