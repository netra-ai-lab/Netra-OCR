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
    from docx.oxml.ns import qn

    data = export(enriched, "docx")[0]
    word = Word(io.BytesIO(data))
    sec = word.sections[0]
    assert abs(sec.page_width.pt - 595.3) < 1 and abs(sec.page_height.pt - 841.9) < 1

    xml = word.element.xml
    assert xml.count("<wp:anchor") >= 1                         # the seal floats on the page
    assert 'w:cs="Moul"' in xml                                 # title font
    assert 'w:lineRule="exact"' in xml
    texts = [p.text for p in word.paragraphs]
    assert "\n".join(BODY) in texts                             # source line breaks kept
    assert len(word.tables) == 1                                # the side-by-side letterhead
    assert [c.text for c in word.tables[0].rows[0].cells] == ["ក្រសួងសុខាភិបាល", "ព្រះរាជាណាចក្រកម្ពុជា"]

    # Fonts are embedded, obfuscated per ECMA-376 17.8.1, and recoverable.
    z = zipfile.ZipFile(io.BytesIO(data))
    assert b"<w:embedTrueTypeFonts" in z.read("word/settings.xml")
    from lxml import etree
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


def test_system_fonts_mode(enriched):
    data = export(enriched, "docx", fonts="system")[0]
    z = zipfile.ZipFile(io.BytesIO(data))
    assert not any(n.startswith("word/fonts/") for n in z.namelist())
    assert b"Khmer OS Muol Light" in z.read("word/document.xml")


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
