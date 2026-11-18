"""Word export that rebuilds the original page: an editable look-alike.

Where ``docx_export`` re-flows the content into a clean document, this
writer follows ``reconstruct.plan``: each scanned page becomes a Word
section of the measured paper size; text keeps the scan's font, size,
weight, colour, line breaks, indents and exact vertical spacing; side-by-
side blocks become a borderless table row; logos, seals and signatures
are floating pictures at their position on the page. Text stays real,
editable Word text.

The fonts used for measuring are embedded (they're OFL-licensed), so the
document renders as intended on machines that don't have them; with
``fonts="system"`` the usual Khmer system fonts are named instead (Khmer
OS Siemreap, Khmer OS Muol Light, ...) and nothing is embedded.
"""

import io
import uuid
from typing import Dict, Set, Tuple

from ..document import Document
from ..reconstruct import enrich_document, is_enriched
from ..reconstruct import fonts as F
from ..reconstruct.plan import Band, Para, PagePlan, Table, build_document, natural_pitch
from .docx_export import _import_docx

EMU_PER_PT = 12700
SPACER_PT = 1.0       # height of the empty paragraphs Word needs between tables
BORDER = "808080"     # table grid lines: scanned grids are usually light


# Latin text is set in a standard font, not the Khmer face's Latin letters:
# a machine may have its own build of the Khmer font installed, which a word
# processor prefers over the embedded copy, and some builds (Debian's Noto
# Sans Khmer, for one) have no Latin letters at all.
# Arial throughout: the Latin letters of the Khmer OS fonts are sans-serif,
# and so are the digits in most Khmer forms and tables.
LATIN_FONTS = {"sans": "Arial", "serif": "Arial", "muol": "Arial"}


def _font_names(style: dict, mode: str) -> Tuple[str, str]:
    """(complex-script font, Latin font) names for a run."""
    fam = F.FAMILIES[style["font"]]
    if mode == "system":
        return fam.system_name, fam.system_name
    return fam.name, LATIN_FONTS[style["font"]]


class _Writer:
    def __init__(self, doc: Document, fonts: str = "embed"):
        _import_docx()
        from docx import Document as WordDocument
        self.doc = doc
        self.fonts = fonts
        self.word = WordDocument()
        self.used: Set[Tuple[str, bool]] = set()   # (family key, bold)
        self.z = 251658240
        self.last_was_table = False

    # ── runs and paragraphs ─────────────────────────────────────────────
    def _rpr(self, style: dict, scale: int = 100, size: float = None, khmer: bool = True):
        """A complete ``w:rPr`` in schema order (Word rejects misordered ones).

        Non-Khmer runs name their Latin font in every slot: some renderers
        classify Latin next to Khmer as complex script and would draw it
        with the Khmer face, which may have no Latin letters.
        """
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn

        rpr = OxmlElement("w:rPr")
        cs, latin = _font_names(style, self.fonts)
        if not khmer:
            cs = latin
        fonts = OxmlElement("w:rFonts")
        for attr, name in (("w:ascii", latin), ("w:hAnsi", latin), ("w:eastAsia", latin), ("w:cs", cs)):
            fonts.set(qn(attr), name)
        rpr.append(fonts)
        if style.get("bold"):
            rpr.append(OxmlElement("w:b"))
            rpr.append(OxmlElement("w:bCs"))
        colour = OxmlElement("w:color")
        colour.set(qn("w:val"), style.get("color", "000000").upper())
        rpr.append(colour)
        if scale != 100:
            w = OxmlElement("w:w")
            w.set(qn("w:val"), str(int(scale)))
            rpr.append(w)
        half_points = str(max(2, int(round((size or style["size"]) * 2))))
        for tag in ("w:sz", "w:szCs"):
            el = OxmlElement(tag)
            el.set(qn("w:val"), half_points)
            rpr.append(el)
        lang = OxmlElement("w:lang")
        lang.set(qn("w:val"), "en-US")
        lang.set(qn("w:bidi"), "km-KH")
        rpr.append(lang)
        if khmer:
            self.used.add((style["font"], bool(style.get("bold"))))
        return rpr

    def _run(self, paragraph, text: str, style: dict, scale: int = 100):
        """Add ``text`` as one run per script; returns the last run."""
        run = None
        for part, khmer in F.script_runs(text) or [("", True)]:
            run = paragraph.add_run(part)
            old = run._r.rPr
            if old is not None:
                run._r.remove(old)
            run._r.insert(0, self._rpr(style, scale, khmer=khmer))
        return run

    @staticmethod
    def _exact(paragraph, pitch: float, before: float = 0.0):
        from docx.enum.text import WD_LINE_SPACING
        from docx.shared import Pt
        pf = paragraph.paragraph_format
        pf.space_before = Pt(max(0.0, before))
        pf.space_after = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        pf.line_spacing = Pt(max(0.7, pitch))
        pf.widow_control = False

    def _mark_size(self, paragraph, size_pt: float = 1.0):
        """Shrink an empty paragraph's mark so it takes no more than its pitch."""
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        ppr = paragraph._p.get_or_add_pPr()
        rpr = OxmlElement("w:rPr")
        for tag in ("w:sz", "w:szCs"):
            el = OxmlElement(tag)
            el.set(qn("w:val"), str(max(2, int(size_pt * 2))))
            rpr.append(el)
        sect = ppr.find(qn("w:sectPr"))
        if sect is not None:
            sect.addprevious(rpr)
        else:
            ppr.append(rpr)

    def spacer(self, container, height: float):
        p = container.add_paragraph()
        self._exact(p, height)
        self._mark_size(p, min(1.0, height))
        self.last_was_table = False
        return p

    def para(self, container, p: Para, paragraph=None):
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt

        para = paragraph if paragraph is not None else container.add_paragraph()
        self._exact(para, p.pitch, p.space_before)
        pf = para.paragraph_format
        pf.alignment = {"center": WD_ALIGN_PARAGRAPH.CENTER, "right": WD_ALIGN_PARAGRAPH.RIGHT,
                        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY}.get(p.align, WD_ALIGN_PARAGRAPH.LEFT)
        pf.left_indent = Pt(p.left)
        pf.right_indent = Pt(p.right)
        pf.first_line_indent = Pt(p.first)
        if p.keep_lines:
            for i, row in enumerate(p.rows):
                run = self._run(para, row.text, p.style, row.scale)
                if i < len(p.rows) - 1:
                    run.add_break()
        else:
            lines = p.text.split("\n")
            for i, line in enumerate(lines):
                run = self._run(para, line, p.style)
                if i < len(lines) - 1:
                    run.add_break()
        self.last_was_table = False
        return para

    # ── tables ──────────────────────────────────────────────────────────
    def _table_props(self, table, widths, borders: bool, indent: float = 0.0, padding: float = 0.0):
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Pt

        tbl = table._tbl
        tblpr = tbl.tblPr
        for child in list(tblpr):
            if child.tag != qn("w:tblStyle"):
                tblpr.remove(child)

        def el(tag, **attrs):
            e = OxmlElement(tag)
            for k, v in attrs.items():
                e.set(qn(k), str(v))
            return e

        twips = [int(round(w * 20)) for w in widths]
        tblpr.append(el("w:tblW", **{"w:w": sum(twips), "w:type": "dxa"}))
        tblpr.append(el("w:tblInd", **{"w:w": int(round(indent * 20)), "w:type": "dxa"}))
        b = el("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            if borders:
                b.append(el(f"w:{edge}", **{"w:val": "single", "w:sz": 4, "w:space": 0, "w:color": BORDER}))
            else:
                b.append(el(f"w:{edge}", **{"w:val": "nil"}))
        tblpr.append(b)
        tblpr.append(el("w:tblLayout", **{"w:type": "fixed"}))
        mar = el("w:tblCellMar")
        pad = int(round(padding * 20))
        for side in ("top", "left", "bottom", "right"):
            mar.append(el(f"w:{side}", **{"w:w": pad if side in ("left", "right") else 0, "w:type": "dxa"}))
        tblpr.append(mar)
        tblpr.append(el("w:tblLook", **{"w:val": "0000", "w:firstRow": 0, "w:lastRow": 0,
                                        "w:firstColumn": 0, "w:lastColumn": 0,
                                        "w:noHBand": 1, "w:noVBand": 1}))
        table.autofit = False
        for i, col in enumerate(table.columns):
            col.width = Pt(widths[i])
            for cell in col.cells:
                cell.width = Pt(widths[i])

    def _before_table(self, space: float):
        """Tables can't carry space before, and two tables in a row merge:
        a spacer paragraph takes the gap."""
        if space >= SPACER_PT or self.last_was_table:
            self.spacer(self.word, max(SPACER_PT, space))
            return max(0.0, space - max(SPACER_PT, space))
        return 0.0

    def columns(self, band: Band, left_margin: float):
        widths = [c.x2 - c.x1 for c in band.columns]
        self._before_table(band.space_before)
        table = self.word.add_table(rows=1, cols=len(widths))
        self._table_props(table, widths, borders=False, indent=band.columns[0].x1 - left_margin)
        for ci, col in enumerate(band.columns):
            cell = table.cell(0, ci)
            first = cell.paragraphs[0]
            if not col.items:
                self._exact(first, SPACER_PT)
                self._mark_size(first)
            for k, item in enumerate(col.items):
                self.para(cell, item, paragraph=first if k == 0 else None)
        self.last_was_table = True

    def table(self, t: Table, space_before: float, left_margin: float):
        """A table with the scan's column widths and exact row heights, text
        aligned in each cell (across and up/down) as on the scan."""
        from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt

        self._before_table(space_before)
        nrows, ncols = len(t.cells), len(t.col_widths)
        table = self.word.add_table(rows=nrows, cols=ncols)
        self._table_props(table, t.col_widths, borders=t.borders, indent=t.left - left_margin,
                          padding=t.padding)
        align = {"center": WD_ALIGN_PARAGRAPH.CENTER, "right": WD_ALIGN_PARAGRAPH.RIGHT}
        for ri, row in enumerate(t.cells):
            height = t.row_heights[ri]
            table.rows[ri].height = Pt(height)
            table.rows[ri].height_rule = WD_ROW_HEIGHT_RULE.EXACTLY
            style = t.header_style if (ri == 0 and t.header_style) else t.style
            for ci, c in enumerate(row[:ncols]):
                cell = table.cell(ri, ci)
                cell.vertical_alignment = {"top": WD_CELL_VERTICAL_ALIGNMENT.TOP,
                                           "bottom": WD_CELL_VERTICAL_ALIGNMENT.BOTTOM}.get(
                    c.valign, WD_CELL_VERTICAL_ALIGNMENT.CENTER)
                p = cell.paragraphs[0]
                lines = c.lines or [""]
                pitch = min(natural_pitch(style), height / len(lines))
                self._exact(p, max(style["size"] * 1.05, pitch))
                p.paragraph_format.alignment = align.get(c.align, WD_ALIGN_PARAGRAPH.LEFT)
                for li, line in enumerate(lines):
                    run = self._run(p, line, style, c.scale)
                    if li < len(lines) - 1:
                        run.add_break()
        self.last_was_table = True

    # ── floating pictures ───────────────────────────────────────────────
    def picture(self, paragraph, asset_id: str, x: float, y: float, w: float, h: float):
        from docx.oxml import parse_xml
        from docx.oxml.ns import nsdecls, qn
        from docx.shared import Emu

        png = self.doc.assets.png_bytes(asset_id)
        if png is None or w <= 0 or h <= 0:
            return
        cx, cy = int(w * EMU_PER_PT), int(h * EMU_PER_PT)
        run = paragraph.add_run()
        inline = run.add_picture(io.BytesIO(png), width=Emu(cx), height=Emu(cy))._inline
        self.z += 1
        anchor = parse_xml(
            f'<wp:anchor {nsdecls("wp")} distT="0" distB="0" distL="0" distR="0" simplePos="0" '
            f'relativeHeight="{self.z}" behindDoc="0" locked="0" layoutInCell="0" allowOverlap="1">'
            f'<wp:simplePos x="0" y="0"/>'
            f'<wp:positionH relativeFrom="page"><wp:posOffset>{int(x * EMU_PER_PT)}</wp:posOffset></wp:positionH>'
            f'<wp:positionV relativeFrom="page"><wp:posOffset>{int(y * EMU_PER_PT)}</wp:posOffset></wp:positionV>'
            f'<wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:wrapNone/></wp:anchor>')
        for tag in ("wp:docPr", "wp:cNvGraphicFramePr", "a:graphic"):
            child = inline.find(qn(tag))
            if child is not None:
                anchor.append(child)
        inline.getparent().replace(inline, anchor)

    # ── pages ───────────────────────────────────────────────────────────
    def page(self, plan: PagePlan, first: bool):
        from docx.enum.section import WD_SECTION
        from docx.oxml.ns import qn
        from docx.shared import Pt

        if first:
            section = self.word.sections[0]
        else:
            section = self.word.add_section(WD_SECTION.NEW_PAGE)
            # The paragraph that now ends the previous section must not take
            # a full line (it could push that page's last line over).
            body = self.word.element.body
            for p in reversed(body.findall(qn("w:p"))):
                ppr = p.find(qn("w:pPr"))
                if ppr is not None and ppr.find(qn("w:sectPr")) is not None:
                    from docx.text.paragraph import Paragraph
                    para = Paragraph(p, self.word._body)
                    self._exact(para, SPACER_PT)
                    self._mark_size(para)
                    break
        left, right, top, bottom = plan.margins
        section.page_width, section.page_height = Pt(plan.width), Pt(plan.height)
        section.left_margin, section.right_margin = Pt(left), Pt(right)
        section.top_margin = Pt(max(0.0, top - SPACER_PT))
        section.bottom_margin = Pt(bottom)
        section.header_distance = section.footer_distance = Pt(0)
        section.gutter = Pt(0)

        # Pictures are anchored to the page from its first paragraph.
        holder = self.spacer(self.word, SPACER_PT)
        for f in plan.floats:
            self.picture(holder, f.image, f.x, f.y, f.width, f.height)

        for band in plan.bands:
            if band.item is None:
                self.columns(band, left)
            elif isinstance(band.item, Para):
                band.item.space_before = band.space_before
                self.para(self.word, band.item)
            else:
                self.table(band.item, band.space_before, left)
        if self.last_was_table:
            self.spacer(self.word, SPACER_PT)

    # ── fonts ───────────────────────────────────────────────────────────
    def embed_fonts(self):
        """Embed the fonts used, obfuscated as the OOXML spec requires
        (ECMA-376 Part 1, 17.8.1): the first 32 bytes are XORed with a key
        taken from the font's GUID."""
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
        from docx.opc.packuri import PackURI
        from docx.opc.part import Part
        from docx.oxml import OxmlElement, parse_xml
        from docx.oxml.ns import qn
        from lxml import etree

        if not self.used:
            return
        table_part = self.word.part.part_related_by(RT.FONT_TABLE)
        root = etree.fromstring(table_part.blob)
        by_family: Dict[str, Set[bool]] = {}
        for fam, bold in self.used:
            by_family.setdefault(fam, set()).add(bold)
        n = 0
        for fam_key, weights in sorted(by_family.items()):
            fam = F.FAMILIES[fam_key]
            font = OxmlElement("w:font")
            font.set(qn("w:name"), fam.name)
            for tag, val in (("w:charset", "00"), ("w:family", "auto"), ("w:pitch", "variable")):
                e = OxmlElement(tag)
                e.set(qn("w:val"), val)
                font.append(e)
            for bold in sorted(weights):
                tag = "w:embedBold" if bold and fam.bold else "w:embedRegular"
                if any(c.tag == qn(tag) for c in font):
                    continue
                path = fam.file(bold)
                with open(path, "rb") as fh:
                    data = bytearray(fh.read())
                guid = "{" + str(uuid.uuid4()).upper() + "}"
                hexd = guid.strip("{}").replace("-", "")
                key = bytes.fromhex(hexd)[::-1]
                for i in range(32):
                    data[i] ^= key[i % 16]
                n += 1
                part = Part(PackURI(f"/word/fonts/font{n}.odttf"),
                            "application/vnd.openxmlformats-officedocument.obfuscatedFont",
                            bytes(data), self.word.part.package)
                rid = table_part.relate_to(part, RT.FONT)
                e = OxmlElement(tag)
                e.set(qn("r:id"), rid)
                e.set(qn("w:fontKey"), guid)
                font.append(e)
            for existing in root.findall(qn("w:font")):
                if existing.get(qn("w:name")) == fam.name:
                    root.remove(existing)
            root.append(font)
        table_part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)

        settings = self.word.settings.element
        if settings.find(qn("w:embedTrueTypeFonts")) is None:
            before = ("w:writeProtection", "w:view", "w:zoom", "w:removePersonalInformation",
                      "w:removeDateAndTime", "w:doNotDisplayPageBoundaries",
                      "w:displayBackgroundShape", "w:printPostScriptOverText",
                      "w:printFractionalCharacterWidth", "w:printFormsData")
            el = parse_xml(f'<w:embedTrueTypeFonts xmlns:w="{settings.nsmap["w"]}"/>')
            pos = 0
            for i, child in enumerate(settings):
                if child.tag in {qn(t) for t in before}:
                    pos = i + 1
            settings.insert(pos, el)

    def save(self) -> bytes:
        if self.fonts == "embed":
            self.embed_fonts()
        buf = io.BytesIO()
        self.word.save(buf)
        return buf.getvalue()


def to_docx_layout(doc: Document, fonts: str = "embed") -> bytes:
    """Rebuild ``doc``'s pages as an editable Word document.

    Pages without measurements (documents made without ``reconstruct``)
    are measured first from their stored page images; the measurements
    are added to ``doc`` in place, so a second export doesn't repeat them.
    """
    if fonts not in ("embed", "system"):
        raise ValueError("fonts must be 'embed' or 'system'")
    if not all(is_enriched(p) for p in doc.pages):
        enrich_document(doc)
    writer = _Writer(doc, fonts)
    plans = build_document(doc)
    if not plans:
        return writer.save()
    for i, plan in enumerate(plans):
        writer.page(plan, first=(i == 0))
    return writer.save()
