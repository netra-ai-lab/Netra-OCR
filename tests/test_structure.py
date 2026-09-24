import numpy as np

from netra_ocr.document import AssetStore
from netra_ocr.layout.base import LayoutRegion
from netra_ocr.structure import build_blocks, infer_align, join_lines, plan_page


def test_join_lines_khmer_wraps_without_space_latin_with_space():
    boxes = [(0, 0, 100, 20), (0, 30, 100, 50)]
    assert join_lines(boxes, ["សួស្តី", "ពិភពលោក"], keep_breaks=False) == "សួស្តីពិភពលោក"
    assert join_lines(boxes, ["hello", "world"], keep_breaks=False) == "hello world"
    assert join_lines(boxes, ["a", "b"], keep_breaks=True) == "a\nb"
    same_row = [(0, 0, 100, 20), (120, 0, 200, 20)]
    assert join_lines(same_row, ["ក", "ខ"], keep_breaks=True) == "ក ខ"


def test_infer_align():
    content = (0, 1000)
    assert infer_align((300, 0, 700, 20), [(300, 0, 700, 20)], content) == "center"
    assert infer_align((600, 0, 1000, 20), [(600, 0, 1000, 20)], content) == "right"
    wrapped = [(0, 0, 1000, 20), (0, 25, 500, 45)]
    assert infer_align((0, 0, 1000, 45), wrapped, content) == "left"


def test_page_structure_end_to_end_without_models():
    img = np.full((600, 500, 3), 255, np.uint8)
    regions = [LayoutRegion("title", (100, 10, 400, 60)),
               LayoutRegion("plain text", (20, 100, 480, 200)),
               LayoutRegion("figure", (20, 300, 200, 450)),
               LayoutRegion("figure_caption", (20, 455, 200, 480)),
               LayoutRegion("abandon", (20, 560, 480, 590))]
    lines = [(120, 15, 380, 50), (20, 105, 480, 140), (20, 150, 300, 185),
             (30, 320, 180, 340),            # text inside the figure: dropped
             (25, 458, 190, 478), (40, 562, 460, 588)]
    texts = ["ចំណងជើង", "បន្ទាត់ទី", "មួយ", "label", "caption", "footer"]
    units = plan_page((500, 600), lines, [], regions)
    assets = AssetStore()
    blocks = build_blocks(0, img, units, lines, texts, [], assets)
    assert [b["type"] for b in blocks] == ["heading", "paragraph", "figure", "caption", "page_footer"]
    assert blocks[0]["level"] == 1 and blocks[0]["align"] == "center"
    assert blocks[1]["text"] == "បន្ទាត់ទីមួយ"
    assert blocks[2]["image"] in assets


# ── behaviours tuned on test_images/ (scanned Khmer letters and notices) ──

from netra_ocr.structure import (consolidate_lines, has_content, ink_mask,  # noqa: E402
                                 split_paragraphs)


def test_ornaments_and_stamp_noise_are_not_text():
    assert not has_content("@-@.@")
    assert not has_content("@-@ 7 @.@ 5 % .")
    assert not has_content(" - ")
    assert has_content("សេចក្តីជូនដំណឹង")
    assert has_content("១២.០៩.១៩៨៤")
    assert has_content("202 726 7742")


def test_duplicate_and_overlapping_line_boxes_are_consolidated():
    img = np.full((100, 400, 3), 255, np.uint8)
    boxes = [(10, 10, 390, 40),      # full row
             (12, 12, 200, 38),      # fragment inside it: dropped
             (10, 60, 220, 90),      # two overlapping fragments of one row: fused
             (150, 61, 390, 89)]
    crops = [None] * len(boxes)
    out, out_crops = consolidate_lines(img, boxes, crops)
    assert out == [(10, 10, 390, 40), (10, 60, 390, 90)]
    assert out_crops[1].size == (380, 30)     # re-cropped from the page


def test_stacked_rows_are_not_fused():
    img = np.full((100, 400, 3), 255, np.uint8)
    boxes = [(44, 173, 175, 212), (42, 192, 222, 228)]   # ceroc letterhead: two rows
    out, _ = consolidate_lines(img, boxes, [None, None])
    assert len(out) == 2


def test_list_items_and_short_rows_split_paragraphs():
    boxes = [(0, 0, 500, 20), (20, 25, 1000, 45), (20, 50, 600, 70),
             (0, 75, 1000, 95), (20, 100, 400, 120), (0, 125, 1000, 145)]
    texts = ["- first item", "second item starts and wraps", "onto a short row",
             "- third item wraps", "short", "• fourth"]
    paras = split_paragraphs(range(6), boxes, texts)
    assert [p for p, _ in paras] == [[0], [1, 2], [3, 4], [5]]


def test_centred_and_narrow_blocks_keep_line_breaks():
    centred = [(300, 0, 700, 20), (350, 25, 650, 45)]
    assert split_paragraphs([0, 1], centred, ["a", "b"]) == [([0, 1], True)]
    narrow = [(0, 0, 200, 20), (0, 25, 190, 45)]           # a motto in the corner
    assert split_paragraphs([0, 1], narrow, ["a", "b"], content_width=1000) == [([0, 1], True)]


def test_page_centred_heading_over_offset_body():
    # Body text column is shifted right (400..900 on a 1000 px page); a title
    # centred on the page is still centred.
    assert infer_align((420, 0, 580, 20), [(420, 0, 580, 20)], (300, 900), page_width=1000) == "center"
    # A full-width row with equal margins is not centred text.
    assert infer_align((50, 0, 950, 20), [(50, 0, 950, 20)], (40, 960), page_width=1000) == "left"


def test_letterhead_abandon_is_body_text_and_margin_abandon_is_footer():
    img = np.full((1000, 800, 3), 255, np.uint8)
    regions = [LayoutRegion("abandon", (500, 20, 760, 130)),     # motto: 13% down -> body
               LayoutRegion("abandon", (100, 950, 700, 980))]    # footer zone
    lines = [(510, 25, 750, 60), (560, 70, 700, 120), (110, 952, 690, 978)]
    units = plan_page((800, 1000), lines, [], regions)
    blocks = build_blocks(0, img, units, lines, ["ព្រះរាជាណាចក្រកម្ពុជា", "ជាតិ សាសនា", "address"], [],
                          AssetStore())
    assert [b["type"] for b in blocks] == ["paragraph", "page_footer"]
    assert blocks[0]["text"] == "ព្រះរាជាណាចក្រកម្ពុជា\nជាតិ សាសនា"


def test_text_only_figure_dissolves_into_text_and_keeps_its_logo():
    img = np.full((600, 600, 3), 255, np.uint8)
    img[20:120, 250:350] = 0                  # the emblem (a detected logo)
    img[140:160, 200:400] = 0                 # text lines
    img[170:190, 220:380] = 0
    regions = [LayoutRegion("figure", (190, 10, 410, 200))]
    lines = [(200, 138, 400, 162), (220, 168, 380, 192)]
    logos = [(245, 15, 355, 125)]
    units = plan_page((600, 600), lines, logos, regions, ink=ink_mask(img))
    assert [u.label for u in units] == ["logo", "plain text"]
    # Without the ink mask the figure is trusted and the text is part of it.
    units = plan_page((600, 600), lines, logos, regions)
    assert [u.label for u in units] == ["figure"]


def test_split_first_line_rejoins_but_spaced_subtitle_does_not():
    img = np.full((400, 500, 3), 255, np.uint8)
    body = [(20, 102, 480, 122), (20, 120, 480, 140), (20, 138, 300, 158)]
    first = [(50, 84, 470, 104)]              # indented first line, cut off by the layout model
    regions = [LayoutRegion("plain text", (50, 84, 470, 104)), LayoutRegion("plain text", (20, 102, 480, 158))]
    units = plan_page((500, 400), first + body, [], regions)
    assert len(units) == 1 and len(units[0].lines) == 4

    subtitle = [(80, 60, 420, 80)]            # centred, with extra space below
    regions = [LayoutRegion("plain text", (80, 60, 420, 80)), LayoutRegion("plain text", (20, 102, 480, 158))]
    units = plan_page((500, 400), subtitle + body, [], regions)
    assert len(units) == 2


def test_stacked_titles_become_one_heading():
    img = np.full((400, 500, 3), 255, np.uint8)
    regions = [LayoutRegion("title", (150, 10, 350, 40)), LayoutRegion("title", (220, 42, 280, 70))]
    lines = [(150, 10, 350, 40), (220, 42, 280, 70)]
    units = plan_page((500, 400), lines, [], regions)
    blocks = build_blocks(0, img, units, lines, ["សេចក្តីជូនដំណឹង", "ស្តីពី"], [], AssetStore())
    assert len(blocks) == 1 and blocks[0]["text"] == "សេចក្តីជូនដំណឹង\nស្តីពី"
