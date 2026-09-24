from netra_ocr.layout.base import LayoutRegion
from netra_ocr.layout.doclayout import dedupe_regions
from netra_ocr.layout.order import assign_lines, group_orphan_lines, xy_cut_order
from netra_ocr.layout.table import build_grid


def test_xy_cut_reads_columns_between_full_width_titles():
    boxes = {
        "title": (50, 10, 550, 40),
        "left1": (50, 60, 290, 200), "left2": (50, 210, 290, 400),
        "right1": (310, 60, 550, 300), "right2": (310, 310, 550, 400),
        "footer": (50, 420, 550, 440),
    }
    order = xy_cut_order(list(boxes), key=lambda k: boxes[k])
    assert order == ["title", "left1", "left2", "right1", "right2", "footer"]


def test_xy_cut_tolerates_slightly_overlapping_lines():
    lines = [(0, 0, 100, 22), (0, 20, 100, 42), (0, 40, 100, 62)]
    assert xy_cut_order([2, 0, 1], key=lambda i: lines[i]) == [0, 1, 2]


def test_assign_lines_prefers_the_smaller_region_and_reports_orphans():
    regions = [LayoutRegion("figure", (0, 0, 400, 400)),
               LayoutRegion("figure_caption", (0, 350, 400, 400))]
    lines = [(10, 360, 300, 390),   # inside both -> caption (smaller)
             (10, 10, 300, 40),     # figure only
             (10, 500, 300, 530)]   # nowhere
    out = assign_lines(lines, regions)
    assert out[1] == [0] and out[0] == [1] and out[-1] == [2]


def test_orphan_lines_group_into_paragraphs():
    lines = [(0, 0, 300, 20), (0, 25, 280, 45), (0, 200, 300, 220)]
    assert group_orphan_lines(lines, [0, 1, 2]) == [[0, 1], [2]]


def test_dedupe_drops_cross_class_duplicates_and_swallowing_boxes():
    regions = [LayoutRegion("title", (200, 41, 341, 92), 0.47),
               LayoutRegion("abandon", (200, 41, 341, 92), 0.20),
               LayoutRegion("title", (142, 143, 391, 184), 0.47),
               LayoutRegion("plain text", (98, 185, 434, 205), 0.24),
               LayoutRegion("title", (97, 143, 434, 205), 0.23)]
    kept = dedupe_regions(regions)
    assert [(r.label, r.bbox) for r in kept] == [
        ("title", (200, 41, 341, 92)), ("title", (142, 143, 391, 184)), ("plain text", (98, 185, 434, 205))]


def test_table_grid_from_line_boxes():
    lines = [((10, 10, 60, 30), "Name"), ((200, 10, 260, 30), "Age"),
             ((10, 50, 90, 70), "Dara"), ((200, 50, 230, 70), "31"),
             ((10, 90, 80, 110), "Sokha"), ((200, 90, 230, 110), "27")]
    assert build_grid(lines) == [["Name", "Age"], ["Dara", "31"], ["Sokha", "27"]]


def test_table_grid_gives_up_without_columns():
    lines = [((10, 10, 300, 30), "a b"), ((10, 50, 300, 70), "c d"),
             ((10, 90, 300, 110), "e f"), ((10, 130, 300, 150), "g h")]
    assert build_grid(lines) is None
