"""netra_ocr.imaging must give exactly what the OpenCV calls it replaces give."""

from pathlib import Path

import numpy as np
import pytest

from netra_ocr import imaging

cv2 = pytest.importorskip("cv2")
PAGES = sorted((Path(__file__).resolve().parent.parent / "test_images").iterdir())


def sorted_rows(stats):
    return sorted(map(tuple, np.asarray(stats).tolist()))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_pages_match_opencv(page):
    bgr = cv2.imread(str(page))
    assert np.array_equal(imaging.read_bgr(page), bgr)
    assert np.array_equal(imaging.bgr_to_rgb(bgr), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    assert np.array_equal(imaging.to_gray(bgr), gray)
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    assert imaging.otsu_threshold(gray) == threshold
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    assert sorted_rows(imaging.connected_components(binary)) == sorted_rows(stats[1:])


def test_gray_matches_opencv_on_every_colour():
    c = np.arange(1 << 24, dtype=np.uint32)
    bgr = np.stack([c & 255, (c >> 8) & 255, c >> 16], -1).astype(np.uint8).reshape(4096, 4096, 3)
    assert np.array_equal(imaging.to_gray(bgr), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))


def test_random_masks_and_edge_cases():
    rng = np.random.default_rng(0)
    for _ in range(300):
        shape = rng.integers(1, 50, size=2)
        mask = (rng.random(shape) < rng.random()).astype(np.uint8) * 255
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        assert sorted_rows(imaging.connected_components(mask)) == sorted_rows(stats[1:])
        gray = rng.integers(0, 256, shape, dtype=np.uint8)
        assert imaging.otsu_threshold(gray) == cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0]
    for flat in (np.zeros((5, 5), np.uint8), np.full((5, 5), 200, np.uint8)):
        assert imaging.otsu_threshold(flat) == cv2.threshold(flat, 0, 255, cv2.THRESH_OTSU)[0]
    assert imaging.connected_components(np.zeros((4, 4), np.uint8)).shape == (0, 5)


def test_unreadable_image(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with pytest.raises(FileNotFoundError):
        imaging.read_bgr(tmp_path / "bad.png")
