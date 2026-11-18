"""Page geometry and paper colour, measured from the scan.

A scan or photo has no physical size, only pixels. The page is matched to
a standard paper size by aspect ratio; the scale (DPI) is then chosen so
the whole scan fits on that paper. PDFs are rendered at a known DPI, so
their size is exact.
"""

from typing import List, Optional, Tuple

import numpy as np

from ..imaging import otsu_threshold, to_gray

# Paper sizes in points, portrait.
PAPERS = {
    "A4": (595.28, 841.89),
    "Letter": (612.0, 792.0),
    "Legal": (612.0, 1008.0),
}
# A5 is left out on purpose: it has A4's aspect ratio, so a photo can't tell
# them apart, and letters are far more often A4.
# How far a scan's aspect ratio may be from a paper's and still count as it.
# Photos of letters are rarely cropped exactly to the sheet.
ASPECT_TOLERANCE = 0.08


def page_geometry(width_px: int, height_px: int,
                  dpi: Optional[float] = None) -> Tuple[float, List[float], Optional[str]]:
    """``(dpi, [width_pt, height_pt], paper_name)`` for a page of this size.

    With a known ``dpi`` (PDF renders), the page is exactly the source size.
    Otherwise the closest paper size by aspect ratio is chosen (portrait or
    landscape) and the scan is scaled to fit inside it; with no close
    match, the page is A4-wide (or A4-landscape-wide) with the scan's own
    aspect ratio.
    """
    w, h = max(1, int(width_px)), max(1, int(height_px))
    if dpi:
        return float(dpi), [round(w * 72 / dpi, 2), round(h * 72 / dpi, 2)], None

    aspect = h / w
    best = None
    for name, (pw, ph) in PAPERS.items():
        for paper_w, paper_h in ((pw, ph), (ph, pw)):
            err = abs(aspect / (paper_h / paper_w) - 1)
            if err <= ASPECT_TOLERANCE and (best is None or err < best[0]):
                best = (err, name, paper_w, paper_h)
    if best is not None:
        _, name, paper_w, paper_h = best
        # Fit both dimensions: the scan never overflows the sheet.
        dpi = max(w / (paper_w / 72), h / (paper_h / 72))
        return dpi, [paper_w, paper_h], name

    paper_w = PAPERS["A4"][1] if w > h else PAPERS["A4"][0]
    dpi = w / (paper_w / 72)
    return dpi, [round(paper_w, 2), round(h * 72 / dpi, 2)], None


def ink_threshold(gray: np.ndarray) -> int:
    """Grey level below which a pixel is ink, for the whole page."""
    return min(otsu_threshold(gray), 200)


def paper_colour(rgb: np.ndarray, gray: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
    """Median colour of the paper (non-ink pixels) as ``(r, g, b)``."""
    if gray is None:
        gray = to_gray(rgb[..., ::-1])
    sample = rgb[gray > ink_threshold(gray)]
    if len(sample) == 0:
        return (255, 255, 255)
    if len(sample) > 200_000:
        sample = sample[:: len(sample) // 200_000]
    return tuple(int(v) for v in np.median(sample, axis=0))


def local_background(rgb: np.ndarray, tile: Optional[int] = None) -> np.ndarray:
    """A smooth per-pixel estimate of the paper colour (float32 HxWx3).

    Photos of documents are unevenly lit, so "how far is this pixel from the
    paper" has to be asked locally. Each tile takes the median of its
    brightest 40% of pixels (paper, not ink), and the tile grid is scaled
    back up smoothly.
    """
    from PIL import Image

    h, w = rgb.shape[:2]
    tile = tile or max(12, min(h, w) // 20)
    ny, nx = max(1, h // tile), max(1, w // tile)
    gray = rgb.astype(np.float32).mean(axis=2)
    small = np.zeros((ny, nx, 3), dtype=np.float32)
    for j in range(ny):
        y1, y2 = j * h // ny, (j + 1) * h // ny
        for i in range(nx):
            x1, x2 = i * w // nx, (i + 1) * w // nx
            g = gray[y1:y2, x1:x2]
            cut = np.percentile(g, 60)
            px = rgb[y1:y2, x1:x2][g >= cut]
            small[j, i] = np.median(px, axis=0) if len(px) else 255
    # A tile that is all ink (a logo, a seal) would otherwise call the ink
    # "paper": take the brighter of each tile and its neighbourhood's median.
    lum = small.mean(axis=2)
    padded = np.pad(small, ((1, 1), (1, 1), (0, 0)), mode="edge")
    plum = np.pad(lum, 1, mode="edge")
    for j in range(ny):
        for i in range(nx):
            win = plum[j:j + 3, i:i + 3]
            k = np.unravel_index(np.argmax(win), win.shape)
            if win[k] - lum[j, i] > 40:
                small[j, i] = padded[j + k[0], i + k[1]]
    up = Image.fromarray(np.clip(small, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return np.asarray(up, dtype=np.float32)
