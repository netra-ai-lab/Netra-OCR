"""Which font, weight and size a scanned line was set in: render and compare.

For each line we already have the recognized text. Rendering that exact
text in each candidate font and measuring the render the same way as the
scan (``style.measure``) gives, per font:

* the scale that makes the render's core band as tall as the scan's,
  which converts straight to a point size;
* how wide the text would be at that size, against the scan's width;
* how heavy its strokes are, against the scan's.

The font whose render matches best on width and weight is the font. Moul
(the formal Khmer title face) is far wider and heavier than body faces at
the same core height, so the two separate cleanly even on small scans.
"""

import os
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from .style import Ink, measure

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
REF_PX = 48          # reference render size (em, px): width and a first size guess
HAS_SHAPING = features.check("raqm")


@dataclass(frozen=True)
class Family:
    key: str
    regular: str
    bold: Optional[str]
    name: str            # family name inside the TTF (and in Word, when embedded)
    system_name: str     # the usual installed equivalent on Khmer-language PCs
    has_latin: bool
    # How text set in the usual system font can compare with our reference
    # render (width, stroke, density ratios): anything between 1 and these
    # counts as a match. Letters mostly use Khmer OS Muol Light, which is
    # narrower and lighter than Moul (measured on real letters).
    expect: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def file(self, bold: bool) -> str:
        return os.path.join(FONT_DIR, self.bold if bold and self.bold else self.regular)


FAMILIES: Dict[str, Family] = {
    "muol": Family("muol", "Moul-Regular.ttf", None, "Moul", "Khmer OS Muol Light", True,
                   expect=(0.91, 0.85, 0.95)),
    "sans": Family("sans", "NotoSansKhmer-Regular.ttf", "NotoSansKhmer-Bold.ttf",
                   "Noto Sans Khmer", "Khmer OS Siemreap", True),
    "serif": Family("serif", "NotoSerifKhmer-Regular.ttf", "NotoSerifKhmer-Bold.ttf",
                    "Noto Serif Khmer", "Khmer OS Battambang", False),
}
DEFAULT_FAMILY = "sans"
# (family, bold) pairs tried for each line.
CANDIDATES: List[Tuple[str, bool]] = [("sans", False), ("sans", True), ("serif", False),
                                      ("serif", True), ("muol", False)]


def _is_khmer(ch: str) -> bool:
    return "ក" <= ch <= "៿" or "᧠" <= ch <= "᧿"


def script_runs(text: str) -> List[Tuple[str, bool]]:
    """Split text into (run, is_khmer) runs; spaces and marks join the run
    they're in, so a Latin fallback font only gets non-Khmer letters."""
    runs: List[Tuple[str, bool]] = []
    for ch in text:
        k = _is_khmer(ch)
        neutral = ch.isspace() or unicodedata.category(ch) in ("Mn", "Cf")
        if runs and (neutral or runs[-1][1] == k):
            runs[-1] = (runs[-1][0] + ch, runs[-1][1])
        else:
            runs.append((ch, k))
    return runs


@lru_cache(maxsize=32)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    engine = ImageFont.Layout.RAQM if HAS_SHAPING else ImageFont.Layout.BASIC
    return ImageFont.truetype(path, size, layout_engine=engine)


def run_fonts(family: str, bold: bool) -> Tuple[str, str]:
    """(Khmer font file, Latin font file) used to render one family."""
    fam = FAMILIES[family]
    khmer = fam.file(bold)
    latin = khmer if fam.has_latin else FAMILIES["sans"].file(bold)
    return khmer, latin


@lru_cache(maxsize=4096)
def render(text: str, family: str, bold: bool, size: int = REF_PX) -> np.ndarray:
    """Grey render (ink dark on white) of ``text``, tightly padded."""
    khmer_path, latin_path = run_fonts(family, bold)
    runs = script_runs(text)
    fonts = [_font(khmer_path if k else latin_path, size) for _, k in runs]
    width = int(sum(f.getlength(r) for (r, _), f in zip(runs, fonts))) + size
    img = Image.new("L", (max(width, 4), size * 3), 255)
    draw = ImageDraw.Draw(img)
    x = size // 2
    for (r, _), f in zip(runs, fonts):
        draw.text((x, size), r, font=f, fill=0, anchor="ls")
        x += f.getlength(r)
    arr = np.asarray(img)
    rows = np.nonzero((arr < 128).any(axis=1))[0]
    if len(rows) == 0:
        return arr[:1]
    pad = size // 8
    return arr[max(0, rows[0] - pad): rows[-1] + 1 + pad]


def text_width_pt(text: str, family: str, bold: bool, size_pt: float) -> float:
    """Advance width of ``text`` set at ``size_pt`` (points), as Word would lay it out."""
    khmer_path, latin_path = run_fonts(family, bold)
    total = 0.0
    for run, k in script_runs(text):
        total += _font(khmer_path if k else latin_path, REF_PX).getlength(run)
    return total * size_pt / REF_PX


@dataclass
class Fit:
    family: str
    bold: bool
    size_px: float       # em size, in scan pixels
    width_ratio: float   # scan width / render width at that size
    stroke_ratio: float  # scan stroke / render stroke
    density_ratio: float


def _measure_at(text: str, family: str, bold: bool, size_px: float) -> Optional[Ink]:
    """Measure ``text`` rendered directly at ``size_px`` -- the scan's own
    scale -- so stroke weight and pixel rounding compare like for like
    (a downscaled large render has thinner strokes than a small one)."""
    img = render(text, family, bold, max(4, int(round(size_px))))
    return measure(img)


def fit(scan: Ink, text: str, family: str, bold: bool) -> Optional[Fit]:
    """Fit one candidate font to one scanned line."""
    if not text.strip() or scan.core < 2:
        return None
    img = render(text, family, bold)
    ref = measure(img, threshold=128)
    if ref is None or ref.core == 0:
        return None
    # First guess from the large render, then refine at the scan's scale
    # (the core band of a small render isn't exactly a scaled large one).
    size = REF_PX * scan.core / ref.core
    small = _measure_at(text, family, bold, size)
    if small is None or small.core == 0:
        return None
    size *= scan.core / small.core
    small = _measure_at(text, family, bold, size)
    if small is None or small.width == 0 or small.stroke == 0 or small.density == 0:
        return None
    return Fit(family=family, bold=bold, size_px=size,
               width_ratio=scan.width / (ref.width * size / REF_PX),
               stroke_ratio=scan.stroke / small.stroke,
               density_ratio=scan.density / small.density)


def score(f: Fit, stroke_bias: float = 1.0, density_bias: float = 1.0) -> float:
    """Lower is better. Justified lines are stretched, so a scan wider than
    the render is penalised less than one narrower than it."""
    ew, es, ed = FAMILIES[f.family].expect

    def off(ratio, expect):
        """Log distance from the accepted band between ``expect`` and 1."""
        lo, hi = min(1.0, expect), max(1.0, expect)
        if ratio > hi:
            return float(np.log(ratio / hi))
        if ratio < lo:
            return float(np.log(ratio / lo))
        return 0.0

    w = off(f.width_ratio, ew)
    w = 0.5 * w if w > 0 else w
    return (abs(w) + abs(off(f.stroke_ratio / stroke_bias, es))
            + 0.5 * abs(off(f.density_ratio / density_bias, ed)))


def fit_all(scan: Ink, text: str,
            candidates: Sequence[Tuple[str, bool]] = CANDIDATES) -> Dict[Tuple[str, bool], Fit]:
    out = {}
    for family, bold in candidates:
        f = fit(scan, text, family, bold)
        if f is not None:
            out[(family, bold)] = f
    return out
