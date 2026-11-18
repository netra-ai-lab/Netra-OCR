"""What a text line looks like on the scan: its ink box, colour and weight.

The detector's line boxes are padded and overlap their neighbours (Khmer
stacks vowels above and subscripts below the base consonants), so size is
measured on the line's *core band*: the dense rows of base consonants,
which every Khmer and Latin line has and which never overlap the next line.
The same measurement is applied to synthetic renders in ``fonts.py``, so
whatever the core band is for a given font, scan and render agree on it.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ..imaging import otsu_threshold

Box = Sequence[int]


@dataclass
class Ink:
    x1: int            # horizontal ink extent (image px)
    x2: int
    top: int           # core band rows [top, bottom)
    bottom: int
    stroke: float      # mean horizontal ink run in the core band, / core height
    density: float     # ink fraction of the core band, x1..x2
    colour: Tuple[int, int, int] = (0, 0, 0)
    height: float = 0.0  # core band height to a fraction of a pixel

    @property
    def core(self) -> float:
        return self.height or float(self.bottom - self.top)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def centre(self) -> float:
        return (self.top + self.bottom) / 2


def core_band(ink: np.ndarray, weight_centre: bool = True,
              latin: bool = False) -> Optional[Tuple[int, int, float]]:
    """The densest band of ``ink`` (bool HxW): ``(top, bottom, height)``.

    ``[top, bottom)`` are whole rows for slicing; ``height`` is the band's
    height with sub-pixel edges (where the row profile crosses half the
    band's typical density, interpolated between rows). A Khmer line's core
    is only 6-10 px tall on a typical phone photo, so one row either way
    would be a 10-15% error in the type size.

    The search favours the middle of the box, since the line itself sits
    there and neighbours only intrude at the edges.
    """
    raw = ink.mean(axis=1)
    h = len(raw)
    if h == 0 or raw.max() <= 0:
        return None
    k = max(1, h // 12)
    prof = np.convolve(raw, np.ones(k) / k, mode="same") if k > 1 else raw
    if weight_centre and h > 2:
        c = (h - 1) / 2
        prof_w = prof * (1 - 0.6 * np.abs(np.arange(h) - c) / c)
    else:
        prof_w = prof
    peak = int(np.argmax(prof_w))

    def expand(level):
        t = b = peak
        while t > 0 and prof[t - 1] >= level:
            t -= 1
        while b < h - 1 and prof[b + 1] >= level:
            b += 1
        return t, b

    top, bottom = expand(0.45 * prof[peak])
    # Latin and digits have no dense band of base consonants: the densest
    # rows can be one stroke (the foot of a "2", a hyphen). A band that thin
    # is widened to the letters' height. (Khmer always has the band.)
    extent = int((raw >= 0.05 * raw.max()).sum())
    if latin and bottom + 1 - top < 0.25 * extent:
        top, bottom = expand(0.15 * prof[peak])

    # Sub-pixel edges on the unsmoothed profile.
    lo, hi = max(0, top - 1), min(h, bottom + 2)
    half = 0.5 * float(np.median(raw[top:bottom + 1]))
    rows = np.nonzero(raw[lo:hi] >= half)[0]
    if len(rows) == 0 or half <= 0:
        return top, bottom + 1, float(bottom + 1 - top)
    t, b = lo + int(rows[0]), lo + int(rows[-1])

    def frac(i, j):        # how far past row i's centre, toward row j, the level is crossed
        if j < 0 or j >= h or raw[i] == raw[j]:
            return 0.5
        return float(np.clip((raw[i] - half) / (raw[i] - raw[j]), 0.0, 1.0))

    top_f = t + 0.5 - frac(t, t - 1)
    bottom_f = b + 0.5 + frac(b, b + 1)
    return t, b + 1, max(0.5, bottom_f - top_f)


def measure(gray: np.ndarray, threshold: Optional[int] = None,
            weight_centre: bool = True, latin: bool = False) -> Optional[Ink]:
    """Measure ink in a grey crop (ink is dark). Coordinates are crop-relative."""
    if gray.size == 0:
        return None
    thr = otsu_threshold(gray) if threshold is None else threshold
    ink = gray < thr
    if ink.mean() < 0.003:
        return None
    band = core_band(ink, weight_centre, latin)
    if band is None:
        return None
    top, bottom, height = band
    core = ink[top:bottom]
    cols = np.nonzero(core.any(axis=0))[0]
    if len(cols) == 0:
        return None
    x1, x2 = int(cols[0]), int(cols[-1]) + 1
    core = core[:, x1:x2]
    edges = np.diff(np.pad(core, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    runs = int((edges == 1).sum())
    stroke = (core.sum() / runs) / height if runs else 0.0
    return Ink(x1=x1, x2=x2, top=top, bottom=bottom, stroke=float(stroke),
               density=float(core.mean()), height=height)


def measure_line(rgb: np.ndarray, gray: np.ndarray, bbox: Box,
                 page_threshold: int, latin: bool = False) -> Optional[Ink]:
    """Measure one detected line on the page. Returns page coordinates.
    ``latin``: the line is mostly not Khmer (see ``core_band``)."""
    h, w = gray.shape
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = gray[y1:y2, x1:x2]
    # Local Otsu separates ink from paper well inside one line; the page
    # threshold guards against lines that are almost all ink or all paper.
    thr = otsu_threshold(crop)
    thr = int(min(max(thr, page_threshold - 40), page_threshold + 40))

    # The line's own ink colour: the darker half of its ink pixels (stroke
    # centres, not anti-aliased edges). A seal or signature in another
    # colour crossing the line is then ignored, so it doesn't make the
    # text look bolder, wider or taller than it is.
    mask = crop < thr
    if mask.sum() < 4:
        return None
    vals = crop[mask]
    dark = mask & (crop <= np.median(vals))
    colour = np.median(rgb[y1:y2, x1:x2][dark], axis=0)
    px = rgb[y1:y2, x1:x2].astype(np.float32)
    # Compare hue, not lightness: chroma vectors (colour minus its grey).
    chroma = px - px.mean(axis=2, keepdims=True)
    ink_chroma = colour - colour.mean()
    foreign = mask & (np.abs(chroma - ink_chroma).max(axis=2) > 45)
    if foreign.any() and foreign.sum() < 0.6 * mask.sum():
        crop = crop.copy()
        crop[foreign] = 255

    ink = measure(crop, thr, latin=latin)
    if ink is None:
        return None
    ink.colour = tuple(int(v) for v in colour)
    ink.x1 += x1; ink.x2 += x1; ink.top += y1; ink.bottom += y1
    return ink


def is_latin(text: str) -> bool:
    """Mostly not Khmer: Latin letters, digits, punctuation."""
    letters = [ch for ch in text if not ch.isspace()]
    khmer = sum("\u1780" <= ch <= "\u17ff" or "\u19e0" <= ch <= "\u19ff" for ch in letters)
    return bool(letters) and khmer < 0.5 * len(letters)


def to_hex(rgb: Tuple[int, int, int]) -> str:
    return "%02x%02x%02x" % tuple(int(max(0, min(255, v))) for v in rgb)


def is_neutral(rgb: Tuple[int, int, int]) -> bool:
    """Grey or black ink: printed black on a scan comes out dark grey."""
    r, g, b = (int(v) for v in rgb)
    hi, chroma = max(r, g, b), max(r, g, b) - min(r, g, b)
    # Saturation relative to brightness: navy ink (48, 48, 80) is blue,
    # though its channels differ by only 32.
    return chroma < 18 or chroma < 0.25 * hi


def text_colour(samples: Sequence[Tuple[int, int, int]]) -> str:
    """One colour for a block's lines, as hex. Neutral ink becomes black;
    coloured ink is kept, more saturated, since scans wash colours out."""
    if not samples:
        return "000000"
    rgb = tuple(int(v) for v in np.median(np.asarray(samples), axis=0))
    if is_neutral(rgb):
        return "000000"
    arr = np.asarray(rgb, dtype=np.float32)
    # Boost saturation around the colour's own grey level, keeping its
    # lightness: the scan's blue is the printed blue mixed with paper.
    grey = arr.mean()
    arr = grey + (arr - grey) * 1.6
    return to_hex(tuple(np.clip(arr, 0, 255)))


def snap_colours(colours: Sequence[str], distance: float = 60.0) -> list:
    """Merge near-identical non-black colours on a page, so one blue heading
    style doesn't come out as five slightly different blues."""
    def rgb(hexs):
        return np.array([int(hexs[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)

    centres: list = []
    out = []
    for c in colours:
        if c == "000000":
            out.append(c)
            continue
        v = rgb(c)
        for centre in centres:
            if np.linalg.norm(rgb(centre) - v) < distance:
                out.append(centre)
                break
        else:
            centres.append(c)
            out.append(c)
    return out
