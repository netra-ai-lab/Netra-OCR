"""Measure ``Family.system_fit``: how the standard Khmer OS fonts named in
Word output compare with the bundled fonts the scan is measured against.

For each bundled face and many real Khmer lines (tests/golden), render the
line in both fonts at the same em size and measure it the way a scanned line
is measured (``reconstruct.style.measure``):

* size factor  = bundled core-band height / Khmer OS core-band height, the
  point size that makes the Khmer OS letters as tall as the scan's;
* width factor = the character scale that, at that size, makes the line as
  wide as the bundled font sets it.

Word fakes bold for Khmer OS Siemreap (it has no bold face), which doesn't
change advance widths, so bold faces are compared against its regular.

Needs the Khmer OS fonts (Windows: C:/Windows/Fonts). Prints the values to
paste into netra_ocr/reconstruct/fonts.py:

    python scripts/calibrate_system_fonts.py [--font-dir C:/Windows/Fonts]
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from netra_ocr.reconstruct import fonts as F          # noqa: E402
from netra_ocr.reconstruct.style import measure       # noqa: E402

FILES = {"Khmer OS Siemreap": "KhmerOSsiemreap.ttf", "Khmer OS Muol": "KhmerOSmuol.ttf"}
PX = 64


def khmer_lines():
    with open(os.path.join(ROOT, "tests", "golden", "pages_cpu.json"), encoding="utf-8") as fh:
        pages = json.load(fh)
    lines = [ln.strip() for text in pages.values() for ln in text.splitlines()]
    return [ln for ln in lines
            if len(ln) >= 12 and sum(map(F._is_khmer, ln)) > 0.8 * len(ln.replace(" ", ""))]


def render(text, path):
    font = F._font(path, PX)
    img = Image.new("L", (int(font.getlength(text)) + PX, PX * 3), 255)
    ImageDraw.Draw(img).text((PX // 2, PX * 2), text, font=font, fill=0, anchor="ls")
    return np.asarray(img), font.getlength(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--font-dir", default="C:/Windows/Fonts")
    args = ap.parse_args()
    if not F.HAS_SHAPING:
        sys.exit("Pillow needs libraqm to shape Khmer")
    lines = khmer_lines()
    print(f"{len(lines)} Khmer lines")
    for key, fam in F.FAMILIES.items():
        system = os.path.join(args.font_dir, FILES[fam.system_name])
        fits = []
        for bold in (False, True):
            size, width = [], []
            for text in lines:
                a, wa = render(text, fam.file(bold))
                b, wb = render(text, system)
                ma, mb = measure(a, threshold=128), measure(b, threshold=128)
                if ma and mb and ma.core and mb.core:
                    k = ma.core / mb.core
                    size.append(k)
                    width.append(wa / (wb * k))
            fits.append((round(float(np.median(size)), 3), round(float(np.median(width)), 3)))
        print(f'"{key}": {fam.system_name}: system_fit=({fits[0]}, {fits[1]})')


if __name__ == "__main__":
    main()
