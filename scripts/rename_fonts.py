"""Give the bundled reconstruction fonts family names of their own.

Word uses an installed font in place of an embedded one with the same
name, and many Khmer PCs have old builds installed as "Moul" and "Noto Sans
Khmer" (no Latin letters, other metrics, a "Bold" no wider than regular).
The original-layout .docx would then be drawn in fonts other than the ones
the scan was measured against. Under names no one has installed, Word has
to use the embedded copy.

Only the name table changes; outlines and metrics stay as they are, so
measuring gives the same result. The fonts are under the SIL OFL 1.1 with no
Reserved Font Name, which allows renamed modified versions.

Dev-only (needs fontTools); run once after replacing a bundled font:

    python scripts/rename_fonts.py
"""

import os
import sys

from fontTools.ttLib import TTFont

FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "netra_ocr", "reconstruct", "fonts")

# file -> new family name; keep in step with Family.name in netra_ocr/reconstruct/fonts.py
RENAME = {
    "Moul-Regular.ttf": "Netra Moul",
    "NotoSansKhmer-Regular.ttf": "Netra Khmer Sans",
    "NotoSansKhmer-Bold.ttf": "Netra Khmer Sans",
    "NotoSerifKhmer-Regular.ttf": "Netra Khmer Serif",
    "NotoSerifKhmer-Bold.ttf": "Netra Khmer Serif",
}
REPLACED = (1, 3, 4, 6, 16, 17, 21, 22, 25)   # every family / full / PostScript name


def rename(path: str, family: str) -> str:
    font = TTFont(path)
    name = font["name"]
    sub = name.getDebugName(2) or "Regular"            # Regular / Bold: unchanged
    old = name.getDebugName(4)
    version = (name.getDebugName(5) or "").replace("Version ", "").split(";")[0].strip()
    ps = family.replace(" ", "") + "-" + sub.replace(" ", "")
    name.names = [r for r in name.names if r.nameID not in REPLACED]
    for value, nid in ((family, 1), (f"{version};NETRA;{ps}", 3), (f"{family} {sub}", 4), (ps, 6)):
        name.setName(value, nid, 3, 1, 0x409)           # Windows, US English
        name.setName(value, nid, 1, 0, 0)               # Mac Roman
    font.save(path)
    return f"{os.path.basename(path)}: {old!r} -> {family} {sub!r} ({ps})"


def main() -> int:
    for file, family in RENAME.items():
        print(rename(os.path.join(FONT_DIR, file), family))
    return 0


if __name__ == "__main__":
    sys.exit(main())
