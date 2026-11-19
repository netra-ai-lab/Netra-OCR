"""Build-time check that the (pruned) image still runs everything the app does.

This script exercises every code path the server can reach -- the line
detector with and without layout analysis, both decoders, PDF input, every
export format, page measuring for the editor and the original-layout Word
file, and the web app itself -- so a prune that goes too far fails the
build instead of the first upload. It also checks that OpenCV and SciPy stay
out of the image: the optional legacy and Tesseract detectors aren't in it.
"""

import importlib.util
import io
import os

from PIL import Image, ImageDraw, ImageFont

from netra_ocr.detectors import available
from netra_ocr.exporters import FORMATS, export
from netra_ocr.ocr_engine import KhmerOCRPipeline

for module in ("cv2", "scipy", "torch"):
    assert importlib.util.find_spec(module) is None, f"{module} is installed in the image"
assert available() == ["yolo"], available()

page = Image.new("RGB", (700, 900), "white")
draw = ImageDraw.Draw(page)
draw.text((220, 60), "Netra OCR", fill="black", font=ImageFont.load_default(size=40))
for i, y in enumerate(range(150, 700, 45)):
    draw.text((60, y), f"Line {i + 1}: the quick brown fox jumps", fill="black",
              font=ImageFont.load_default(size=24))
png = io.BytesIO()
page.save(png, "PNG")
pdf = io.BytesIO()
page.save(pdf, "PDF")

for layout, decoder in [(True, "ar"), (False, "blockwise")]:
    pipe = KhmerOCRPipeline(layout=layout, decoder=decoder)
    doc = pipe.process_document(png.getvalue())
    assert doc.blocks, f"layout={layout}/{decoder}: no blocks"
    print(f"yolo layout={layout!s:5s} {decoder:9s} -> {len(doc.blocks)} blocks")

doc = KhmerOCRPipeline(layout=True).process_document(pdf.getvalue())
for fmt in FORMATS:
    data, name, _ = export(doc, fmt, "smoke")
    assert data, fmt
    print(f"export {fmt}: {len(data)} bytes")

# The web app's pipeline also measures each page (fonts, sizes, pictures) so
# the editor and the original-layout Word file share one layout.
from netra_ocr.reconstruct import fonts as F, plan as PL  # noqa: E402

doc = KhmerOCRPipeline(layout=True, reconstruct=True).process_document(png.getvalue())
assert doc.pages[0].get("size_pt") and any(b.get("style") for b in doc.blocks), "page not measured"
layout = PL.to_dict(PL.build_document(doc)[0])
assert layout["bands"], "empty layout"
data, _, _ = export(doc, "docx", "smoke", layout="original")
print(f"measured: {len(layout['bands'])} bands, original-layout docx {len(data)} bytes, "
      f"text shaping {'on' if F.HAS_SHAPING else 'OFF'}")

from netra_ocr.server.app import create_app  # noqa: E402

app = create_app()
paths = {getattr(r, "path", "") for r in app.routes}
assert {"/v1/jobs/{job_id}/layout", "/docfonts", "/static"} <= paths, paths
for name in ("Moul-Regular.ttf", "NotoSansKhmer-Regular.ttf", "NotoSerifKhmer-Bold.ttf"):
    assert os.path.exists(os.path.join(F.FONT_DIR, name)), f"editor font missing: {name}"
print("smoke test passed")
