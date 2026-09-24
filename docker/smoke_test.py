"""Build-time check that the (pruned) image still runs everything the app does.

The Dockerfile uninstalls packages that netra-ocr's dependencies declare but
inference never imports (plotting, dataframes, training augmentation). This
script exercises every code path the server can reach -- all three line
detectors, layout analysis, both decoders, PDF input, every export format and
the web app itself -- so a prune that goes too far fails the build instead of
the first upload.
"""

import io

import cv2
import numpy as np
from PIL import Image

from netra_ocr.exporters import FORMATS, export
from netra_ocr.ocr_engine import KhmerOCRPipeline

page = np.full((900, 700, 3), 255, np.uint8)
cv2.putText(page, "Netra OCR", (220, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
for i, y in enumerate(range(170, 700, 45)):
    cv2.putText(page, f"Line {i + 1}: the quick brown fox jumps", (60, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
png = cv2.imencode(".png", page)[1].tobytes()
pdf = io.BytesIO()
Image.fromarray(page).save(pdf, "PDF")

for detector, layout, decoder in [("yolo", True, "ar"), ("yolo", False, "blockwise"),
                                  ("legacy", True, "ar"), ("tesseract", True, "ar")]:
    pipe = KhmerOCRPipeline(detector=detector, layout=layout, decoder=decoder)
    doc = pipe.process_document(png)
    assert doc.blocks, f"{detector}/{decoder}: no blocks"
    print(f"{detector:9s} layout={layout!s:5s} {decoder:9s} -> {len(doc.blocks)} blocks")

doc = KhmerOCRPipeline(layout=True).process_document(pdf.getvalue())
for fmt in FORMATS:
    data, name, _ = export(doc, fmt, "smoke")
    assert data, fmt
    print(f"export {fmt}: {len(data)} bytes")

from netra_ocr.server.app import create_app  # noqa: E402

create_app()
print("smoke test passed")
