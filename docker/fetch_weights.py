"""Download Netra OCR's model bundles into /models/netra-ocr (NETRA_BUNDLES).

Run at image build time so the container works offline. The folder has one
Netra Runtime bundle per model (see netra_ocr/models.py), with only the graphs
the CPU runs (no fp16 copies).
"""

import shutil
import sys
from pathlib import Path

import netra_runtime

from netra_ocr.models import BUNDLES, MODEL_REPO

target = Path(sys.argv[1] if len(sys.argv) > 1 else "/models/netra-ocr")
for folder in BUNDLES.values():
    # Copied out of the Hugging Face cache, whose files are symlinks to blobs.
    shutil.copytree(netra_runtime.resolve(f"{MODEL_REPO}/{folder}", device="cpu"), target / folder)
    print(target / folder)
