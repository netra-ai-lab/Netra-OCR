"""Download Netra OCR's models into /models/netra-ocr (NETRA_BUNDLES).

Run at image build time so the container works offline. The folder has one
subfolder per model (see netra_ocr/models.py), with only the graphs the CPU
runs (no fp16 copies).
"""

import shutil
import sys
from pathlib import Path

from netra_ocr.inference import hub
from netra_ocr.models import BUNDLES, MODEL_REPO

target = Path(sys.argv[1] if len(sys.argv) > 1 else "/models/netra-ocr")
for folder in BUNDLES.values():
    # Copied out of the Hugging Face cache, whose files are symlinks to blobs.
    shutil.copytree(hub.fetch(f"{MODEL_REPO}/{folder}", "cpu"), target / folder)
    print(target / folder)
