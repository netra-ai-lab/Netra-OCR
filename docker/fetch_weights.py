"""Download every model the app needs into the Hugging Face cache (HF_HOME).

Run at image build time so the container works offline (HF_HUB_OFFLINE=1).
The YOLO text-line detector is bundled in the wheel and needs no download.
"""

from huggingface_hub import hf_hub_download

from netra_ocr.layout.doclayout import DEFAULT_FILENAME as LAYOUT_FILE, DEFAULT_REPO as LAYOUT_REPO
from netra_ocr.recognition.recognize_text import DECODER_FILENAMES, DEFAULT_MODEL_REPO

for filename in DECODER_FILENAMES.values():
    print("recognizer:", hf_hub_download(DEFAULT_MODEL_REPO, filename))
print("layout:", hf_hub_download(LAYOUT_REPO, LAYOUT_FILE))
