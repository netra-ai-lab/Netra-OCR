# Third-party licenses

Netra OCR's own code is MIT-licensed (see `LICENSE`). Some optional components
it uses are not, and they matter when you **distribute** Netra OCR (for
example the Docker images) or **offer it as a network service**.

| Component | Used for | License | Installed by |
| :--- | :--- | :--- | :--- |
| Netra OCR text-line detector weights ([`lines/`](https://huggingface.co/Netra-Lab/Netra-OCR/tree/main/lines)) | Default `yolo` line and logo detector | AGPL-3.0 (trained with Ultralytics YOLO) | downloaded on first use; baked into Docker images |
| [DocLayout-YOLO DocStructBench weights](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench) ([`layout/`](https://huggingface.co/Netra-Lab/Netra-OCR/tree/main/layout)) | Page layout analysis | Apache-2.0 (per its model card) | downloaded on first use; baked into Docker images |
| [Netra Runtime](https://github.com/netra-ai-lab/Netra-Runtime) | Runs every model | MIT | core |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Inference engine under Netra Runtime | MIT | `cpu` / `gpu` extras; Docker images |
| [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) / PDFium | PDF rendering | Apache-2.0 / BSD-3-Clause | `pdf`, `server` extras |
| [FastAPI](https://github.com/fastapi/fastapi), [Uvicorn](https://github.com/encode/uvicorn) | Web server | MIT / BSD-3-Clause | `server` extra |
| [python-docx](https://github.com/python-openxml/python-docx) | Word export | MIT | `docx`, `server` extras |
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | Optional line detector | Apache-2.0 | `tesseract` extra; Docker images |
| [Kantumruy Pro](https://fonts.google.com/specimen/Kantumruy+Pro) | UI typeface | SIL OFL 1.1 | bundled in the web UI |
| [ultralytics](https://github.com/ultralytics/ultralytics), [doclayout-yolo](https://github.com/opendatalab/DocLayout-YOLO) | Training and exporting the detectors | AGPL-3.0 | `research` extra only |
| [PyTorch](https://github.com/pytorch/pytorch) | Training and exporting | BSD-3-Clause | `research` extra only |

## What AGPL-3.0 means here

The app itself runs no AGPL code, but the default `yolo` detector's weights
are AGPL-3.0, and the Docker images include them. If you distribute Netra OCR
with those weights, or let people use a modified version over a network, you
must make the corresponding source available to them under the AGPL. Running
it unmodified for yourself or your organisation carries no extra obligation.
The recognizer on its own (`netra_ocr.recognition`), layout analysis, and the
`legacy` and `tesseract` detectors don't involve AGPL terms. The `research`
extra installs `ultralytics` and `doclayout-yolo`, which are AGPL-3.0 code.

## Training data

The recognition weights were trained partly on
[SoyVitou/KhmerSynthetic1M](https://huggingface.co/datasets/SoyVitou/KhmerSynthetic1M),
which is licensed for research/academic use only. Check that license before
any commercial use of the published weights.
