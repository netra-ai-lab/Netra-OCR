# Third-party licenses

Netra OCR's own code is MIT-licensed (see `LICENSE`). Some optional components
it uses are not, and they matter when you **distribute** Netra OCR (for
example the Docker images) or **offer it as a network service**.

| Component | Used for | License | Installed by |
| :--- | :--- | :--- | :--- |
| [ultralytics](https://github.com/ultralytics/ultralytics) | YOLO text-line detector runtime | AGPL-3.0 | `yolo`, `server` extras; Docker images |
| [doclayout-yolo](https://github.com/opendatalab/DocLayout-YOLO) | Page layout analysis | AGPL-3.0 | `layout`, `server` extras; Docker images |
| [DocLayout-YOLO DocStructBench weights](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench) | Layout model weights | AGPL-3.0 (per upstream) | downloaded on first use; baked into Docker images |
| [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) / PDFium | PDF rendering | Apache-2.0 / BSD-3-Clause | `pdf`, `server` extras |
| [PyTorch](https://github.com/pytorch/pytorch) | Model runtime | BSD-3-Clause | core |
| [FastAPI](https://github.com/fastapi/fastapi), [Uvicorn](https://github.com/encode/uvicorn) | Web server | MIT / BSD-3-Clause | `server` extra |
| [python-docx](https://github.com/python-openxml/python-docx) | Word export | MIT | `docx`, `server` extras |
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | Optional line detector | Apache-2.0 | `tesseract` extra; Docker images |
| [Kantumruy Pro](https://fonts.google.com/specimen/Kantumruy+Pro) | UI typeface | SIL OFL 1.1 | bundled in the web UI |

## What AGPL-3.0 means here

The Docker images and the `server`/`layout`/`yolo` extras combine Netra OCR
with AGPL-3.0 code. If you distribute that combination, or let people use a
modified version over a network, you must make the corresponding source
available to them under the AGPL. Running it unmodified for yourself or your
organisation carries no extra obligation. The recognizer on its own
(`pip install netra-ocr`, `netra_ocr.recognition`) and the `legacy` detector
don't use any AGPL code.

## Training data

The recognition weights were trained partly on
[SoyVitou/KhmerSynthetic1M](https://huggingface.co/datasets/SoyVitou/KhmerSynthetic1M),
which is licensed for research/academic use only. Check that license before
any commercial use of the published weights.
