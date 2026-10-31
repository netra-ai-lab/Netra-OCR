# Usage reference

Everything the README's Usage section leaves out: the recognizer on its own, the full pipeline, detectors, decoders, models and devices, output formats, every option, the web app's settings and the standalone Hugging Face script.

## 1. Recognizer only

If you already have line crops from your own detector, layout engine or manual cropping, you don't need Netra-OCR's detectors. The recognizer takes an image (a path or a PIL `Image`, cropped to one text line) and returns the text.

```python
from netra_ocr.recognition import recognize, recognize_batch

text = recognize("line_crop.png")                                  # one crop, path or PIL.Image
texts = recognize_batch(my_line_crops, batch_size=16)              # many crops; the model stays loaded

text = recognize("line_crop.png", decoder="blockwise")             # same text
text = recognize("line_crop.png", bundle="bundles/my-recognizer")  # your own exported model folder
```

Command line, one image, no detection step:

```bash
python -m netra_ocr.recognition.recognize_text --image line_crop.png --decoder blockwise
```

To bring your own detector, implement `netra_ocr.detectors.base.BaseTextDetector` (one method, `detect(image_path) -> list[DetectedLine]`) and pass its `DetectedLine.crop` values to `recognize_batch`. You don't need to register it in `DETECTOR_REGISTRY` or use `KhmerOCRPipeline` unless you want its output formatting and CLI.

## 2. Full pipeline

`KhmerOCRPipeline` detects text lines (and optionally logos) in a page image with one of the bundled detectors, recognises each crop, and writes the result in the format you choose.

```python
from netra_ocr.ocr_engine import KhmerOCRPipeline

pipeline = KhmerOCRPipeline(detector="yolo", conf=0.4, decoder="blockwise")   # create once, reuse
result_text = pipeline.process_image(
    image_path="document.png",
    output_path="document.docx",   # the extension picks the format
    batch_size=16,
    save_debug=False,
)
```

```bash
netra_ocr --image scan.jpg --output result.txt                              # YOLO detector, plain text
netra_ocr --image scan.jpg --output result.txt --detector legacy --pad 4    # classic CV, no extras needed
netra_ocr --image scan.jpg --output result.docx --detector yolo             # Word, logos embedded as images
netra_ocr --image scan.jpg --output result.json --detector yolo             # text + bounding box per line
netra_ocr --image scan.jpg --output result.txt --detector yolo --conf 0.4   # YOLO confidence threshold (default 0.25)
netra_ocr --image scan.jpg --output result.txt --detector tesseract --batch_size 32
netra_ocr --image scan.jpg --output result.txt --decoder blockwise          # same text
netra_ocr --image scan.jpg --output result.txt --detector yolo --debug      # per-line .txt and logo .png in debug_<name>/
```

### Structured documents (layout analysis)

```python
from netra_ocr.ocr_engine import KhmerOCRPipeline
from netra_ocr.exporters import save_document

pipeline = KhmerOCRPipeline(detector="yolo", layout=True)
doc = pipeline.process_document("letter.pdf")               # images, multi-page TIFF or PDF ("pdf" extra)
for block in doc.blocks:                                    # heading / paragraph / table / figure / caption / ...
    print(block["type"], block.get("text") or block.get("rows"))
save_document(doc, "letter.docx")                           # .docx .html .md .json .txt
```

Command line: `netra_ocr --image letter.pdf --output letter.docx --layout`.

### Output formats

The output format follows the file extension:
- **`.txt` / `.md`:** plain UTF-8 text, one line per detected text line.
- **`.docx`:** a Word document, with text lines as paragraphs and detected logos as inline images. Needs the `docx` extra.
- **`.json`:** image size plus each line's text and bounding box, enough to rebuild the layout.

### Detectors

| Detector | Requires | Description |
| :--- | :--- | :--- |
| `yolo` (default) | nothing extra | YOLOv26s trained on Khmer documents. Detects text lines (class 0) and logos (class 1). Logos are embedded in `.docx` output; other formats get text only. Text boxes are widened to cover the full line. The model (~38 MB) downloads on first use, like the others (see below). |
| `tesseract` | `pip install "netra-ocr[tesseract]"` + the system Tesseract binary | Tesseract plus graph clustering. No extra model. |
| `legacy` | nothing extra | Classic computer vision: MSER, gradient analysis and multi-channel binarisation. Accepts `pad`. |

### Decoders

Both decoders share one model and vocabulary, so switching never changes the text.

| Decoder | Model folder | Notes |
| :--- | :--- | :--- |
| `ar` (default) | `recognition/` | Greedy autoregressive decoding, one token per step, with a key/value cache so each step only runs the new token. |
| `blockwise` | `recognition-blockwise/` | [Stern et al. 2018](https://arxiv.org/abs/1811.03115) blockwise parallel decoding: proposes several tokens per step (2.3 on average) and verifies them against the same decoder, so the output equals `ar`. With the cached decoder, steps are already cheap, so it isn't faster on a CPU, a CUDA GPU or WebGPU. |

### Models and devices

Every model runs on [ONNX Runtime](https://onnxruntime.ai), through Netra OCR's own `netra_ocr/inference/`; PyTorch isn't needed. The models are ONNX model folders in the Hugging Face repo [Netra-Lab/Netra-OCR](https://huggingface.co/Netra-Lab/Netra-OCR), one folder each (`lines/`, `layout/`, `recognition/`, `recognition-blockwise/`). Each one downloads the first time it's used, only the files the device needs, into the Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`); after that, `HF_HUB_OFFLINE=1` runs without a network.

| Setting | Default | Meaning |
| :--- | :--- | :--- |
| `NETRA_DEVICE` | `auto` | `auto` uses an NVIDIA GPU when the `gpu` extra is installed and one is present, else the CPU. Or `cpu`, `cuda`, `cuda:1`, `tensorrt`, `openvino`. |
| `NETRA_BUNDLES` | *(unset)* | A local folder with the same four subfolders, used instead of Hugging Face (the Docker image does this). |

Install exactly one ONNX Runtime build: `pip install "netra-ocr[cpu]"` or `pip install "netra-ocr[gpu]"`. Their packages conflict, so don't install both.

### Post-processing

Raw decoder output is cleaned of common gibberish: control and replacement characters, and runaway repeats of characters, clusters or tokens from decoder loops. Machine-readable zone (MRZ) lines on passports and ID cards are detected and left alone, so legitimate `<` filler such as `IDKHM1011052875<<<<<<<<` survives.

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image_path` | `str` | required | The input image. |
| `detector` | `str` | `yolo` | `yolo`, `tesseract` or `legacy`. |
| `conf` | `float` | `0.25` | YOLO confidence threshold. Only for `detector="yolo"`. |
| `pad` | `int` | `None` (auto) | Pixels added around each box. For `yolo` and `legacy`. |
| `output_path` | `str` | `None` | Destination file; the extension picks `.txt`, `.md`, `.json` or `.docx`. |
| `decoder` | `str` | `ar` | `ar` or `blockwise` (same text, usually faster). |
| `batch_size` | `int` | `16` | Text lines per recognition batch. |
| `save_debug` | `bool` | `False` | Write per-segment `.txt` and logo `.png` files to `debug_<name>/`. |

## 3. Web app and REST API

With Docker, see the README's Quick start. Without Docker:

```bash
pip install "netra-ocr[server,cpu]"     # or [server,gpu]
netra_ocr serve            # http://localhost:8000; add --host 0.0.0.0 to allow other machines
```

What it does:
- **Input:** a scan, a phone photo or a multi-page PDF (JPG, PNG, TIFF, BMP, WebP, PDF; up to 50 MB and 100 pages by default).
- **Layout analysis:** [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) finds headings, paragraphs, tables, figures, captions, page headers and footers, and the reading order. Netra's line detector and recognizer then read the Khmer and English text.
- **Editable result:** a structured document, not a text dump. Fix misreads in the browser, with each block linked to its region on the page. Change a block's type, alignment and order, split or merge blocks, and edit table cells. Edits autosave, with undo.
- **Export:** a structured Word document (real heading styles, native tables, figures, captions, per-page headers and footers, Khmer set in Kantumruy Pro), plus HTML, Markdown, JSON (blocks and bounding boxes) and plain text.

REST API, with interactive docs at `http://localhost:8000/docs`:

```bash
curl -F file=@scan.jpg http://localhost:8000/v1/ocr                 # image: returns the finished document
curl -F file=@letter.pdf http://localhost:8000/v1/ocr               # PDF: returns a job to poll
curl http://localhost:8000/v1/jobs/<id>                             # status and page progress
curl -OJ "http://localhost:8000/v1/jobs/<id>/export?format=docx"    # docx | html | md | json | txt
```

| Setting | Default | Meaning |
| :--- | :--- | :--- |
| `NETRA_API_KEY` | *(unset)* | Require `X-API-Key` / `Authorization: Bearer` on `/v1/*`. Set it before exposing the port beyond your machine. |
| `NETRA_MAX_UPLOAD_MB` / `NETRA_MAX_PDF_PAGES` | `50` / `100` | Upload limits. |
| `NETRA_JOB_TTL_HOURS` | `24` | Documents and their edits are deleted this long after the last change. |
| `NETRA_DETECTOR` / `NETRA_DECODER` | `yolo` / `ar` | Defaults for new uploads. |
| `NETRA_DEVICE` / `NETRA_BUNDLES` | `auto` / *(unset)* | See [Models and devices](#models-and-devices). |

**Building the image yourself.** Run `docker compose up --build`. The build ends with a smoke test that runs every detector, both decoders, PDF input and every export format, so a broken image fails to build instead of failing on your first upload.

## 4. Without the package: the Hugging Face script

The original PyTorch checkpoints, with a standalone script (these need PyTorch and support beam search):

```bash
pip install torch torchvision transformers pillow huggingface_hub
wget https://huggingface.co/Darayut/khmer-text-recognition/resolve/main/configuration_khmerocr.py
wget https://huggingface.co/Darayut/khmer-text-recognition/resolve/main/inference.py
python inference.py --image "path/to/image.png" --method beam --beam_width 3
```

```python
from inference import KhmerOCR

ocr = KhmerOCR()                                             # downloads the weights
print(ocr.predict("test_image.jpg", method="beam", beam_width=3))
```
