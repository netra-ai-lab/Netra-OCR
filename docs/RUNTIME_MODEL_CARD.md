---
license: other
license_name: mixed
license_link: https://huggingface.co/Netra-Lab/Netra-OCR#license-and-attribution
language:
- km
- en
library_name: onnx
pipeline_tag: image-to-text
tags:
- ocr
- khmer-ocr
- document-layout
- onnx
- netra-lab
---

<p align="center">
  <img src="https://huggingface.co/Netra-Lab/Netra-OCR/resolve/main/assets/header.png" width="260" alt="Netra Lab" />
</p>

<hr>

<p align="center">
  <a href="https://github.com/netra-ai-lab/Netra-OCR"><b>GitHub</b></a> |
  <a href="https://pypi.org/project/netra-ocr/"><b>PyPI</b></a> |
  <a href="https://huggingface.co/Netra-Lab/Netra-OCR/tree/main/pytorch"><b>PyTorch weights</b></a>
</p>

# Netra OCR

The models behind the Netra OCR app, exported to ONNX. Each folder holds the ONNX graphs plus a `netra-model.json` config with the settings the app's pre- and post-processing need (line height and chunk width, decoder shape, class names, thresholds). The [`netra-ocr`](https://pypi.org/project/netra-ocr/) package runs them on ONNX Runtime on CPU and CUDA, and its [`web/`](https://github.com/netra-ai-lab/Netra-OCR/tree/master/web) folder runs the same files in the browser (WebGPU / WebAssembly), with no PyTorch.

Each folder is one model:

| Folder | Task | What it is | Download | Variants |
|---|---|---|---|---|
| `recognition/` | text recognition | Khmer + English line recognizer, 19M params, greedy decoding with a KV cache | 41 MB | – |
| `recognition-blockwise/` | text recognition | the same recognizer plus a blockwise proposal head (same output, fewer decoder passes) | 44 MB | – |
| `lines/` | object detection | YOLO text-line and logo detector | 38 MB (19 MB on CUDA) | `fp16` (CUDA default) |
| `layout/` | object detection | DocLayout-YOLO (DocStructBench) page-layout analyzer | 39 MB | `fp16` (CUDA default) |

A full page pipeline on the CPU downloads about 118 MB. The recognizers' and the layout model's weights are *stored* in half precision and cast back to float32 when the model loads, so they run in float32 at the same speed; only the weights' rounding changes. On 325 test lines, the text matches the float32 files on 325/325 lines on the CPU and 321/325 on CUDA (on the other 4, the number of errors against the labels is unchanged). The layout model gives identical pages. The line detector stays float32, because half-precision storage moved boxes on 2 of 10 test pages.

## Usage

```bash
pip install "netra-ocr[cpu]"     # or [gpu] for CUDA
```

```python
from netra_ocr.inference import Detector, Recognizer

recognizer = Recognizer("hf://Netra-Lab/Netra-OCR/recognition")   # device: "auto" (default), "cpu", "cuda"
print(recognizer("line.png").text)

detector = Detector("hf://Netra-Lab/Netra-OCR/lines")
for box in detector("page.png"):
    print(box.label, box.score, box.box)
```

```js
// web/src/index.js in the Netra-OCR repo
import { loadRecognizer } from "./web/src/index.js";
const recognizer = await loadRecognizer("hf://Netra-Lab/Netra-OCR/recognition");
const { text } = await recognizer.recognize(imageElement);
```

`hf://` downloads only the folder you ask for, and only the files your device runs, once, into the Hugging Face cache. For whole pages (detection, layout, reading order, Word/HTML/Markdown export) use `netra_ocr.KhmerOCRPipeline`, the `netra_ocr serve` web app or the Netra OCR Docker image.

## Parity and speed

Measured against the PyTorch models on `test_set/khob_eval` (325 lines) and 10 test pages, on an i7-13700KF and an RTX 3060.

- **Recognizer:** text identical to PyTorch greedy decoding on 325/325 lines, on CPU and CUDA, single and batched. Per line: CPU 36.7 → about 13 ms, CUDA 23 → 12.8 ms.
- **Blockwise:** identical text, 2.32 tokens per decoder pass. Not faster than the KV-cached greedy decoder on CPU, CUDA or WebGPU (82 vs 85 ms per line in Chrome).
- **Full page pipeline:** CPU 2.18 → 1.14 s/page, GPU 1.24 → 0.70 s/page. 7 of 10 pages identical to the PyTorch pipeline, 0.43% characters different overall (the letterbox resize is Netra OCR's own float64 bilinear resize rather than OpenCV's, which moves about 1% of borderline boxes).
- **fp16:** default on CUDA for the detectors (DocLayout 65 → 39 ms). Not used on WebGPU, where it is broken in onnxruntime-web 1.31.
- **Browser:** WebAssembly gives the same text on 325/325 lines and detector boxes at IoU 1.0; WebGPU in Chrome gave the same text on 100/100 lines.

The recognizer's CER is the PyTorch model's: see the [results](https://github.com/netra-ai-lab/Netra-OCR/blob/master/docs/RESULTS.md).

## PyTorch checkpoints

`pytorch/` holds the recognizer the ONNX folders were exported from, for fine-tuning and research. The app doesn't download it.

| File | Contents |
|---|---|
| `pytorch/khmerocr_cluster_ar.pth` | the recognizer (cluster vocabulary, autoregressive decoder), exported as `recognition/` |
| `pytorch/khmerocr_cluster_blockwise.pth` | the same model plus the blockwise proposal head, exported as `recognition-blockwise/` |
| `pytorch/char2idx_cluster.json` | their vocabulary |

To fine-tune on your own text lines, from a clone of the [GitHub repo](https://github.com/netra-ai-lab/Netra-OCR):

```bash
bash training/setup.sh
bash training/finetune.sh my_lines/ --name my-model     # starts from pytorch/khmerocr_cluster_ar.pth
bash training/export.sh training/runs/my-model/best.pth
```

See [training/README.md](https://github.com/netra-ai-lab/Netra-OCR/blob/master/training/README.md).

## Files

| File | Contents |
|---|---|
| `*/netra-model.json` | config: pre/post-processing settings, graphs, precision variants, sha256 of every file |
| `*/*.onnx` | opset-18 graphs (`encoder`/`decoder_step` for recognition, `model` for detection) |
| `recognition*/vocab.json` | tokenizer vocabulary (class names of the detectors are in the config) |

The Python reader is [`netra_ocr/inference/`](https://github.com/netra-ai-lab/Netra-OCR/tree/master/netra_ocr/inference); the exporters are in [`training/netra_ocr_train/scripts/`](https://github.com/netra-ai-lab/Netra-OCR/tree/master/training/netra_ocr_train/scripts).

## License and attribution

- **`recognition/`, `recognition-blockwise/`, `pytorch/`:** MIT. Trained by Netra Lab; the training data includes [SoyVitou/KhmerSynthetic1M](https://huggingface.co/datasets/SoyVitou/KhmerSynthetic1M), whose terms restrict use to research — check them before commercial use.
- **`lines/`:** AGPL-3.0, as a model trained with Ultralytics YOLO.
- **`layout/`:** Apache-2.0. Converted from [juliozhao/DocLayout-YOLO-DocStructBench](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench) by [opendatalab/DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO); please cite their paper.
