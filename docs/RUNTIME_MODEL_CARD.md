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
- netra-runtime
- netra-lab
---

<p align="center">
  <img src="https://huggingface.co/Darayut/khmer-text-recognition/resolve/main/assets/header.png" width="260" alt="Netra Lab" />
</p>

<hr>

<p align="center">
  <a href="https://github.com/netra-ai-lab/Netra-OCR"><b>GitHub</b></a> |
  <a href="https://github.com/netra-ai-lab/Netra-Runtime"><b>Netra Runtime</b></a> |
  <a href="https://pypi.org/project/netra-ocr/"><b>PyPI</b></a> |
  <a href="https://huggingface.co/Darayut/khmer-text-recognition"><b>PyTorch weights</b></a>
</p>

# Netra OCR

The models behind the Netra OCR app, exported as [Netra Runtime](https://github.com/netra-ai-lab/Netra-Runtime) bundles: ONNX graphs plus a `netra-model.json` manifest that says how to pre-process, decode and post-process. The same bundles run on CPU and CUDA from Python and in the browser (WebGPU / WebAssembly), with no PyTorch.

Each folder is one bundle:

| Folder | Task | What it is | Variants |
|---|---|---|---|
| `recognition/` | text recognition | Khmer + English line recognizer, 19M params, greedy decoding with a KV cache | – |
| `recognition-blockwise/` | text recognition | the same recognizer plus a blockwise proposal head (same output, fewer decoder passes) | – |
| `lines/` | object detection | YOLO text-line and logo detector | `fp16` (CUDA default) |
| `layout/` | object detection | DocLayout-YOLO (DocStructBench) page-layout analyzer | `fp16` (CUDA default) |

## Usage

```bash
pip install "netra-runtime[cpu]"     # or [gpu] for CUDA
```

```python
import netra_runtime

recognizer = netra_runtime.load("hf://Netra-Lab/Netra-OCR/recognition", device="auto")
print(recognizer("line.png").text)

detector = netra_runtime.load("hf://Netra-Lab/Netra-OCR/lines")
for box in detector("page.png"):
    print(box.label, box.score, box.box)
```

```js
import { load } from "@netra-ai-lab/runtime-web";
const recognizer = await load("hf://Netra-Lab/Netra-OCR/recognition");
const { text } = await recognizer.recognize(imageElement);
```

`hf://` downloads only the folder you ask for, once, into the Hugging Face cache. For whole pages (detection, layout, reading order, Word/HTML/Markdown export) use the [`netra-ocr`](https://pypi.org/project/netra-ocr/) package or the Netra OCR Docker image, which load these bundles.

## Parity and speed

Measured against the PyTorch models on `test_set/khob_eval` (325 lines) and 10 test pages, on an i7-13700KF and an RTX 3060.

- **Recognizer:** text identical to PyTorch greedy decoding on 325/325 lines, on CPU and CUDA, single and batched. Per line: CPU 36.7 → about 13 ms, CUDA 23 → 12.8 ms.
- **Blockwise:** identical text, 2.32 tokens per decoder pass. Not faster than the KV-cached greedy decoder on CPU or CUDA; faster on WebGPU (64 vs 86 ms per line).
- **Full page pipeline:** CPU 2.18 → 1.14 s/page, GPU 1.24 → 0.70 s/page. 7 of 10 pages identical to the PyTorch pipeline, 0.43% characters different overall (letterbox resize is defined by the bundle spec rather than OpenCV, which moves about 1% of borderline boxes).
- **fp16:** default on CUDA for the detectors (DocLayout 65 → 39 ms). Not used on WebGPU, where it is broken in onnxruntime-web 1.31.
- **Browser:** WebAssembly gives the same text on 325/325 lines and detector boxes at IoU 1.0; WebGPU in Chrome gave the same text on 100/100 lines.

The recognizer's CER is the PyTorch model's: see [Darayut/khmer-text-recognition](https://huggingface.co/Darayut/khmer-text-recognition).

## Files

| File | Contents |
|---|---|
| `*/netra-model.json` | manifest: task, components, graphs, variants, sha256 of every file |
| `*/*.onnx` | opset-18 graphs (`encoder`/`decoder_step` for recognition, `model` for detection) |
| `recognition*/vocab.json` | tokenizer vocabulary (class names of the detectors are in the manifest) |

The bundle format is specified in [Netra-Runtime/spec/bundle.md](https://github.com/netra-ai-lab/Netra-Runtime/blob/main/spec/bundle.md).

## License and attribution

- **`recognition/`, `recognition-blockwise/`:** MIT. Trained by Netra Lab; the training data includes [SoyVitou/KhmerSynthetic1M](https://huggingface.co/datasets/SoyVitou/KhmerSynthetic1M), whose terms restrict use to research — check them before commercial use.
- **`lines/`:** AGPL-3.0, as a model trained with Ultralytics YOLO.
- **`layout/`:** Apache-2.0. Converted from [juliozhao/DocLayout-YOLO-DocStructBench](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench) by [opendatalab/DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO); please cite their paper.
