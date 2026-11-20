---
license: other
license_name: mixed
license_link: https://huggingface.co/Netra-Lab/Netra-OCR#license-and-attribution
language:
- km
- en
library_name: onnx
pipeline_tag: image-to-text
datasets:
- Netra-Lab/khmer-document-synthetic-low-res
- Netra-Lab/khmer-scene-text-synthetic-contrast
- SoyVitou/KhmerSynthetic1M
- seanghay/khmer-hanuman-100k
metrics:
- cer
tags:
- ocr
- khmer-ocr
- text-recognition
- document-layout
- squeeze-and-excitation
- onnx
- netra-lab
---

<p align="center">
  <img src="https://huggingface.co/Netra-Lab/Netra-OCR/resolve/main/assets/header.png" width="260" alt="Netra Lab" />
</p>

<hr>

<p align="center">
  <a href="https://github.com/netra-ai-lab/Netra-OCR"><b>GitHub</b></a> |
  <a href="https://github.com/netra-ai-lab/Netra-OCR/pkgs/container/netra-ocr"><b>Docker Image</b></a> |
  <a href="https://pypi.org/project/netra-ocr/"><b>PyPI</b></a>
</p>

# Netra OCR

**Khmer and English scans, photos and PDFs in, editable documents out.** Netra-OCR is a **19M-parameter** text-line recognizer for Khmer and English, trained from scratch on 1.3M synthetic line images. A Squeeze-and-Excitation CNN and a Transformer encoder read each line in overlapping chunks, a bidirectional LSTM joins the chunks, and a Transformer decoder writes the text one Khmer character cluster at a time. It runs on the CPU.

## Results

Character error rate (%) on single text lines, lower is better. Netra-OCR uses beam search with width 3.

| Model | Params | [KHOB](https://huggingface.co/datasets/Netra-Lab/khob-eval) | Legal Documents | [Printed Words](https://huggingface.co/datasets/Netra-Lab/khmer-printed-words) |
|---|---:|---:|---:|---:|
| Tesseract 5 (`khm`) | – | 5.12 | 41.78 | 20.14 |
| Surya OCR | – | 14.52 | 51.16 | 28.45 |
| Qwen2.5-VL-3B (LoRA on Khmer) | 3B | 89.21 | 84.63 | 98.62 |
| DeepSeek-OCR (LoRA on Khmer) | 3B | 33.23 | 44.22 | 69.79 |
| **Netra-OCR** | **19M** | **0.96** | **5.31** | **2.95** |

- **Where it wins:** the lowest error on all three sets, including phone photos of legal documents (5.31% against 41.78% for Tesseract).
- **Where it's weaker:** hard photos and short isolated words cost accuracy (5.31% and 2.95% against 0.96% on KHOB), and English can come back with small errors.

The baselines' outputs are normalised the same way. The two vision-language models had a short LoRA fine-tune on 6,000 of our lines, since neither documents Khmer support. Legal Documents is not public, because the documents belong to real people.

<p align="center"><img src="https://huggingface.co/Netra-Lab/Netra-OCR/resolve/main/assets/results.png" width="100%" alt="Character error rate of Tesseract, Surya OCR, Qwen2.5-VL, DeepSeek-OCR and Netra-OCR on KHOB, legal documents and printed words"></p>

The full benchmark, with word error rate, exact match and the zero-shot vision-language models, is in [docs/RESULTS.md](https://github.com/netra-ai-lab/Netra-OCR/blob/master/docs/RESULTS.md).

## Architecture

<p align="center"><img src="https://huggingface.co/Netra-Lab/Netra-OCR/resolve/main/assets/ocr-architecture.png" width="100%" alt="Netra-OCR architecture: chunked text line, SE network, patch embedding, Transformer encoder, merger, BiLSTM, Transformer decoder"></p>

*A text line is chunked, each chunk is encoded in parallel, and the vision tokens are merged into one sequence. A bidirectional LSTM smooths the chunk seams, and the Transformer decoder writes the text one character cluster at a time.*

- **Chunking:** each line is resized to 48 px high and split into 48 × 100 px chunks that overlap by 16 px, so long lines are never squeezed.
- **Encoder:** a Squeeze-and-Excitation CNN (5 VGG-style blocks, SE over height only in blocks 3–5), a patch module and a 2-layer Transformer encoder turn each chunk into 384-dimensional tokens.
- **Across chunks:** the tokens of all chunks are concatenated, and a one-layer bidirectional LSTM carries context across each seam.
- **Decoder:** a 2-layer Transformer decoder over a 2,101-token vocabulary. Khmer is written as character clusters (a consonant with its subscripts and vowels), which makes Khmer sequences about 43% shorter.

### Blockwise parallel decoding

<p align="center"><img src="https://huggingface.co/Netra-Lab/Netra-OCR/resolve/main/assets/blockwise_decoding.png" width="90%" alt="One step of blockwise parallel decoding: the head guesses 4 tokens, the decoder agrees with the first two and differs at the third, so the two matches plus the decoder's token are kept"></p>

*One step of blockwise parallel decoding ([Stern et al., 2018](https://arxiv.org/abs/1811.03115)). A small proposal head guesses the next 4 tokens; the decoder checks all of them in one parallel pass, keeps the guesses it agrees with, and adds its own token where it first disagrees. Every kept token is one greedy decoding would also write, so the output is identical.*

On KHOB (325 lines) in PyTorch:

| Decoder | CER | ms per line | Tokens per step | Speed-up |
|---|---:|---:|---:|---:|
| Autoregressive | 1.17 | 17.60 | 1 | 1.00× |
| **Blockwise, 4 proposed** | **1.17** | **11.35** | **2.7** | **1.55×** |

The ONNX models in this repo cache the decoder's keys and values, which makes each pass cheap, so there both decoders run at about the same speed (see [Parity and speed](#parity-and-speed)).

## Try the app

The Docker image bundles every model in this repo with a browser editor. It rebuilds each page as it looked (pictures, seals, columns, alignment and font sizes), lets you correct any block, and exports Word, HTML, Markdown, JSON or plain text. Files never leave your machine.

```bash
docker run -d --name netra-ocr -p 8000:8000 -v netra-data:/data ghcr.io/netra-ai-lab/netra-ocr
# then open http://localhost:8000
```

<p align="center"><img src="https://huggingface.co/Netra-Lab/Netra-OCR/resolve/main/assets/editor.png" width="100%" alt="The Netra OCR web app: a scanned Khmer press release with its detected blocks outlined on the left, and the editable page rebuilt from it on the right"></p>

*A scanned Khmer press release (left, detected blocks outlined) and the editable page the app rebuilt from it (right), in version 1.3.1 on the CPU.*

## Models in this repo

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

## Training data and test sets

The recognizer was trained only on synthetic lines:
- [Netra-Lab/khmer-document-synthetic-low-res](https://huggingface.co/datasets/Netra-Lab/khmer-document-synthetic-low-res): 100,000 document-style lines, made for this project;
- [Netra-Lab/khmer-scene-text-synthetic-contrast](https://huggingface.co/datasets/Netra-Lab/khmer-scene-text-synthetic-contrast): 100,000 scene-text lines on real backgrounds, made for this project;
- [SoyVitou/KhmerSynthetic1M](https://huggingface.co/datasets/SoyVitou/KhmerSynthetic1M): 1,000,000 lines;
- [seanghay/khmer-hanuman-100k](https://huggingface.co/datasets/seanghay/khmer-hanuman-100k): 100,000 lines.

It was tested on [Netra-Lab/khob-eval](https://huggingface.co/datasets/Netra-Lab/khob-eval) (325 real lines, cropped from the [Khmer OCR Benchmark](https://github.com/EKYCSolutions/khmer-ocr-benchmark-dataset) by EKYC Solutions) and [Netra-Lab/khmer-printed-words](https://huggingface.co/datasets/Netra-Lab/khmer-printed-words) (short words in 10 fonts).

## Learn more

| | |
|---|---|
| [README](https://github.com/netra-ai-lab/Netra-OCR#readme) | The project overview: quick start, install, usage and limitations |
| [Usage](https://github.com/netra-ai-lab/Netra-OCR/blob/master/docs/USAGE.md) | Every detector, decoder, output format and option, and the REST API |
| [Architecture](https://github.com/netra-ai-lab/Netra-OCR/blob/master/docs/ARCHITECTURE.md) | Chunking, the SE network, the BiLSTM, decoding and tokenisation |
| [Results](https://github.com/netra-ai-lab/Netra-OCR/blob/master/docs/RESULTS.md) | The full benchmark and the decoder comparison |
| [Training](https://github.com/netra-ai-lab/Netra-OCR/blob/master/training/README.md) | Fine-tuning, training, evaluating and exporting the recognizer |
| [Paper](https://github.com/netra-ai-lab/Netra-OCR/blob/master/docs/paper/A%20Squeeze-and-Excitation%20Transformer%20Network%20for%20Khmer%20Optical%20Character%20Recognition.pdf) | A Squeeze-and-Excitation Transformer Network for Khmer Optical Character Recognition |
| [Issues](https://github.com/netra-ai-lab/Netra-OCR/issues) | Bug reports and questions |

## License and attribution

- **`recognition/`, `recognition-blockwise/`, `pytorch/`:** MIT. Trained by Netra Lab; the training data includes [SoyVitou/KhmerSynthetic1M](https://huggingface.co/datasets/SoyVitou/KhmerSynthetic1M), whose terms restrict use to research — check them before commercial use.
- **`lines/`:** AGPL-3.0, as a model trained with Ultralytics YOLO.
- **`layout/`:** Apache-2.0. Converted from [juliozhao/DocLayout-YOLO-DocStructBench](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench) by [opendatalab/DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO); please cite their paper.
