---
library_name: transformers
license: mit
datasets:
- Darayut/khmer-document-synthetic-low-res
- Darayut/khmer-scene-text-synthetic-contrast
- SoyVitou/KhmerSynthetic1M
- seanghay/khmer-hanuman-100k
language:
- km
- en
metrics:
- cer
pipeline_tag: image-to-text
tags:
- ocr
- khmer-ocr
- text-recognition
- khmer-text-recognition
- transformer
- squeeze-and-excitation
- netra-lab
---

<div align="center">
  <img src="https://huggingface.co/Darayut/khmer-text-recognition/resolve/main/assets/symbol.png" height="72" alt="Netra Lab" style="vertical-align: middle; margin-right: 24px;" />
  <img src="https://huggingface.co/Darayut/khmer-text-recognition/resolve/main/assets/wordmark-latin.png" height="56" alt="Netra" style="vertical-align: middle;" />
</div>

<hr>

<p align="center">
  <a href="https://github.com/netra-ai-lab/Netra-OCR"><b>GitHub</b></a> |
  <a href="https://github.com/netra-ai-lab/Netra-OCR/pkgs/container/netra-ocr"><b>Docker Image</b></a> |
  <a href="https://pypi.org/project/netra-ocr/"><b>PyPI</b></a> |
  <a href="https://huggingface.co/collections/Darayut/khmer-text-synthetic"><b>Datasets</b></a> |
  <a href="https://huggingface.co/spaces/Darayut/Khmer-Text-Recognition"><b>Inference Space</b></a>
</p>

# Netra-OCR

*A Squeeze-and-Excitation Transformer Network for Khmer Optical Character Recognition*

A **19M-parameter** text-line recognizer for **Khmer and English**, trained from scratch on **1.3M** synthetic text-line images. A Squeeze-and-Excitation network and a Transformer encoder read the image, and a Transformer decoder writes the text. Khmer is decoded as **character clusters**, which makes Khmer sequences about 43% shorter than one token per code point.

The model reads **one text line at a time**. For whole pages (scans, phone photos, PDFs), the [`netra-ocr`](https://pypi.org/project/netra-ocr/) package adds line detection, layout analysis and export to Word, HTML, Markdown and JSON, and the Netra OCR app runs it all from one Docker image.

**Code:** [github.com/netra-ai-lab/Netra-OCR](https://github.com/netra-ai-lab/Netra-OCR), with the package, the web app and the full benchmark.

## Results

Character error rate in %, on single text-line crops. Lower is better.

| Model | Params | KHOB | Legal Documents | Printed Words |
|---|---|---|---|---|
| Tesseract 5 (`khm`) | – | 5.12 | 41.78 | 20.14 |
| Surya OCR | – | 14.52 | 51.16 | 28.45 |
| Qwen2.5-VL-3B (LoRA on Khmer) | 3B | 89.21 | 84.63 | 98.62 |
| DeepSeek-OCR (LoRA on Khmer) | 3B | 33.23 | 44.22 | 69.79 |
| **Netra-OCR** | **19M** | **0.96** | **5.31** | **2.95** |

- **Where it wins:** the lowest error on all three sets; on KHOB 0.96% against 5.12% for Tesseract 5, the next best.
- **Where it is weakest:** phone photos of legal documents (5.31%) and short isolated words (2.95%), against 0.96% on KHOB.
- **Baselines:** Qwen2.5-VL and DeepSeek-OCR don't document Khmer support, so each had a short LoRA fine-tune (300 steps on 6,000 of our training lines). Netra-OCR uses beam search of width 3. Printed Words uses the first 500 of its 5,000 lines for every engine.

<p align="center"><img src="https://huggingface.co/Darayut/khmer-text-recognition/resolve/main/assets/results.png" width="100%" alt="Character error rate of Tesseract, Surya OCR, Qwen2.5-VL, DeepSeek-OCR and Netra-OCR on KHOB, legal documents and printed words"></p>

*Character error rate on the three test sets.* The full tables, with word error rate, exact match and the zero-shot models, are in [docs/RESULTS.md](https://github.com/netra-ai-lab/Netra-OCR/blob/master/docs/RESULTS.md).

### Decoders

The two released checkpoints share one model and differ only in decoding. The `blockwise` checkpoint adds a small proposal head ([Stern et al. 2018](https://arxiv.org/abs/1811.03115)) to the frozen model:

<p align="center"><img src="https://huggingface.co/Darayut/khmer-text-recognition/resolve/main/assets/blockwise_decoding.png" width="90%" alt="One step of blockwise parallel decoding: propose 4 tokens, verify them in one pass, keep the matching prefix plus the decoder's own next token"></p>

*One decoding step*, reading ព្រះរាជាណាចក្រកម្ពុជា (“Kingdom of Cambodia”) one character cluster per token. The head proposes 4 tokens, the model checks them in one parallel pass, and the proposals are kept up to the first disagreement, plus the model's own token there. Every kept token is one greedy decoding would write, so the output is identical.

On KHOB (325 lines, greedy decoding):

| Decoder | CER | WER | Exact match | ms per line | Tokens per step | Speed-up |
|---|---|---|---|---|---|---|
| `ar` (greedy) | 1.17% | 17.11% | 84.92% | 17.60 | 1 | 1.00× |
| `blockwise` | 1.17% | 17.11% | 84.92% | 11.35 | 2.7 | 1.55× |

Beam search (`beam_width=3`) lowers KHOB CER to 0.96% at a higher cost per line.

## Architecture

<p align="center"><img src="https://huggingface.co/Darayut/khmer-text-recognition/resolve/main/assets/ocr-architecture.png" width="100%" alt="Netra-OCR architecture: chunked text line, SE network, patch embedding, Transformer encoder, merger, BiLSTM, Transformer decoder"></p>

*A text line is chunked, each chunk is encoded in parallel, the vision tokens are merged and smoothed across the chunk seams by a BiLSTM, and the decoder writes the text one character cluster at a time.*

- **Input:** one text-line image of any width, resized to 48 px high and cut into 48 × 100 px chunks that overlap by 16 px.
- **Squeeze-and-Excitation network:** 5 VGG-style convolution blocks (64 → 512 channels) with height-only pooling in the later stages. Blocks 3–5 add a 1D SE module that averages over height only, so each horizontal position gets its own channel weights.
- **Patch module and Transformer encoder:** each feature-map column becomes a 384-dimensional token; a 2-layer, 8-head encoder lets the tokens of one chunk attend to each other.
- **Merging and BiLSTM:** the tokens of all chunks are concatenated with a line-level positional embedding, and a one-layer bidirectional LSTM carries context across each chunk seam.
- **Transformer decoder:** 2 layers, 8 heads, over a 2,101-token vocabulary of Khmer clusters, Latin characters, digits and punctuation.
- **Blockwise head** (`khmerocr_cluster_blockwise.pth` only): a small proposal head on the frozen model ([Stern et al. 2018](https://arxiv.org/abs/1811.03115)) that proposes up to 4 tokens per step and keeps only those the model itself would produce.

[docs/ARCHITECTURE.md](https://github.com/netra-ai-lab/Netra-OCR/blob/master/docs/ARCHITECTURE.md) explains each module.

## How it was made

1. **Tokenisation.** Latin text is tokenised by character. Khmer is tokenised as character clusters (a consonant with its subscripts and vowels, e.g. `ក្រ`), with single characters as the fallback. On a 35,000-line Khmer corpus this shortens sequences by 42.65%, with lossless round-tripping.
2. **Training.** Adam (learning rate 1e-4) with a staged cyclic schedule and cross-entropy loss, on 50,000 randomly sampled, augmented lines per epoch.
3. **Warm start.** The cluster-vocabulary model was warm-started from the earlier character-level model (vision layers and shared token embeddings transferred), then trained on the full 1.3M lines.
4. **Blockwise head.** The converged model was frozen and the proposal head trained on top of it.

**Training data** (synthetic, 1.3M lines):

| Dataset | Lines | Source | Augmentation |
|---|---|---|---|
| [khmer-document-synthetic-low-res](https://huggingface.co/datasets/Darayut/khmer-document-synthetic-low-res) | 100,000 | Pillow + a Khmer corpus, 11 fonts | Erosion, noise, thinning and thickening, perspective distortion |
| [khmer-scene-text-synthetic-contrast](https://huggingface.co/datasets/Darayut/khmer-scene-text-synthetic-contrast) | 102,500 | SynthTIGER + Stanford Background | Rotation, blur, noise, real backgrounds |
| [KhmerSynthetic1M](https://huggingface.co/datasets/SoyVitou/KhmerSynthetic1M) | 1,000,000 | SoyVitou | Applied by the source authors |
| [khmer-hanuman-100k](https://huggingface.co/datasets/seanghay/khmer-hanuman-100k) | 100,000 | seanghay | Applied by the source authors |

**Test sets:**

| Test set | Type | Lines | What it tests |
|---|---|---|---|
| [KHOB](https://github.com/EKYCSolutions/khmer-ocr-benchmark-dataset) | Real | 325 | The public Khmer OCR benchmark: clean backgrounds, compression artefacts |
| Legal Documents | Real, not public | 220 | Phone photos of birth certificates, diplomas and ID cards: degradation, lighting, distortion |
| Printed Words | Synthetic | 500 | Short isolated words in 10 fonts |

<p align="center"><img src="https://huggingface.co/Darayut/khmer-text-recognition/resolve/main/assets/dataset-overview.png" width="100%" alt="Example text lines from each training and test set"></p>

## Usage

```bash
pip install netra-ocr
```

```python
from netra_ocr.recognition import recognize, recognize_batch

text = recognize("line_crop.png")                         # one text line: a path or a PIL.Image
texts = recognize_batch(line_crops, batch_size=8)          # many lines; the model stays loaded
text = recognize("line_crop.png", decoder="blockwise")     # same text, 1.55x faster
text = recognize("line_crop.png", beam_width=3)            # slower, slightly more accurate
```

The checkpoint downloads from this repository on first use and is cached.

**Whole documents.**

```bash
pip install "netra-ocr[yolo,layout,pdf,docx]"
```

```python
from netra_ocr.ocr_engine import KhmerOCRPipeline
from netra_ocr.exporters import save_document

pipeline = KhmerOCRPipeline(detector="yolo", layout=True)
doc = pipeline.process_document("letter.pdf")     # images, multi-page TIFF or PDF
save_document(doc, "letter.docx")                 # .docx .html .md .json .txt
```

**The Netra OCR app,** with no Python needed. It runs on your own computer, on the CPU, and needs no internet after the download:

```bash
docker run -d --name netra-ocr -p 8000:8000 -v netra-data:/data ghcr.io/netra-ai-lab/netra-ocr
# then open http://localhost:8000
```

Upload a scan, a phone photo or a PDF, correct the result block by block next to the page image, and export it to Word. See the [GitHub README](https://github.com/netra-ai-lab/Netra-OCR#5-quick-start) for details.

## Files

| File | Contents |
|---|---|
| `khmerocr_cluster_ar.pth` | **The current model:** cluster vocabulary, autoregressive decoder. The default for `netra-ocr`. |
| `khmerocr_cluster_blockwise.pth` | The same model with the blockwise proposal head: the same text as greedy `ar`, 1.55× faster. |
| `model.safetensors`, `config.json`, `*_khmerocr.py`, `inference.py`, `vocab.json`, tokenizer files | An **earlier character-level model** (124-token vocabulary), loadable with `transformers` and `trust_remote_code=True`. Kept for existing users; new projects should use the cluster model. |
| `khmerocr_epoch*.pth` | Earlier training checkpoints of the character-level model. |
| `assets/` | The figures on this card. |

The cluster model's vocabulary (`char2idx_cluster.json`) ships inside the `netra-ocr` package.

## Limitations

- **Text lines only.** Whole pages need a line detector first; the `netra-ocr` package includes one. Seals, logos and handwriting aren't read.
- **Khmer first.** Trained mostly on Khmer, so English text can come back with small errors. Only Khmer and English are supported.
- **Hard photos and short words.** Error rises on degraded phone photos (5.31% on legal documents) and on isolated words (2.95%).
- **Synthetic training data.** No real scans were used in training, so unusual fonts or layouts can fall outside what the model has seen.

## License and attribution

- **Code and weights:** MIT.
- **Training data:** part of it, [KhmerSynthetic1M](https://huggingface.co/datasets/SoyVitou/KhmerSynthetic1M), is licensed by its authors for research and academic use only. Check its terms before any commercial use of this model.
- **The app:** the `netra-ocr` package's YOLO detector and layout analysis use AGPL-3.0 components (`ultralytics`, `doclayout-yolo`); see [THIRD_PARTY_LICENSES.md](https://github.com/netra-ai-lab/Netra-OCR/blob/master/THIRD_PARTY_LICENSES.md). The recognizer alone doesn't.

## References

1. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.** Dosovitskiy et al., ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
2. **TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models.** Li et al., AAAI 2023. [arXiv:2109.10282](https://arxiv.org/abs/2109.10282)
3. **Toward a Low-Resource Non-Latin-Complete Baseline: An Exploration of Khmer Optical Character Recognition.** Buoy, Iwamura, Srun and Kise, IEEE Access 11, 2023. [DOI: 10.1109/ACCESS.2023.3332361](https://doi.org/10.1109/ACCESS.2023.3332361)
4. **Squeeze-and-Excitation Networks.** Hu, Shen and Sun, CVPR 2018. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
5. **Bidirectional Recurrent Neural Networks.** Schuster and Paliwal, IEEE Transactions on Signal Processing, 1997. [DOI: 10.1109/78.650093](https://doi.org/10.1109/78.650093)
6. **Blockwise Parallel Decoding for Deep Autoregressive Models.** Stern, Shazeer and Uszkoreit, NeurIPS 2018. [arXiv:1811.03115](https://arxiv.org/abs/1811.03115)
7. **DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception.** Zhao, Kang, Wang and He, 2024. [arXiv:2410.12628](https://arxiv.org/abs/2410.12628)
8. **Balraj98** (2018). *Stanford background dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/balraj98/stanford-background-dataset
9. **EKYC Solutions** (2022). *Khmer OCR benchmark dataset (KHOB)* [Data set]. GitHub. https://github.com/EKYCSolutions/khmer-ocr-benchmark-dataset
10. **Em, Valy, Gosselin and Kong** (2024). *Khmer text recognition dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/emhengly/khmer-text-recognition-dataset
