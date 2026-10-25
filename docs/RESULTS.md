# Results

All numbers are on single text-line crops. CER and WER are in %, and exact match is the share of lines read with no error. Lower CER and WER are better.

## 1. Against other OCR engines

Every engine's output is normalised the same way: `khmerspell` normalisation and whitespace collapse. Printed Words uses the first 500 of its 5,000 lines for every engine.

| Engine | KHOB CER | KHOB WER | KHOB exact | Legal CER | Legal WER | Legal exact | Printed CER | Printed WER | Printed exact |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tesseract 5 (`khm` + `eng`, single line) | 5.12 | 59.29 | 41.5 | 41.78 | 58.70 | 32.7 | 20.14 | 55.40 | 46.0 |
| Surya OCR | 14.52 | 46.61 | 65.5 | 51.16 | 79.57 | 48.2 | 28.45 | 79.20 | 29.0 |
| Qwen2.5-VL-3B, zero-shot | 136.60 | 307.67 | 0.0 | 102.74 | 116.52 | 15.5 | 123.41 | 132.20 | 0.0 |
| Qwen2.5-VL-3B, LoRA on Khmer | 89.21 | 118.58 | 0.0 | 84.63 | 97.61 | 15.5 | 98.62 | 107.80 | 0.0 |
| DeepSeek-OCR, zero-shot | 147.40 | 120.06 | 4.3 | 798.68 | 832.61 | 14.6 | 99.08 | 116.60 | 0.0 |
| DeepSeek-OCR, LoRA on Khmer | 33.23 | 91.45 | 10.5 | 44.22 | 70.22 | 18.2 | 69.79 | 103.00 | 1.6 |
| DeepSeek-OCR, community Khmer fine-tune | 46.71 | 97.64 | 1.9 | 47.21 | 81.30 | 15.5 | 61.59 | 99.40 | 1.2 |
| **Netra-OCR (beam 3)** | **0.96** | **15.34** | **86.2** | **5.31** | **22.61** | **67.3** | **2.95** | **23.60** | **76.4** |

Test sets: KHOB 325 lines, Legal Documents 220, Printed Words 500.

**Baselines.**
- **Tesseract 5** uses the official `khm` and `eng` models from `tessdata_best`, in single-line mode (`--psm 7`).
- **Surya OCR** (`datalab-to/surya-ocr-2`) reads each crop as one text block.
- **Qwen2.5-VL-3B-Instruct** and **DeepSeek-OCR** don't document Khmer support. Zero-shot, they score above 100% CER: their output has more wrong characters than the reference has characters. The "LoRA on Khmer" rows are our short fine-tunes: rank 16, 300 steps, 6,000 lines sampled from `khmer-hanuman-100k` and our two synthetic sets.
- **Community fine-tune** is `KrorngAI/deepseek_ocr_Khmer_finetuned`, a LoRA of DeepSeek-OCR trained on Khmer scene text.
- PaddleOCR and EasyOCR have no Khmer support, so they aren't listed.

## 2. Decoders

Greedy autoregressive decoding against blockwise parallel decoding, on KHOB (325 lines), with the same base model:

| Decoder | CER | WER | Exact match | ms per line | Lines per second | Tokens accepted per step |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Autoregressive, greedy | 1.17 | 17.11 | 84.9 | 17.60 | 56.8 | 1 |
| **Blockwise, 4 tokens proposed** | **1.17** | **17.11** | **84.9** | **11.35** | **88.1** | **2.7** |

- **Same output.** Every accuracy number is identical, as blockwise decoding guarantees: each proposed token is checked against the same decoder's own greedy choice.
- **Speed.** Blockwise is 1.55× faster end to end. The decoder runs about 2.7× fewer passes, but the encoder and the rest of the pipeline cost the same.
- **Greedy vs beam.** Greedy CER is 1.17% here, against 0.96% with beam search of width 3 (section 1). Blockwise decoding is greedy only.
