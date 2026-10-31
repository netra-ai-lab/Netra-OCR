# Training, fine-tuning and evaluating the recognizer

Everything here works on Netra OCR's text-line recognizer (the CNN + Transformer model that reads a line crop). Each job is one script. Run them from the repository root:

| Script | What it does |
| --- | --- |
| `bash training/setup.sh` | Creates `.venv` with the app and the training code (PyTorch). Run once. |
| `bash training/finetune.sh DATA` | Fine-tunes the published model on your own lines. |
| `bash training/train.sh` | Trains a model from scratch. |
| `bash training/evaluate.sh MODEL...` | Scores models: CER, WER, exact match, speed. |
| `bash training/export.sh CHECKPOINT` | Turns a checkpoint into the ONNX model folder the app runs. |

Every script prints its options with `--help`. They also take any option of the underlying command (`netra-train train|evaluate|export --help`).

None of this ships in `pip install netra-ocr`. The app (`netra_ocr/`) never imports PyTorch or anything in this folder.

## 1. Setup

```bash
git clone https://github.com/netra-ai-lab/Netra-OCR && cd Netra-OCR
bash training/setup.sh
```

This:
- creates `.venv` in the repository root (or uses the existing one; `VENV=path` picks another);
- installs the app with ONNX Runtime (the GPU build when `nvidia-smi` finds a GPU), plus this folder's package `netra-ocr-train` (PyTorch, `datasets`, `khmerspell`, ONNX);
- uses Python 3.10 or newer, and [uv](https://docs.astral.sh/uv/) when it's installed.

The environment takes about 6 GB, mostly PyTorch's CUDA libraries. A first install took 13 minutes here, mostly downloading.

The scripts use `.venv` on their own; you don't need to activate it.

A CUDA GPU is strongly recommended for training. Evaluation and export run fine on a CPU.

## 2. Your data

Every script takes the same two kinds of data:
- **A folder** of line images, each with a text file of the same name holding that line's text (UTF-8):

  ```
  my_lines/
    images/0001.png   images/0002.jpg   ...
    labels/0001.txt   labels/0002.txt   ...
  ```

  One image is one line of text, cropped closely, like the output of the app's line detector.
  - Image formats: PNG, JPG, BMP, TIFF and WebP.
  - Images without a label file are skipped, with a warning.
- **A Hugging Face dataset**, as `user/name`, or `user/name:split:text_column` when the split isn't `train` or the text column isn't `label` or `text`. It needs an `image` column.

Several sources can be given at once, e.g. `--data my_lines/ more_lines/ user/name`.

Characters the vocabulary doesn't cover are listed in a warning at the start of training. The model can't learn them; they read back as a space. The vocabulary (`netra_ocr_train/char2idx_cluster.json`) covers Khmer, Latin letters, digits and common punctuation. Zero-width spaces in labels are dropped, because they're invisible in the image.

## 3. Fine-tune

```bash
bash training/finetune.sh my_lines/ --name my-model
```

This starts from the published model's weights: [`pytorch/khmerocr_cluster_ar.pth`](https://huggingface.co/Netra-Lab/Netra-OCR/tree/main/pytorch) in Netra-Lab/Netra-OCR, downloaded once.

What it does:
- holds out 5% of your lines (50–500) to judge each epoch;
- trains for 10 epochs (learning rate 3e-5, 1 warm-up epoch, batch 32, light augmentation: blur, noise, pixelation, thinning, shift);
- writes everything to `training/runs/my-model/`.

Any default can be overridden:

```bash
bash training/finetune.sh my_lines/ --name my-model --epochs 20 --batch-size 16
bash training/finetune.sh my_lines/ --name my-model --val-data my_test_lines/   # your own validation set
```

How much data:
- a few hundred lines already adapt the model to a new font or document type;
- a few thousand lines are better;
- your lines are added to what the model knows, not a replacement for it. Check the published test sets afterwards (section 5) to see that nothing else got worse.

| File in the run folder | |
| --- | --- |
| `best.pth` | The epoch with the lowest validation CER. Evaluate and export this one. |
| `last.pth` | The latest epoch, with the optimizer state, for `--resume`. |
| `metrics.csv` | Per epoch: training loss, validation CER/WER/exact match, learning rate, minutes. |
| `train.log` | The full log. Before epoch 1 it also scores the starting model, as the baseline. |

An interrupted run continues with:

```bash
bash training/finetune.sh my_lines/ --name my-model --resume training/runs/my-model/last.pth
```

## 4. Train from scratch

```bash
bash training/train.sh --name scratch
```

By default this trains on the four public datasets the published model used (about 1.3M lines, ~16 GB, downloaded once into `training/.hf_cache/`).

The defaults:
- 100 epochs of 50,000 random lines each;
- learning rate 1e-4 with 5 warm-up epochs, then a cosine decay;
- batch 32;
- bf16 on the GPU.

Pass `--data` to use other data. `--resume training/runs/scratch/last.pth --name scratch` continues a run, with `--epochs` as the total to reach.

> **Licence:** [`SoyVitou/KhmerSynthetic1M`](https://huggingface.co/datasets/SoyVitou/KhmerSynthetic1M) is for research only, with no commercial use. To train a model for commercial use, pass your own `--data` without it.

Speed: on an RTX 3060, a short run trained about 40 lines per second (250 lines plus validation in 7 seconds).

## 5. Evaluate

```bash
bash training/evaluate.sh published                                     # the model the app ships
bash training/evaluate.sh published training/runs/my-model/best.pth     # compare two models
bash training/evaluate.sh training/runs/my-model/best.pth --test my_test_lines/ user/dataset
```

A model can be:
- `published`: the app's own model, run exactly as the app runs it (ONNX Runtime);
- a checkpoint (`.pth`), decoded greedily like the app (`--beam 3` for beam search, as in the paper's table);
- a folder from `export.sh`.

The default test set is `training/test_set/khob_eval` when you have it, else the public [KHOB](https://huggingface.co/datasets/SoyVitou/KHOB-LEVEL-1) set from Hugging Face.

The output:
- **A table:**

  ```
  model                                test set     lines     CER     WER   exact  ms/line
  published                            khob_eval      325   1.29%  18.29%  83.69%      6.5
  training/runs/my-model/best.pth      khob_eval      325   ...
  ```

- **Files** in `training/runs/eval-<date-time>/`: one CSV per model and test set with every line's target and prediction, raw and normalized, plus `summary.json`.

Text is normalized the same way for every model before comparing:
- Khmer canonical ordering (`khmerspell`);
- whitespace collapsed;
- zero-width characters dropped.

So CER counts real reading errors only.

## 6. Use your model in the app

```bash
bash training/export.sh training/runs/my-model/best.pth
```

This writes `training/runs/my-model/bundles/recognition/`: ONNX graphs, `netra-model.json` and the vocabulary, about 80 MB. It then checks on `training/test_set/khob_eval` that every line reads the same as the PyTorch checkpoint (`CHECK=DIR` picks another folder, `CHECK=` skips the check).

Then point the app at the folder. The line detector and layout model still come from Hugging Face:

```bash
NETRA_BUNDLES=training/runs/my-model/bundles netra_ocr serve                  # the web app
NETRA_BUNDLES=training/runs/my-model/bundles netra_ocr --image page.jpg --output page.docx --layout
```

```python
from netra_ocr.recognition import recognize
recognize("line.png", bundle="training/runs/my-model/bundles/recognition")
```

The folder only replaces the default `ar` decoder. `decoder="blockwise"` still uses the published blockwise model (see section 7).

To halve the folder's size, store the weights as fp16 (the maths stays fp32, as in the published model):

```bash
.venv/bin/python -m netra_ocr_train.scripts.make_variant training/runs/my-model/bundles/recognition store-fp16
```

Then evaluate the folder again.

## 7. Advanced

| Tool | |
| --- | --- |
| `scripts/init_blockwise_decoder.py`, `train.py --decoder-type blockwise` | Blockwise decoding ([Stern et al. 2018](https://arxiv.org/abs/1811.03115)): freezes a trained model and trains a small head that proposes several tokens per step. The text is identical to greedy decoding. `scripts/sweep_block_size.sh` trains and compares several block sizes. |
| `scripts/build_cluster_vocab.py`, `scripts/validate_tokenizer.py` | Build the grapheme-cluster vocabulary from a corpus, and check it round-trips. A new vocabulary needs `scripts/migrate_checkpoint.py` to carry the weights over, then training. |
| `scripts/export_yolo.py`, `scripts/make_variant.py` | Export the YOLO line detector and the layout model (needs `pip install -e "training[yolo]"`, AGPL-3.0), and make fp16/int8 copies of any model folder. |
| `scripts/strip_optimizer.py` | Drop the optimizer state from a checkpoint (a third of the size). |
| `data_generation/`, `fonts/`, `texts/` | The synthetic text-line generators, their fonts and corpus. |
| `baselines/` | The third-party engines in the results table (Tesseract, Surya, Qwen2.5-VL, DeepSeek-OCR) and their LoRA fine-tunes, scored the same way. They need their own environments; see each script's docstring. |
| `notebooks/` | The original training notebooks, kept as the reference for the preprocessing. |

## Layout

```
training/
  setup.sh finetune.sh train.sh evaluate.sh export.sh   the scripts above
  netra_ocr_train/          the package (netra-train command)
    model/                  the network: SE-CNN, Transformer encoder, BiLSTM, decoder, blockwise head
    data.py                 data specs -> training tensors; augment.py
    metrics.py              normalization, CER / WER, the test-set folder format
    predictor.py, preprocessor.py, cluster_tokenizer.py, tokenization/   PyTorch inference
    scripts/                train, evaluate, export_recognizer and the tools above
    char2idx_cluster.json   the vocabulary
  baselines/  data_generation/  fonts/  texts/  notebooks/
  runs/                     your runs and evaluations (git-ignored)
  weights/, test_set/       local checkpoints and test sets (git-ignored)
```

How a model reaches the app:

```
train / fine-tune (PyTorch) -> best.pth -> export.sh -> model folder (ONNX + netra-model.json) -> NETRA_BUNDLES or Hugging Face -> app
```

For a release:
1. Upload the folder to [Netra-Lab/Netra-OCR](https://huggingface.co/Netra-Lab/Netra-OCR).
2. Move `MODEL_REVISION` in `netra_ocr/models.py` to the new commit (see `docs/RELEASING.md`).
