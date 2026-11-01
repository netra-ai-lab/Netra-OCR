# Training, fine-tuning and evaluating the recognizer

Everything here works on Netra OCR's text-line recognizer: the CNN + Transformer model that reads one cropped line of text. Each job is one script, run from the repository root:

| Script | What it does |
| --- | --- |
| `bash training/setup.sh` | Creates `.venv` with the app and the training code (PyTorch). Run once. |
| `bash training/check_data.sh DATA` | Checks a dataset before you train on it. |
| `bash training/finetune.sh DATA` | Fine-tunes the published model on your lines. |
| `bash training/train.sh` | Trains a model from scratch. |
| `bash training/evaluate.sh MODEL...` | Scores models: CER, WER, exact match, speed. |
| `bash training/export.sh CHECKPOINT` | Turns a checkpoint into the ONNX model folder the app runs. |

Every script prints its options with `--help`, and passes any other option to the command underneath (`netra-train train|evaluate|export|check --help`).

None of this ships in `pip install netra-ocr`. The app (`netra_ocr/`) never imports PyTorch or anything in this folder.

**Quick start**, with the 24-line example dataset in this folder:

```bash
bash training/setup.sh
bash training/check_data.sh training/examples/lines
bash training/finetune.sh training/examples/lines --name try-it --epochs 2
bash training/evaluate.sh published training/runs/try-it/best.pth
```

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

## 2. Datasets

### What a dataset is

One **image** is one line of text, cropped closely (like the boxes the app's line detector finds), with its **text** in UTF-8. Any height works: the model resizes every line to 48 pixels high. Image formats: PNG, JPG, BMP, TIFF and WebP.

Every script (`--data`, `--val-data`, `--test`, `CHECK=`) takes a dataset in any of these forms:

**A folder with `images/` and `labels/`.** Each label file has the image's name, with `.txt`:

```
my_lines/
├── images/
│   ├── 0001.png
│   ├── 0002.jpg
│   └── ...
└── labels/
    ├── 0001.txt        ← the text in 0001.png, e.g. សូប៊ីក្រោម
    ├── 0002.txt
    └── ...
```

**A folder of images with the text beside each one**, as `.txt` or `.gt.txt` (the tesstrain layout):

```
my_lines/
├── 0001.png
├── 0001.gt.txt
├── 0002.png
├── 0002.gt.txt
└── ...
```

**A manifest file** listing image paths (relative to the file) and texts:

| File | Each line |
| --- | --- |
| `labels.tsv` (or `.txt`) | `images/0001.png<TAB>សូប៊ីក្រោម` (the PaddleOCR layout) |
| `labels.csv` | a header row first: `image,text`. Other accepted column names: `path`/`file_name` for the image, `label`/`transcription` for the text. |
| `labels.jsonl` | `{"image": "images/0001.png", "text": "សូប៊ីក្រោម"}` |

**A Hugging Face dataset**, by its id:

| Spec | Means |
| --- | --- |
| `user/name` | the `train` split |
| `user/name:validation` | another split |
| `user/name:train[:5000]` | the first 5,000 rows |
| `user/name:train:transcription` | the text is in the `transcription` column (by default the first of `text`, `label`, `transcription` it finds) |

The image column is found by its type.

**A preset name**, for the published datasets:

| Preset | Dataset | Lines | Use |
| --- | --- | --- | --- |
| `synthetic` | the four datasets the published model was trained on: [SoyVitou/KhmerSynthetic1M](https://huggingface.co/datasets/SoyVitou/KhmerSynthetic1M), [seanghay/khmer-hanuman-100k](https://huggingface.co/datasets/seanghay/khmer-hanuman-100k), [Netra-Lab/khmer-scene-text-synthetic-contrast](https://huggingface.co/datasets/Netra-Lab/khmer-scene-text-synthetic-contrast), [Netra-Lab/khmer-document-synthetic-low-res](https://huggingface.co/datasets/Netra-Lab/khmer-document-synthetic-low-res) | ~1.3M, ~16 GB | training from scratch (the default) |
| `khob` | [Netra-Lab/khob-eval](https://huggingface.co/datasets/Netra-Lab/khob-eval): scanned government documents, cropped from the [KHOB benchmark](https://github.com/EKYCSolutions/khmer-ocr-benchmark-dataset) | 325 | testing (the default test set) |
| `printed-words` | [Netra-Lab/khmer-printed-words](https://huggingface.co/datasets/Netra-Lab/khmer-printed-words): short synthetic words in 10 fonts | 5,000 | testing |

Hugging Face datasets download once, into `training/.hf_cache/`.

> **Licence:** KhmerSynthetic1M is for research only, with no commercial use. To train a model for commercial use, leave it out of `--data`. Any of the four can also be used on its own, by its id.

Several datasets can be given at once and are mixed together:

```bash
--data my_lines/ more_lines.tsv user/name:train[:5000]
```

The example in [`examples/`](examples/) has 24 lines both as a folder (`examples/lines/`) and as a manifest (`examples/lines.tsv`).

### Characters

The vocabulary (`netra_ocr_train/char2idx_cluster.json`) covers Khmer, Latin letters, digits and common punctuation. The model can't learn other characters; they read back as a space. The checker and the start of training list any your labels contain.

Zero-width spaces and byte-order marks in labels are dropped, because they're invisible in the image.

### Check your data first

```bash
bash training/check_data.sh my_lines/
```

It opens every image (a progress bar shows while it reads) and prints what it found:

```
== training/examples/lines
   format    folder: images/ + labels/
   lines     24
   height    47 – 55 (median 49) px
   width     67 – 216 (median 108) px
   aspect    1.2 – 4.5 (median 2.2) (width / height)
   text      10 – 21 (median 11) characters
   ✓ every character is in the vocabulary
   examples
     0001.jpg                     សូប៊ីក្រោម
     0002.jpg                     វេនេហ្ស៊ុយឡា
     0003.jpg                     កុមារាកុមារី
     0004.jpg                     ថ្កុំថ្កើង
     0005.jpg                     ប្រតិព័ទ្ធ
   ✓ ready

All good.
```

On a folder with problems it says what to fix, and exits with 1:

```
== broken
   format    folder: images/ + labels/
   lines     23
   ⚠ skipped 1: no label for 0003.jpg
   ...
   ✗ 1 image can't be read: 0005.jpg (UnidentifiedImageError)
   ⚠ characters the model can't learn (read back as a space): '★' U+2605 ×1, '©' U+00A9 ×1
   ...
   ✗ fix the errors above first
```

| Message | Meaning |
| --- | --- |
| `⚠ skipped N: no label for X` | an image without a label file (or a manifest row whose image is missing). Training skips it. |
| `✗ N images can't be read` | a broken or non-image file. Remove or replace it: training would stop on it. |
| `⚠ N labels are empty` | an empty text. Those lines teach the model to read nothing. |
| `⚠ most images are about as tall as they are wide` | probably whole paragraphs or pages. The model reads one line per image. |
| `⚠ characters the model can't learn` | see [Characters](#characters). |

A Hugging Face dataset is checked on its first 1,000 lines (`--sample N` for more).

## 3. Fine-tune

```bash
bash training/finetune.sh my_lines/ --name my-model
```

This starts from the published model's weights: [`pytorch/khmerocr_cluster_ar.pth`](https://huggingface.co/Netra-Lab/Netra-OCR/tree/main/pytorch) in Netra-Lab/Netra-OCR, downloaded once.

What it does:
- holds out 5% of your lines (between 50 and 500, and at most a fifth) to judge each epoch;
- scores the published model on them first, as the baseline;
- trains for up to 10 epochs, and stops early after 3 epochs without a better validation CER;
- learning rate 3e-5 with 1 warm-up epoch, batch 32, light augmentation (blur, noise, pixelation, thinning, shift);
- writes everything to `training/runs/my-model/`.

### What you'll see

A real run: 2,000 lines of printed Khmer words, 3 epochs, on an RTX 3060. The snapshots below are copied from its terminal.

**While an epoch trains**, the log so far stays on screen and a progress bar at the bottom updates in place: the lines done, the time spent and left in this epoch, the speed, and the running loss and learning rate.

```
INFO - Run folder: training/runs/my-model
INFO - Starting from the weights of published (fresh optimizer, epoch 0)
INFO - Loading data (Hugging Face datasets download on the first use)...
INFO - my_lines: 2,000 lines (folder: images/ + labels/)
WARNING - Characters the vocabulary can't encode (learned as <unk>, read back as a space): 'ឫ'×17, '\u200c'×4, 'ឪ'×1
INFO - Held out 100 of the 2,000 lines for validation
INFO - Training on 1,900 lines, validating on 100
INFO - 3 epochs of 1,900 lines (60 batches of 32), on the GPU (NVIDIA GeForce RTX 3060)
INFO - Before training: val CER 2.01%, WER 13.00%, exact 87.00%
epoch 1/3:  51%|██████████          | 960/1900 [00:23<00:03, 286.68line/s, loss=0.7441, lr=1.5e-05]
```

**After each epoch**, the epoch's bar is replaced by a `validating` bar over the held-out lines (here after epoch 2):

```
INFO - epoch 1/3: loss 0.4828 · val CER 1.56% WER 10.00% exact 90.00% (best) · 32 s · about 1.1 min left
validating:  48%|██████████████████████                        | 48/100 [00:00<00:00, 133.85line/s]
```

Then that bar is replaced by a summary line for the epoch:
- the training loss;
- the validation CER, WER and exact match;
- `(best)` when it's the best epoch so far, else the best CER so far;
- how long the epoch took, and about how long the whole run has left.

**When it finishes:**

```
INFO - Before training: val CER 2.01%, WER 13.00%, exact 87.00%
INFO - epoch 1/3: loss 0.4828 · val CER 1.56% WER 10.00% exact 90.00% (best) · 32 s · about 1.1 min left
INFO - epoch 2/3: loss 0.1787 · val CER 1.10% WER 6.00% exact 94.00% (best) · 11 s · about 22 s left
INFO - epoch 3/3: loss 0.1638 · val CER 1.01% WER 5.00% exact 95.00% (best) · 8 s
INFO - Done in 1.1 min, peak GPU memory 2.4 GB. Best validation CER 1.01% (epoch 3): training/runs/my-model/best.pth
Fine-tuned model: training/runs/my-model/best.pth  (per-epoch scores in training/runs/my-model/metrics.csv)
Compare it with the published model (add --test YOUR_TEST_LINES for your own test set):
  bash training/evaluate.sh published training/runs/my-model/best.pth
Then export it for the app:
  bash training/export.sh training/runs/my-model/best.pth
```

What to look for:
- the loss falls, quickly in the first epoch and slowly after;
- the validation CER falls below the `Before training` score. If it rises instead, the model is forgetting more than it learns: use more data, or fewer epochs;
- the first epoch is slower: the GPU tunes its kernels and the images are read from disk for the first time.

Without a terminal (`nohup`, a pipe, CI), the bars are off, and a line every 50 steps (`--log-every`) shows progress instead:

```
INFO - epoch 1/1 step 50/60 loss=0.2877 lr=2.50e-05
INFO - epoch 1/1 step 60/60 loss=0.2499 lr=3.00e-05
```

To watch a run in the background, follow its log: `tail -f training/runs/my-model/train.log`.

### The run folder

| File | |
| --- | --- |
| `best.pth` | The epoch with the lowest validation CER, weights only (77 MB). Evaluate and export this one. |
| `last.pth` | The latest epoch, with the optimizer state (230 MB), for `--resume`. |
| `metrics.csv` | One row per epoch. |
| `train.log` | The full log, including the per-step lines. |

`metrics.csv` from the run above (CER/WER/exact as fractions, time in minutes):

```
epoch,train_loss,val_cer,val_wer,val_exact,lr,minutes
1,0.48275,0.01555,0.10000,0.90000,3.000e-05,0.54
2,0.17871,0.01098,0.06000,0.94000,1.588e-05,0.19
3,0.16383,0.01006,0.05000,0.95000,1.005e-06,0.14
```

### Options

Any default can be overridden:

```bash
bash training/finetune.sh my_lines/ --name my-model --epochs 20 --patience 5
bash training/finetune.sh my_lines/ --name my-model --batch-size 16            # less GPU memory
bash training/finetune.sh my_lines/ --name my-model --val-data my_val_lines/   # your own validation set
bash training/finetune.sh my_lines/ --name my-model --patience 0               # never stop early
```

An interrupted run continues where it stopped:

```bash
bash training/finetune.sh my_lines/ --name my-model --resume training/runs/my-model/last.pth
```

### How much data, and keeping what the model knows

- A few hundred lines already adapt the model to a new font or document type; a few thousand are better.
- Fine-tuning on one kind of text can make the model worse at others. Check with a published test set afterwards (section 5). In the run above, the held-out lines improved from 1.32% to 0.91% CER while KHOB stayed about the same (1.29% → 1.33%).
- If another test set gets worse, mix some public lines into the training data:

  ```bash
  bash training/finetune.sh my_lines/ Netra-Lab/khmer-document-synthetic-low-res:train[:5000] --name my-model
  ```

## 4. Train from scratch

```bash
bash training/train.sh --name scratch
```

By default this trains on the `synthetic` preset: the four public datasets the published model used (about 1.3M lines, ~16 GB, downloaded once). Pass `--data` to use other data, in any form from section 2.

The defaults:
- 100 epochs of 50,000 random lines each;
- learning rate 1e-4 with 5 warm-up epochs, then a cosine decay;
- batch 32;
- bf16 on the GPU;
- no early stopping (`--patience N` turns it on).

`--resume training/runs/scratch/last.pth --name scratch` continues a run, with `--epochs` as the total to reach.

> **Licence:** KhmerSynthetic1M is for research only. For a model you'll use commercially, pass your own `--data` without it.

Speed depends on the line length:
- short printed words ran at about 570 lines per second on an RTX 3060;
- document lines from `Netra-Lab/khmer-document-synthetic-low-res` took about 40 seconds per 1,900 lines.

So expect very roughly 15–20 minutes per 50,000-line epoch on document lines (an estimate, not a measured full run). The summary line after each epoch gives the time left for your data.

## 5. Evaluate

```bash
bash training/evaluate.sh published                                     # the model the app ships, on KHOB
bash training/evaluate.sh published training/runs/my-model/best.pth     # compare two models
bash training/evaluate.sh published training/runs/my-model/best.pth --test khob my_test_lines/
```

A model can be:
- `published`: the app's own model, run exactly as the app runs it (ONNX Runtime);
- a checkpoint (`.pth`), decoded greedily like the app (`--beam 3` for beam search, as in the paper's table);
- a folder from `export.sh`.

A test set is any dataset form from section 2. The default is `khob`. Use lines the model wasn't trained on, or the scores will look better than they are.

A progress bar per model and test set shows the running CER:

```
published on khob_eval: 100%|███████████████████████| 325/325 [00:04<00:00, 76.74line/s, CER=1.29%]
```

Then a table compares them all. This one is from the fine-tune above, with 500 printed-word lines it never saw as `my_test_lines`:

```
model                            test set                      lines     CER     WER   exact  ms/line
published                        khob_eval                       325   1.29%  18.29%  83.69%     12.5
published                        my_test_lines                   500   1.32%  12.20%  88.60%      3.8
training/runs/my-model/best.pth  khob_eval                       325   1.33%  17.11%  84.92%     16.3
training/runs/my-model/best.pth  my_test_lines                   500   0.91%   8.80%  92.00%      7.6

Predictions and summary.json in training/runs/eval-20261101-034821/
```

| Column | |
| --- | --- |
| CER | character error rate: the edits (insertions, deletions, substitutions) needed to turn the prediction into the label, divided by the label's length. Lower is better. |
| WER | the same over words. Khmer has no spaces between words, so a word here is a space-separated chunk, often a whole phrase; one wrong character fails it. |
| exact | lines read with no error at all. |
| ms/line | time per line. ONNX (`published`, exported folders) is faster than a `.pth` in PyTorch. |

The run folder (`training/runs/eval-<date-time>/`, or `--out DIR`) has:
- one CSV per model and test set, with every line's label and prediction, raw and normalized;
- `summary.json`, with the table's numbers.

Text is normalized the same way for every model before comparing:
- Khmer canonical ordering (`khmerspell`);
- whitespace collapsed;
- zero-width characters dropped.

So CER counts real reading errors only.

## 6. Use your model in the app

```bash
bash training/export.sh training/runs/my-model/best.pth
```

This writes `training/runs/my-model/bundles/recognition/`: ONNX graphs, `netra-model.json` and the vocabulary, about 80 MB. It then checks that every line of the `khob` test set reads the same as with the PyTorch checkpoint:

```
Model folder written to training/runs/my-model/bundles/recognition: {"decoder_step.onnx": "26.3 MB", "encoder.onnx": "54.4 MB", ...}
checking on cpu: 100%|██████████████████████████████████████████| 21/21 [00:17<00:00,  1.40batch/s]
[Recognizer(netra-ocr-recognition-finetuned 1.0.0, device=cpu, variant=fp32)] identical to PyTorch on 325/325 lines of khob
Model folder: training/runs/my-model/bundles/recognition
Run the app with it:   NETRA_BUNDLES=training/runs/my-model/bundles netra_ocr serve
Score it:              bash training/evaluate.sh training/runs/my-model/bundles/recognition
```

`CHECK=my_test_lines/ bash training/export.sh ...` checks on another dataset; `CHECK=` skips the check. The export fails if fewer than 99% of the lines match.

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
  setup.sh check_data.sh finetune.sh train.sh evaluate.sh export.sh   the scripts above
  examples/                 a 24-line example dataset, as a folder and as a manifest
  netra_ocr_train/          the package (netra-train command)
    model/                  the network: SE-CNN, Transformer encoder, BiLSTM, decoder, blockwise head
    data.py                 dataset forms and presets -> training tensors; augment.py
    metrics.py              normalization, CER / WER
    predictor.py, preprocessor.py, cluster_tokenizer.py, tokenization/   PyTorch inference
    scripts/                train, evaluate, check_data, export_recognizer and the tools above
    char2idx_cluster.json   the vocabulary
  baselines/  data_generation/  fonts/  texts/  notebooks/
  runs/                     your runs and evaluations (git-ignored)
  weights/, test_set/       local checkpoints and test sets (git-ignored)
```

How a model reaches the app:

```
your data -> check_data.sh -> finetune.sh / train.sh (PyTorch) -> best.pth -> export.sh -> model folder (ONNX) -> NETRA_BUNDLES -> app
```

For a release:
1. Upload the folder to [Netra-Lab/Netra-OCR](https://huggingface.co/Netra-Lab/Netra-OCR).
2. Move `MODEL_REVISION` in `netra_ocr/models.py` to the new commit (see `docs/RELEASING.md`).
