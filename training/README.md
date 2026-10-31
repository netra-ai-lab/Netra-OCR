# Training Netra OCR's models

Everything here is research: the PyTorch recognizer, its training and
evaluation, the synthetic-data generators, and the step that turns a trained
checkpoint into the model files the app runs. The app (`netra_ocr/`) never
imports anything from this folder, and `pip install netra-ocr` doesn't ship it.

## How a model gets from here to the app

```
train (PyTorch)  ->  checkpoint (.pth / .pt)  ->  export  ->  bundle  ->  Hugging Face  ->  app
                     training/weights/            ONNX graphs + manifest   Netra-Lab/Netra-OCR
```

A **bundle** is a folder with the network as ONNX graphs (layers *and*
weights), a `netra-model.json` manifest describing the pre-processing and
decoding around the network, and side files such as the vocabulary.
[Netra Runtime](https://github.com/netra-ai-lab/Netra-Runtime) reads the
manifest and runs the graphs, so the app needs neither PyTorch nor this code.

## Layout

| Path | Holds |
| --- | --- |
| `netra_ocr_train/model/` | The recognizer's architecture (SE network, Transformer encoder, BiLSTM, decoder, blockwise head) |
| `netra_ocr_train/predictor.py`, `preprocessor.py`, `postprocess.py`, `tokenization/`, `cluster_tokenizer.py` | PyTorch inference, used by training checks and evaluation |
| `netra_ocr_train/*.json` | Vocabularies (`char2idx_cluster.json` is the current one) |
| `netra_ocr_train/scripts/` | `train.py`, `evaluate_benchmark.py`, `export_runtime_bundle.py`, checkpoint tools, VLM baselines |
| `data_generation/`, `fonts/`, `texts/` | Synthetic text-line generators, their fonts and corpus |
| `notebooks/` | The original training notebooks, kept as a reference for the preprocessing contract |
| `weights/` | Checkpoints (git-ignored, except the YOLO line detector in `yolo26s_best/`) |
| `test_set/` | Local benchmark sets, e.g. `khob_eval` (git-ignored) |

## Setup

From the repository root, in the same environment as the app:

```bash
pip install -e ".[cpu]" -e training
```

Evaluation also uses `datasets` and `khmerspell`, which aren't listed as
dependencies yet.

Run every command from the repository root; the default paths assume it.

## Common commands

```bash
# Train (or continue) the recognizer
python -m netra_ocr_train.scripts.train --decoder-type ar \
    --resume training/weights/khmerocr_cluster_ar_init.pth --epochs 100

# Benchmark CER / WER / speed
python -m netra_ocr_train.scripts.evaluate_benchmark --local-dir training/test_set/khob_eval

# Export the recognizer as a bundle, checking it matches PyTorch
python -m netra_ocr_train.scripts.export_runtime_bundle \
    --model training/weights/khmerocr_cluster_ar.pth --out bundles/recognition \
    --check training/test_set/khob_eval

# Export the YOLO line detector as a bundle
netra-runtime export-yolo training/weights/yolo26s_best/best.pt bundles/lines --name netra-ocr-lines
```

Then upload the bundle folders to
[Netra-Lab/Netra-OCR](https://huggingface.co/Netra-Lab/Netra-OCR) and the app
picks them up (see `docs/RELEASING.md`).
