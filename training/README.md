# Training Netra OCR's models

Everything here is research: the PyTorch recognizer, its training and
evaluation, the synthetic-data generators, and the step that turns a trained
checkpoint into the model files the app runs. The app (`netra_ocr/`) never
imports anything from this folder, and `pip install netra-ocr` doesn't ship it.

## How a model gets from here to the app

```
train (PyTorch)  ->  checkpoint (.pth / .pt)  ->  export  ->  model folder  ->  Hugging Face  ->  app
                     training/weights/            ONNX graphs + config    Netra-Lab/Netra-OCR
```

A **model folder** holds the network as ONNX graphs (layers *and* weights),
a `netra-model.json` config with the settings the app's pre- and
post-processing need (line height, chunk width, decoder shape, class names,
thresholds), and side files such as the vocabulary. The app's
`netra_ocr/inference/` (and `web/` in the browser) runs the graphs on ONNX
Runtime, so it needs neither PyTorch nor this code. The exporters write the
config with `netra_ocr_train/model_folder.py`.

## Layout

| Path | Holds |
| --- | --- |
| `netra_ocr_train/model/` | The recognizer's architecture (SE network, Transformer encoder, BiLSTM, decoder, blockwise head) |
| `netra_ocr_train/predictor.py`, `preprocessor.py`, `postprocess.py`, `tokenization/`, `cluster_tokenizer.py` | PyTorch inference, used by training checks and evaluation |
| `netra_ocr_train/*.json` | Vocabularies (`char2idx_cluster.json` is the current one) |
| `netra_ocr_train/scripts/` | `train.py`, `evaluate_benchmark.py`, `export_recognizer.py`, `export_yolo.py`, `make_variant.py`, checkpoint tools, VLM baselines |
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

# Export the recognizer, checking it matches PyTorch
python -m netra_ocr_train.scripts.export_recognizer \
    --model training/weights/khmerocr_cluster_ar.pth --out bundles/recognition \
    --check training/test_set/khob_eval

# Export the YOLO line detector and the layout model
python -m netra_ocr_train.scripts.export_yolo training/weights/yolo26s_best/best.pt bundles/lines \
    --name netra-ocr-lines --license AGPL-3.0
python -m netra_ocr_train.scripts.export_yolo doclayout_docstructbench.pt bundles/layout \
    --name netra-ocr-layout --imgsz 1024 --conf 0.2 --loader doclayout_yolo:YOLOv10

# Add fp16 copies of the detector graphs, used by default on CUDA
python -m netra_ocr_train.scripts.make_variant bundles/lines fp16 --prefer-on cuda tensorrt
```

Then upload the model folders to
[Netra-Lab/Netra-OCR](https://huggingface.co/Netra-Lab/Netra-OCR) and the app
picks them up (see `docs/RELEASING.md`).
