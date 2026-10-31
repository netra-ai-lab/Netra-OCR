"""Score recognition models on test sets: CER, WER, exact match and speed.

Usually run through ``bash training/evaluate.sh`` (from the repository root):

    bash training/evaluate.sh published                          # the model the app ships
    bash training/evaluate.sh training/runs/my-run/best.pth      # a training checkpoint
    bash training/evaluate.sh published my-run/best.pth --test my_test_lines/

A model is one of:
  published             the app's own ONNX recognizer, exactly as netra_ocr runs it
  path/to/folder        an exported model folder (training/export.sh), run the same way
  path/to/model.pth     a PyTorch checkpoint, decoded greedily (--beam N for beam search)

A test set is any data spec (netra_ocr_train/data.py): a preset (khob,
printed-words), a folder (images/ + labels/, or images with a .txt beside
each), a manifest (.tsv/.csv/.jsonl) or a Hugging Face dataset id. Default:
khob, the 325-line KHOB benchmark (Netra-Lab/khob-eval).

A progress bar per model and test set shows the running CER.

Text is compared after the same normalization for every model
(netra_ocr_train/metrics.py), so the numbers are comparable across runs.
Writes one predictions CSV per model and test set, plus summary.json, to --out.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

from ..metrics import score, write_records
from ..utils import HF_CACHE_DIR, RUNS_DIR, setup_logging, use_hf_cache

logger = logging.getLogger(__name__)

DEFAULT_TEST = "khob"
BATCH = 16


class _OnnxModel:
    """The app's recognizer (netra_ocr.inference) on an exported folder or the published model."""

    def __init__(self, spec: str, device: str | None):
        from netra_ocr import models
        from netra_ocr.inference import Recognizer
        source = models.source("ar") if spec == "published" else spec
        self.recognizer = Recognizer(source, device=device or "auto")

    def __call__(self, images):
        return [r.text for r in self.recognizer.batch(images, batch_size=16)]


class _TorchModel:
    """A PyTorch checkpoint through OCRPredictor."""

    def __init__(self, spec: str, vocab: str, device: str | None, beam: int):
        import torch
        from ..cluster_tokenizer import ClusterTokenizer
        from ..config import OCRConfig
        from ..model.model import KhmerOCR
        from ..predictor import OCRPredictor
        from ..utils import autodetect_config, resolve_checkpoint
        path = resolve_checkpoint(spec)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = OCRPredictor(model_path=path, tokenizer=ClusterTokenizer(vocab),
                                      config=OCRConfig(**autodetect_config(path), device=device),
                                      model_class=KhmerOCR)
        self.beam = beam
        self.cuda = device.startswith("cuda")

    def __call__(self, images):
        out = self.predictor.predict_batch(images, beam_width=self.beam, batch_size=8, progress=False)
        if self.cuda:
            import torch
            torch.cuda.synchronize()
        return out


def load_model(spec: str, vocab: str, device: str | None, beam: int):
    if spec == "published" or Path(spec).is_dir():
        return _OnnxModel(spec, device)
    return _TorchModel(spec, vocab, device, beam)


def _name(spec: str) -> str:
    path = Path(spec)
    if spec == "published" or not path.exists():
        return spec.replace("/", "_")
    return f"{path.parent.name}_{path.stem}" if path.is_file() else path.name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help='"published", an exported model folder, or a .pth checkpoint')
    parser.add_argument("--test", nargs="+", default=[DEFAULT_TEST],
                        help="Test sets: presets, folders, manifests or Hugging Face ids (default: khob)")
    parser.add_argument("--limit", type=int, help="Score only the first N lines of each test set")
    parser.add_argument("--beam", type=int, default=1, help="Beam width for .pth checkpoints (default 1: greedy, as the app)")
    parser.add_argument("--device", help="cpu, cuda, ... (default: GPU if available)")
    parser.add_argument("--vocab", default=str(Path(__file__).resolve().parents[1] / "char2idx_cluster.json"),
                        help="Vocabulary of the .pth checkpoints")
    parser.add_argument("--out", help="Results folder (default: training/runs/eval-<date-time>)")
    parser.add_argument("--data-dir", default=str(HF_CACHE_DIR), help="Hugging Face cache for test sets")
    args = parser.parse_args(argv)

    setup_logging()
    use_hf_cache(args.data_dir)
    from ..data import DataError, load_lines

    tests = args.test
    out = Path(args.out or RUNS_DIR / f"eval-{time.strftime('%Y%m%d-%H%M%S')}")

    sets = {}
    for spec in tests:
        try:
            rows = load_lines(spec)
        except DataError as e:
            raise SystemExit(f"\n{e}") from None
        n = min(len(rows), args.limit or len(rows))
        sets[spec] = ([rows[i]["image"] for i in tqdm(range(n), desc=f"reading {spec}", unit="line",
                                                       leave=False, disable=None)],
                      [rows[i]["label"] for i in range(n)])

    results = []
    for model_spec in args.models:
        logger.info(f"Loading {model_spec}")
        model = load_model(model_spec, args.vocab, args.device, args.beam)
        model(sets[tests[0]][0][:2])   # warm up, so the timing is steady-state
        for test_spec, (images, labels) in sets.items():
            predictions, elapsed, edits, chars = [], 0.0, 0, 0
            bar = tqdm(total=len(images), desc=f"{model_spec} on {Path(test_spec).name}", unit="line",
                       dynamic_ncols=True, disable=None)
            for i in range(0, len(images), BATCH):
                started = time.perf_counter()
                batch = model(images[i:i + BATCH])
                elapsed += time.perf_counter() - started
                predictions += batch
                so_far, _ = score(labels[i:i + BATCH], batch)
                edits, chars = edits + so_far["char_edits"], chars + so_far["chars"]
                bar.update(len(batch))
                bar.set_postfix(CER=f"{100 * edits / max(1, chars):.2f}%")
            bar.close()
            metrics, records = score(labels, predictions)
            metrics.update(model=model_spec, test=test_spec, ms_per_line=1000 * elapsed / max(1, len(images)))
            results.append(metrics)
            write_records(records, out / f"{_name(model_spec)}__{_name(test_spec)}.csv")

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    width = max(len(r["model"]) for r in results)
    lines = [f"{'model':<{width}}  {'test set':<28} {'lines':>6} {'CER':>7} {'WER':>7} {'exact':>7} {'ms/line':>8}"]
    for r in results:
        lines.append(f"{r['model']:<{width}}  {Path(r['test']).name[:28]:<28} {r['n']:>6} {100 * r['cer']:>6.2f}% "
                     f"{100 * r['wer']:>6.2f}% {100 * r['exact_match']:>6.2f}% {r['ms_per_line']:>8.1f}")
    print("\n" + "\n".join(lines) + f"\n\nPredictions and summary.json in {out}/")
    return results


if __name__ == "__main__":
    main()
