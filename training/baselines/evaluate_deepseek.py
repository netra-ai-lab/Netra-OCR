"""Evaluate DeepSeek-OCR (deepseek-ai/DeepSeek-OCR) against a local
{images/, labels/} benchmark directory (e.g. training/test_set/khob_eval).

Must be run under the separate `.venv-deepseek` environment (pinned
transformers==4.46.3) -- DeepSeek-OCR's remote-modeling code imports
`LlamaFlashAttention2`, a class removed by the main project venv's newer
transformers (needed by surya-ocr/Qwen2.5-VL). See CLAUDE.md for why.

Usage (run as a plain script, NOT `python -m ...` -- `-m` would import the
`netra_ocr` package's __init__ first, which pulls in cv2/ultralytics that
aren't installed in this venv):
    source .venv-deepseek/bin/activate
    python training/baselines/evaluate_deepseek.py \\
        --local-dir training/test_set/khob_eval --out-dir eval_out_khob_deepseek

Loads netra_ocr_train/metrics.py directly via importlib (bypassing the `netra_ocr`
package __init__, which pulls in cv2/ultralytics -- not installed in this
venv) so normalization/scoring stays identical to the other engines' scripts.
"""

import argparse
import importlib.util
import logging
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_COMMON_PATH = Path(__file__).resolve().parents[1] / "netra_ocr_train" / "metrics.py"
_spec = importlib.util.spec_from_file_location("ocr_eval_common", _COMMON_PATH)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
PROMPT = "<image>\nFree OCR."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="eval_out_deepseek")
    parser.add_argument("--model", default=MODEL_NAME,
                        help="HF repo id of a DeepSeek-OCR-architecture model "
                             "(e.g. KrorngAI/deepseek_ocr_Khmer_finetuned)")
    parser.add_argument("--label", default="deepseek-ocr",
                        help="Row label in the summary and prediction-CSV filename")
    args = parser.parse_args()

    image_paths, targets_raw = common.load_local_dataset(Path(args.local_dir))
    if args.limit:
        image_paths = image_paths[: args.limit]
        targets_raw = targets_raw[: args.limit]
    logger.info(f"Loaded {len(image_paths)} rows from {args.local_dir}")

    logger.info(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, use_safetensors=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()

    predictions = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, img_path in enumerate(image_paths):
        out = model.infer(
            tokenizer, prompt=PROMPT, image_file=str(img_path),
            output_path="/tmp/deepseek_ocr_scratch",
            base_size=1024, image_size=640, crop_mode=True,
            save_results=False, eval_mode=True,
        )
        predictions.append(out.strip() if out else "")
        if (i + 1) % 25 == 0:
            logger.info(f"  {i + 1}/{len(image_paths)}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n = len(image_paths)
    ms_per_image = 1000.0 * elapsed / n if n else 0.0
    lines_per_sec = n / elapsed if elapsed else 0.0

    metrics, records = common.score(targets_raw, predictions)
    metrics.update({"label": args.label, "ms_per_image": ms_per_image, "lines_per_sec": lines_per_sec})

    out_dir = Path(args.out_dir)
    common.write_records(records, out_dir / f"{args.label}_predictions.csv")
    common.print_summary([metrics])


if __name__ == "__main__":
    main()
