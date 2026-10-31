"""Evaluate a third-party OCR engine (Surya OCR, Qwen2.5-VL, or Tesseract)
against a local {images/, labels/} benchmark directory (e.g.
test_set/khob_eval), using the same normalization/CER/WER/exact-match scoring
as training/evaluate.sh (see netra_ocr_train/metrics.py), so results are directly
comparable across engines.

Usage:
    python training/baselines/evaluate_vlm.py --engine surya \\
        --local-dir training/test_set/khob_eval --out-dir eval_out_khob_surya
    python training/baselines/evaluate_vlm.py --engine qwen \\
        --local-dir training/test_set/khob_eval --out-dir eval_out_khob_qwen
    python training/baselines/evaluate_vlm.py --engine tesseract \\
        --local-dir training/test_set/khob_eval --out-dir eval_out_khob_tess

The tesseract engine needs a tesseract binary (set TESSERACT_CMD if it isn't
on PATH) and khm/eng traineddata (set TESSDATA_PREFIX to their directory).

DeepSeek-OCR is NOT included here -- it needs a separate venv with an old
pinned transformers version (incompatible with this venv's Surya/Qwen2.5-VL
deps). See evaluate_deepseek.py.
"""

import argparse
import logging
import os
import re
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from netra_ocr_train.metrics import load_local_dataset, load_images, score, write_records, print_summary

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(html: str) -> str:
    return _HTML_TAG_RE.sub(" ", html)


def run_surya(images):
    """full_page=True runs surya's whole-document prompt, which -- on a crop
    that's already a single text line -- has no real layout to describe and
    tends to devolve into repetition loops (confirmed empirically: garbled,
    heavily-repeated output on khob_eval smoke-test rows). Instead, build one
    synthetic "Text" layout box covering the entire crop per image and use
    block mode (per-block BLOCK_PROMPT, meant for exactly this: OCR-ing one
    already-known text region), skipping surya's own layout model entirely
    since we already know each image is one block.
    """
    os.environ.setdefault("TORCH_DEVICE", "cuda")
    llama_bin = os.environ.get("LLAMA_CPP_BINARY")
    if llama_bin:
        os.environ.setdefault("LD_LIBRARY_PATH", str(Path(llama_bin).parent))

    from surya.inference import SuryaInferenceManager
    from surya.recognition import RecognitionPredictor
    from surya.layout.schema import LayoutBox, LayoutResult

    manager = SuryaInferenceManager(method="llamacpp")
    recognition_predictor = RecognitionPredictor(manager)

    layout_results = []
    for img in images:
        w, h = img.size
        box = LayoutBox(polygon=[0.0, 0.0, float(w), float(h)], label="Text",
                         raw_label="Text", position=0)
        layout_results.append(LayoutResult(bboxes=[box], image_bbox=[0.0, 0.0, float(w), float(h)]))

    t0 = time.perf_counter()
    page_results = recognition_predictor(images, layout_results=layout_results, full_page=False)
    elapsed = time.perf_counter() - t0

    predictions = []
    for page in page_results:
        text = " ".join(_strip_html(b.html).strip() for b in page.blocks if b.html)
        predictions.append(" ".join(text.split()))
    manager.stop()
    return predictions, elapsed


def run_qwen(images, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    prompt_text = "Read all the text in this image. Output only the transcribed text, nothing else."
    predictions = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, img in enumerate(images):
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                            padding=True, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        out_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        predictions.append(out_text.strip())
        if (i + 1) % 25 == 0:
            logger.info(f"  {i + 1}/{len(images)}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return predictions, elapsed


def run_tesseract(images):
    """Tesseract 5 with the official khm (+eng) traineddata, --psm 7 (treat
    the image as a single text line) since every benchmark image is a
    pre-cropped line -- letting tesseract's own page segmentation run on
    these crops would only add spurious splits."""
    import pytesseract

    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

    predictions = []
    t0 = time.perf_counter()
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang="khm+eng", config="--psm 7")
        predictions.append(text.strip())
        if (i + 1) % 100 == 0:
            logger.info(f"  {i + 1}/{len(images)}")
    elapsed = time.perf_counter() - t0
    return predictions, elapsed


ENGINES = {"surya": run_surya, "qwen": run_qwen, "tesseract": run_tesseract}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=list(ENGINES))
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="eval_out_vlm")
    args = parser.parse_args()

    image_paths, targets_raw = load_local_dataset(Path(args.local_dir))
    if args.limit:
        image_paths = image_paths[: args.limit]
        targets_raw = targets_raw[: args.limit]
    logger.info(f"Loaded {len(image_paths)} rows from {args.local_dir}")
    images = load_images(image_paths)

    logger.info(f"Running engine={args.engine} ...")
    predictions, elapsed = ENGINES[args.engine](images)

    n = len(images)
    ms_per_image = 1000.0 * elapsed / n if n else 0.0
    lines_per_sec = n / elapsed if elapsed else 0.0

    metrics, records = score(targets_raw, predictions)
    metrics.update({"label": args.engine, "ms_per_image": ms_per_image, "lines_per_sec": lines_per_sec})

    out_dir = Path(args.out_dir)
    write_records(records, out_dir / f"{args.engine}_predictions.csv")
    print_summary([metrics])


if __name__ == "__main__":
    main()
