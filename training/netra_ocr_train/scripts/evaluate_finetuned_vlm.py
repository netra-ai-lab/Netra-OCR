"""Evaluate a finetune_vlm_khmer.py LoRA (Qwen2.5-VL-3B or DeepSeek-OCR)
against a local {images/, labels/} benchmark directory, with the same
normalization/scoring as every other engine (ocr_eval_common.py).

Must run under `.venv-unsloth`, as a plain script (not `-m`):

    source .venv-unsloth/bin/activate
    python training/netra_ocr_train/scripts/evaluate_finetuned_vlm.py \\
        --model qwen --lora training/weights/lora_qwen_khmer \\
        --local-dir training/test_set/khob_eval --out-dir eval_out_khob_qwen_ft
"""

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_COMMON_PATH = Path(__file__).resolve().parent / "ocr_eval_common.py"
_spec = importlib.util.spec_from_file_location("ocr_eval_common", _COMMON_PATH)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

QWEN_PROMPT = "Read all the text in this image. Output only the transcribed text, nothing else."
DEEPSEEK_PROMPT = "<image>\nFree OCR. "


def run_qwen(lora_dir, image_paths):
    import torch
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        lora_dir, load_in_4bit=False, use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)

    images = common.load_images(image_paths)
    predictions = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, img in enumerate(images):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": QWEN_PROMPT}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=[img], text=[text], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        trimmed = out[0][inputs.input_ids.shape[1]:]
        predictions.append(processor.decode(trimmed, skip_special_tokens=True).strip())
        if (i + 1) % 25 == 0:
            logger.info(f"  {i + 1}/{len(images)}")
    torch.cuda.synchronize()
    return predictions, time.perf_counter() - t0


def run_deepseek(lora_dir, image_paths, deepseek_dir):
    import os

    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
    import torch
    from unsloth import FastVisionModel
    from transformers import AutoModel

    # The LoRA adapter_config points at the base snapshot; make sure its
    # remote code package resolves the same way training did.
    sys.path.insert(0, str(Path(deepseek_dir).resolve().parent))

    model, tokenizer = FastVisionModel.from_pretrained(
        lora_dir, load_in_4bit=False, auto_model=AutoModel,
        trust_remote_code=True, unsloth_force_compile=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    infer_model = model
    if not hasattr(infer_model, "infer"):  # PEFT wrapper hides base methods
        infer_model = model.merge_and_unload()

    predictions = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, img_path in enumerate(image_paths):
        out = infer_model.infer(
            tokenizer, prompt=DEEPSEEK_PROMPT, image_file=str(img_path),
            output_path="/tmp/deepseek_ocr_ft_scratch",
            base_size=1024, image_size=640, crop_mode=True,
            save_results=False, eval_mode=True,
        )
        predictions.append(out.strip() if out else "")
        if (i + 1) % 25 == 0:
            logger.info(f"  {i + 1}/{len(image_paths)}")
    torch.cuda.synchronize()
    return predictions, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["qwen", "deepseek"])
    parser.add_argument("--lora", required=True, help="LoRA dir from finetune_vlm_khmer.py")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="eval_out_finetuned")
    parser.add_argument("--label", default=None)
    parser.add_argument("--deepseek-dir", default="training/weights/deepseek_ocr_base")
    args = parser.parse_args()

    label = args.label or f"{args.model}-khmer-ft"

    image_paths, targets_raw = common.load_local_dataset(Path(args.local_dir))
    if args.limit:
        image_paths = image_paths[: args.limit]
        targets_raw = targets_raw[: args.limit]
    logger.info(f"Loaded {len(image_paths)} rows from {args.local_dir}")

    if args.model == "qwen":
        predictions, elapsed = run_qwen(args.lora, image_paths)
    else:
        predictions, elapsed = run_deepseek(args.lora, image_paths, args.deepseek_dir)

    n = len(image_paths)
    metrics, records = common.score(targets_raw, predictions)
    metrics.update({
        "label": label,
        "ms_per_image": 1000.0 * elapsed / n if n else 0.0,
        "lines_per_sec": n / elapsed if elapsed else 0.0,
    })
    common.write_records(records, Path(args.out_dir) / f"{label}_predictions.csv")
    common.print_summary([metrics])


if __name__ == "__main__":
    main()
