"""Short LoRA fine-tune of a general-purpose VLM (Qwen2.5-VL-3B or
DeepSeek-OCR) on this project's own Khmer line-crop training data, so the
`test_set/` comparison against our purpose-built model is fair -- both VLMs
scored CER > 1.0 zero-shot because they have no Khmer training coverage at
all (see OCR_COMPARISON_RESULTS.md).

Must be run under `.venv-unsloth` (unsloth + transformers==4.56.2 +
trl==0.22.2, the pins from unsloth's own DeepSeek-OCR notebook). Run as a
plain script, not `-m` (the netra_ocr package __init__ imports cv2 etc.,
which aren't installed there):

    source .venv-unsloth/bin/activate
    python training/baselines/finetune_vlm_khmer.py --model qwen
    python training/baselines/finetune_vlm_khmer.py --model deepseek \\
        --deepseek-dir training/weights/deepseek_ocr_base

Training data: an even sample from seanghay/khmer-hanuman-100k plus the two
Netra-Lab/* synthetic sets (all already in .hf_cache, symlinked into
~/.cache/huggingface) -- KhmerSynthetic1M is deliberately excluded
(research-only license). Recipes follow unsloth's DeepSeek-OCR notebook
(LoRA r=16, lr 2e-4, adamw_8bit, linear schedule) and its vision-notebook
equivalent for Qwen.

DeepSeek needs a local snapshot of unsloth/DeepSeek-OCR (patched remote code
that works on transformers 4.56.x, unlike deepseek-ai's original which pins
4.46) whose directory is importable as `deepseek_ocr` -- see --deepseek-dir.
The big safetensors file can be hardlinked from the deepseek-ai HF cache
instead of re-downloading (identical weights).
"""

import argparse
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QWEN_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_PROMPT = "Read all the text in this image. Output only the transcribed text, nothing else."
DEEPSEEK_PROMPT = "<image>\nFree OCR. "

TRAIN_SOURCES = [  # (repo_id, text_column)
    ("seanghay/khmer-hanuman-100k", "text"),
    ("Netra-Lab/khmer-scene-text-synthetic-contrast", "label"),
    ("Netra-Lab/khmer-document-synthetic-low-res", "label"),
]


def load_training_rows(num_samples: int, seed: int):
    """Returns a list of {"image": PIL.Image, "label": str}, sampled evenly
    across TRAIN_SOURCES. Materialized lazily via per-row indexing so we never
    decode more images than requested."""
    from datasets import load_dataset

    rng = random.Random(seed)
    per_source = num_samples // len(TRAIN_SOURCES)
    picks = []  # (dataset, row_idx, text_column)
    for repo_id, text_column in TRAIN_SOURCES:
        ds = load_dataset(repo_id, split="train")
        idxs = rng.sample(range(len(ds)), min(per_source, len(ds)))
        picks.extend((ds, i, text_column) for i in idxs)
        logger.info(f"{repo_id}: sampled {len(idxs)} of {len(ds)} rows")
    rng.shuffle(picks)
    return picks


class LazyConversationDataset:
    """Torch-style dataset producing {"messages": ...} lazily (images are
    decoded per __getitem__, not up front)."""

    def __init__(self, picks, build_messages):
        self.picks = picks
        self.build_messages = build_messages

    def __len__(self):
        return len(self.picks)

    def __getitem__(self, i):
        ds, idx, text_column = self.picks[i]
        row = ds[idx]
        image = row["image"].convert("RGB")
        label = str(row[text_column]).strip()
        return {"messages": self.build_messages(image, label)}


def qwen_messages(image, label):
    return [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": QWEN_PROMPT},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": label}]},
    ]


def deepseek_messages(image, label):
    return [
        {"role": "<|User|>", "content": DEEPSEEK_PROMPT, "images": [image]},
        {"role": "<|Assistant|>", "content": label},
    ]


def build_qwen(args):
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        QWEN_ID,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0, bias="none",
        random_state=args.seed,
    )
    from unsloth.trainer import UnslothVisionDataCollator
    collator = UnslothVisionDataCollator(model, tokenizer)
    return model, tokenizer, collator, qwen_messages


def build_deepseek(args):
    import os

    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
    from unsloth import FastVisionModel
    from transformers import AutoModel

    ds_dir = Path(args.deepseek_dir).resolve()
    if not (ds_dir / "modeling_deepseekocr.py").exists():
        raise SystemExit(f"--deepseek-dir {ds_dir} is not a DeepSeek-OCR snapshot")
    # The unsloth-notebook collator imports `deepseek_ocr.modeling_deepseekocr`;
    # make the snapshot importable under that exact package name.
    if ds_dir.name != "deepseek_ocr":
        alias = ds_dir.parent / "deepseek_ocr"
        if not alias.exists():
            alias.symlink_to(ds_dir)
        ds_dir = alias
    sys.path.insert(0, str(ds_dir.parent))

    model, tokenizer = FastVisionModel.from_pretrained(
        str(ds_dir),
        load_in_4bit=False,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        r=16, lora_alpha=16, lora_dropout=0, bias="none",
        random_state=args.seed,
    )
    from deepseek_ocr_collator import DeepSeekOCRDataCollator
    collator = DeepSeekOCRDataCollator(
        tokenizer=tokenizer, model=model,
        image_size=640, base_size=1024, crop_mode=True,
        train_on_responses_only=True,
    )
    return model, tokenizer, collator, deepseek_messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["qwen", "deepseek"])
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out-dir", default=None,
                        help="LoRA output dir (default training/weights/lora_<model>_khmer)")
    parser.add_argument("--deepseek-dir", default="training/weights/deepseek_ocr_base",
                        help="Local unsloth/DeepSeek-OCR snapshot (deepseek only)")
    args = parser.parse_args()

    out_dir = args.out_dir or f"training/weights/lora_{args.model}_khmer"

    # Import unsloth before torch/transformers usage so its patches apply.
    if args.model == "qwen":
        model, tokenizer, collator, build_messages = build_qwen(args)
    else:
        model, tokenizer, collator, build_messages = build_deepseek(args)

    picks = load_training_rows(args.num_samples, args.seed)
    train_dataset = LazyConversationDataset(picks, build_messages)
    logger.info(f"Training on {len(train_dataset)} sampled rows")

    import torch
    from transformers import Trainer, TrainingArguments
    from unsloth import FastVisionModel, is_bf16_supported

    FastVisionModel.for_training(model)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=10,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=args.seed,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            output_dir=str(Path(out_dir) / "trainer_out"),
            report_to="none",
            dataloader_num_workers=2,
            remove_unused_columns=False,  # required for vision fine-tuning
        ),
    )
    stats = trainer.train()
    logger.info(f"train_runtime={stats.metrics['train_runtime']:.0f}s "
                f"loss={stats.metrics.get('train_loss'):.4f}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info(f"Saved LoRA to {out_dir}")


if __name__ == "__main__":
    main()
