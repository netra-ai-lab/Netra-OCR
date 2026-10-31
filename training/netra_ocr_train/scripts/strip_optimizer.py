"""Strip a training checkpoint down to a publish/inference-ready weight file.

train.py saves resumable checkpoints: model_state_dict + optimizer_state_dict
+ epoch + meta. The optimizer state (Adam's per-parameter momentum/variance
buffers) is only needed to *resume training* and is often larger than the
model itself -- e.g. the cluster-AR checkpoint is 231 MB, of which 154 MB is
optimizer state. Inference (predictor.py / recognize_text.py) never touches it.

This drops the optimizer state (and epoch), keeping model_state_dict and meta
(meta carries decoder_type/block_size, which autodetect_config reads for the
blockwise decoder), so the result is small enough to commit to git while
loading identically for inference.

Usage:
    python -m netra_ocr_train.scripts.strip_optimizer \\
        --src training/weights/khmerocr_se_ar_last.pth \\
        --dst training/weights/khmerocr_cluster_ar.pth
"""

import argparse
from pathlib import Path

import torch


def strip(src: str, dst: str):
    ckpt = torch.load(src, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        # Already a raw/inference state_dict -- nothing to strip.
        out = ckpt
    else:
        out = {"model_state_dict": ckpt["model_state_dict"]}
        if "meta" in ckpt:
            out["meta"] = ckpt["meta"]
    torch.save(out, dst)
    src_mb = Path(src).stat().st_size / 1e6
    dst_mb = Path(dst).stat().st_size / 1e6
    print(f"{Path(src).name} ({src_mb:.0f} MB) -> {Path(dst).name} ({dst_mb:.0f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Training checkpoint (.pth) to strip")
    parser.add_argument("--dst", required=True, help="Output inference-only checkpoint (.pth)")
    args = parser.parse_args()
    strip(args.src, args.dst)


if __name__ == "__main__":
    main()
