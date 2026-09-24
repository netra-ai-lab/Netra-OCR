import os
from dataclasses import dataclass, field
import torch


def _default_device() -> str:
    # NETRA_DEVICE lets a deployment pin the device (e.g. force "cpu" inside a
    # GPU image); otherwise use CUDA whenever it's available.
    return os.environ.get("NETRA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def describe_device() -> str:
    """Human-readable device for logs and the UI, e.g. "GPU (NVIDIA GeForce RTX 3060)"."""
    if _default_device().startswith("cuda") and torch.cuda.is_available():
        return f"GPU ({torch.cuda.get_device_name(0)})"
    if torch.version.cuda and not torch.cuda.is_available():
        return "CPU (no GPU visible)"
    return "CPU"


@dataclass
class OCRConfig:
    """Configuration for OCR Inference Pipeline."""
    img_height: int = 48
    chunk_width: int = 100
    chunk_overlap: int = 16
    emb_dim: int = 384
    max_seq_len: int = 4096
    decode_max_len: int = 512
    decoder_type: str = "ar"    # "ar" (plain autoregressive) or "blockwise" (Stern et al. 2018)
    block_size: int = 4         # only used when decoder_type == "blockwise"
    device: str = field(default_factory=_default_device)
