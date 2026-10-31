from dataclasses import dataclass, field
import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


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
