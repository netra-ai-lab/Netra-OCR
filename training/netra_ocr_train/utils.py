import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

def autodetect_config(model_path: str | Path) -> dict:
    """
    Peeks into the .pth checkpoint to infer model dimensions.
    Returns a dictionary of overrides for OCRConfig.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")

    logger.info(f"Inspecting checkpoint: {path.name}...")
    # Load on CPU just to check shapes (fast)
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    detected = {}

    # 1. Detect Embedding Dim & Encoder Seq Len from 'global_pos'
    if 'global_pos' in state_dict:
        shape = state_dict['global_pos'].shape
        detected['max_seq_len'] = shape[0]
        detected['emb_dim'] = shape[1]
        logger.info(f"   ↳ Auto-detected: emb_dim={shape[1]}, max_seq_len={shape[0]}")

    # 2. Detect decoder type + block size. Prefer the checkpoint's own 'meta'
    # dict (present on the cluster_ar / cluster_blockwise checkpoints); fall
    # back to sniffing state_dict key prefixes for older checkpoints without
    # 'meta' (plain AR decoders store weights under 'dec.*' directly, while a
    # BlockwiseParallelWrapper nests the frozen base decoder under 'dec.base.*').
    meta = checkpoint.get('meta', {}) if isinstance(checkpoint, dict) else {}
    if 'decoder_type' in meta:
        detected['decoder_type'] = meta['decoder_type']
        if 'block_size' in meta:
            detected['block_size'] = meta['block_size']
    elif any(k.startswith('dec.base.') for k in state_dict):
        detected['decoder_type'] = 'blockwise'
    else:
        detected['decoder_type'] = 'ar'
    logger.info(f"   ↳ Auto-detected: decoder_type={detected['decoder_type']}"
                + (f", block_size={detected.get('block_size')}" if detected['decoder_type'] == 'blockwise' else ""))

    # 3. Detect Decoder Max Length (position embedding is 'dec.pos_emb' for a
    # plain AR decoder, or 'dec.base.pos_emb' when wrapped for blockwise decoding).
    pos_emb_key = 'dec.base.pos_emb' if detected['decoder_type'] == 'blockwise' else 'dec.pos_emb'
    if pos_emb_key in state_dict:
        shape = state_dict[pos_emb_key].shape
        detected['decode_max_len'] = shape[0]
        logger.info(f"   ↳ Auto-detected: decode_max_len={shape[0]}")

    return detected