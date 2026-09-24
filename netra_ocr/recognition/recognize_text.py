import argparse
import logging
import os
from .config import OCRConfig
from .utils import setup_logging, autodetect_config
from .cluster_tokenizer import ClusterTokenizer
from .predictor import OCRPredictor
from .model.model import KhmerOCR

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# GLOBAL SETTINGS & STATE
# ==============================================================================
# The default recognition weight is *not* bundled with the package (it's
# ~77 MB) -- it's downloaded on first use from the HF model repo and cached
# locally by huggingface_hub (~/.cache/huggingface by default), so `pip
# install netra-ocr` stays small. Only the tiny vocab JSON is bundled, since
# it's needed immediately with no network round-trip.
DEFAULT_MODEL_REPO = "Darayut/khmer-text-recognition"
# "ar" (plain autoregressive, one token/step) and "blockwise" (Stern et al.
# 2018 blockwise-parallel decoding -- usually several tokens/step, same
# output as "ar") both share the same vocab.
DECODER_FILENAMES = {
    "ar": "khmerocr_cluster_ar.pth",
    "blockwise": "khmerocr_cluster_blockwise.pth",
}
DEFAULT_VOCAB_PATH = os.path.join(CURRENT_DIR, "char2idx_cluster.json")

# Predictor cache, keyed by (model_path, vocab_path) -- lets ar/blockwise (or
# any other explicit override) be swapped between without reloading a model
# that's already been loaded once.
_PREDICTOR_CACHE: dict = {}


def _default_model_path(decoder: str) -> str:
    from huggingface_hub import hf_hub_download
    if decoder not in DECODER_FILENAMES:
        raise ValueError(f"Unknown decoder '{decoder}'. Available: {list(DECODER_FILENAMES)}")
    return hf_hub_download(repo_id=DEFAULT_MODEL_REPO, filename=DECODER_FILENAMES[decoder])


def _get_predictor(model_path=None, vocab_path=None, decoder: str = "ar"):
    """
    Internal function to load a given (model, vocab) pair only once.
    """
    # Resolving the default model path may download it from the Hub, so this
    # only runs when the (model_path, vocab_path) pair hasn't been loaded yet
    # (guarded by the cache lookup below).
    resolved_model_path = model_path or _default_model_path(decoder)
    resolved_vocab_path = vocab_path or DEFAULT_VOCAB_PATH

    cache_key = (resolved_model_path, resolved_vocab_path)
    if cache_key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[cache_key]

    # Load failures propagate to the caller (no sys.exit): a long-running
    # process such as the web server must be able to report and survive them.
    detected_cfg = autodetect_config(resolved_model_path)
    config = OCRConfig(**detected_cfg)
    tokenizer = ClusterTokenizer(resolved_vocab_path)

    predictor = OCRPredictor(
        model_path=resolved_model_path,
        tokenizer=tokenizer,
        config=config,
        model_class=KhmerOCR
    )
    _PREDICTOR_CACHE[cache_key] = predictor
    return predictor

# ==============================================================================
# PUBLIC API
# ==============================================================================

def recognize(image_input, beam_width: int = 3, model_path=None, vocab_path=None, decoder: str = "ar") -> str:
    """
    Recognizes text from an image path.

    Args:
        image_path (str): Path to the image file.
        beam_width (int): Beam search width (default 3). Ignored when the
            resolved model uses blockwise decoding (always greedy-equivalent).
        model_path (str): Optional override for model path.
        vocab_path (str): Optional override for vocab path.
        decoder (str): "ar" (default) or "blockwise". Only used to pick the
            default model when model_path is not given.

    Returns:
        str: The predicted text.
    """
    predictor = _get_predictor(model_path, vocab_path, decoder)
    try:
        # Most predictors can handle PIL images. If yours requires a path,
        # we'd need to modify predictor.py, but let's assume it handles objects.
        result_text = predictor.predict(image_input, beam_width=beam_width)
        return result_text
    except Exception:
        logger.exception("Prediction error")
        return ""

def recognize_batch(image_list: list, beam_width: int = 1, batch_size: int = 8, model_path=None, vocab_path=None, decoder: str = "ar") -> list:
    if not image_list:
        return []

    predictor = _get_predictor(model_path, vocab_path, decoder)
    try:
        # Pass the batch_size to the predictor
        return predictor.predict_batch(image_list, beam_width=beam_width, batch_size=batch_size)
    except Exception:
        logger.exception("Batch prediction error; retrying one image at a time")
        return [recognize(img, beam_width, model_path, vocab_path, decoder) for img in image_list]

# ===================
# CLI ENTRY POINT
# ===================
def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Khmer OCR Inference Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=None, help="Path to .pth (default: auto-download from the HF model repo)")
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB_PATH, help="Path to vocab json")
    parser.add_argument("--decoder", type=str, default="ar", choices=list(DECODER_FILENAMES),
                        help="Which default checkpoint to download when --model isn't given (default: ar)")
    parser.add_argument("--beam", type=int, default=3, help="Beam width (1 for greedy). Ignored for blockwise decoding.")
    parser.add_argument("--output", type=str, help="Save result to text file")

    args = parser.parse_args()

    # Call the API function
    text = recognize(args.image, args.beam, args.model, args.vocab, args.decoder)
    
    print("\n" + "="*40)
    print(f"RESULT: {text}")
    print("="*40 + "\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

    """
    USAGE EXAMPLE:

        python recognize_text.py --image  "test_img_1.png"

        python recognize_text.py \
        --image "sample.png" \
        --model "weight/model.pth" \
        --vocab "char2idx.json" \
        --beam 5 \
        --output "results/sample_output.txt"
        
    ARGUMENTS:
        --image: Path to the input image (Required).
        --model: Path to .pth file (Default: auto-download from the HF model repo).
        --vocab: Path to .json vocab (Default: char2idx_cluster.json).
        --beam: Beam width. Set to 1 for Greedy Search (Default: 3).
        --output: (Optional) Text file to save the result.
    
    RUN VIA PYTHON:
        from recognize_text import recognize

        # Basic usage (uses defaults defined in the file)
        text = recognize("test_image_1.png")
        print(text)

        # Processing multiple images (Model stays loaded!)
        images = ["img1.png", "img2.png", "img3.png"]
        for img in images:
            print(f"{img}: {recognize(img)}")

        # Override settings if needed
        text_custom = recognize("test_image.png", beam_width=5, model_path="other_model.pth")
    """