"""Text-line recognition on ONNX Runtime (no PyTorch).

The model is ``recognition`` (decoder="ar") or ``recognition-blockwise``
(decoder="blockwise") from ``netra_ocr.models``, downloaded once from Hugging
Face. Both decode greedily and give the same text; blockwise proposes several
tokens per decoder pass. The model code itself is in ``netra_ocr.inference``.
"""

import argparse

from .. import models


def _recognizer(decoder: str, bundle: str | None):
    if bundle is not None:
        from ..inference import Recognizer
        return Recognizer(bundle, device=models.device())
    if decoder not in models.DECODERS:
        raise ValueError(f"Unknown decoder '{decoder}'. Available: {list(models.DECODERS)}")
    return models.load(decoder)


def recognize(image_input, decoder: str = "ar", bundle: str | None = None) -> str:
    """Recognize one text-line image (a path, PIL Image or RGB array).

    decoder: "ar" (default) or "blockwise".
    bundle: a recognizer model folder or ``hf://`` source, instead of the default.
    """
    return _recognizer(decoder, bundle)(image_input).text


def recognize_batch(image_list: list, batch_size: int = 16, decoder: str = "ar",
                    bundle: str | None = None) -> list:
    """Recognize many text-line images, ``batch_size`` lines per model call."""
    if not image_list:
        return []
    return [result.text for result in _recognizer(decoder, bundle).batch(image_list, batch_size=batch_size)]


def main():
    parser = argparse.ArgumentParser(description="Recognize one Khmer/English text-line image")
    parser.add_argument("--image", required=True, help="Path to a text-line image")
    parser.add_argument("--decoder", default="ar", choices=list(models.DECODERS))
    parser.add_argument("--bundle", default=None, help="Recognizer model folder or hf:// source (default: Netra-Lab/Netra-OCR)")
    parser.add_argument("--output", help="Save the text to this file")
    args = parser.parse_args()

    text = recognize(args.image, args.decoder, args.bundle)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
