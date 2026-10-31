"""Write reference outputs from the Python side (netra_ocr.inference) for test/parity.mjs.

    python web/test/make_fixtures.py --models MODELS_DIR --lines LINE_IMAGES_DIR \
        --pages PAGE_IMAGES_DIR --out FIXTURES_DIR

MODELS_DIR holds the model folders (recognition/, recognition-blockwise/,
lines/, layout/), e.g. a copy of Netra-Lab/Netra-OCR. Images are decoded once
here (PIL) and stored as raw RGB, so the JS side is compared on identical
pixels -- image decoding itself is the browser's job.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from netra_ocr.inference import Detector, Recognizer

RECOGNIZERS = ("recognition", "recognition-blockwise")
DETECTORS = ("lines", "layout")


def save_rgb(image: Image.Image, path: Path) -> dict:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb.tofile(path)
    return {"file": path.name, "width": rgb.shape[1], "height": rgb.shape[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--lines", type=Path, required=True)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-lines", type=int, default=400)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lines = []
    paths = sorted(args.lines.iterdir())[:args.max_lines]
    recognizers = {name: Recognizer(args.models / name, device="cpu")
                   for name in RECOGNIZERS if (args.models / name).exists()}
    first = next(iter(recognizers.values()))
    for i, path in enumerate(paths):
        image = Image.open(path)
        entry = save_rgb(image, args.out / f"line_{i}.rgb")
        chunks = first.preprocess(image)
        chunks.tofile(args.out / f"line_{i}.chunks")
        entry["chunks"] = list(chunks.shape)
        for name, recognizer in recognizers.items():
            result = recognizer(image)
            entry[name] = {"text": result.text, "confidence": result.confidence}
        lines.append(entry)

    pages = []
    detectors = {name: Detector(args.models / name, device="cpu")
                 for name in DETECTORS if (args.models / name).exists()}
    for i, path in enumerate(sorted(args.pages.iterdir())):
        image = Image.open(path)
        entry = save_rgb(image, args.out / f"page_{i}.rgb")
        for name, detector in detectors.items():
            entry[name] = [{"box": list(d.box), "score": d.score, "classId": d.class_id}
                           for d in detector(np.asarray(image.convert("RGB")))]
        pages.append(entry)

    (args.out / "fixtures.json").write_text(json.dumps({"lines": lines, "pages": pages}, ensure_ascii=False))
    print(f"{len(lines)} lines, {len(pages)} pages -> {args.out}")


if __name__ == "__main__":
    main()
