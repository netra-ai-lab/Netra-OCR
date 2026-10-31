"""Time the page pipeline, stage by stage.

    python scripts/benchmark.py                        # test_images/, layout on, NETRA_DEVICE or auto
    python scripts/benchmark.py --device cuda --decoder blockwise --no-layout
    python scripts/benchmark.py --pages some/dir --repeat 3

Prints seconds per page for each stage (layout model, line detector,
recognizer encoder and decoder, everything else), so a speed change shows
where it came from. One page is processed first to warm the models up.
"""

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", type=Path, default=Path("test_images"), help="Folder of page images or PDFs")
    parser.add_argument("--device", help="cpu, cuda, ... (default: NETRA_DEVICE, else auto)")
    parser.add_argument("--decoder", default="ar", choices=["ar", "blockwise"])
    parser.add_argument("--no-layout", action="store_true", help="Lines only, no layout model")
    parser.add_argument("--repeat", type=int, default=1, help="Passes over the pages")
    args = parser.parse_args()
    if args.device:
        os.environ["NETRA_DEVICE"] = args.device

    from netra_ocr import models
    from netra_ocr.ocr_engine import KhmerOCRPipeline

    pages = sorted(p for p in args.pages.iterdir() if p.is_file())
    pipe = KhmerOCRPipeline(layout=not args.no_layout, decoder=args.decoder)
    pipe.process_document(str(pages[0]))

    spent = defaultdict(float)
    counts = defaultdict(int)

    def timed(name, fn):
        def wrapper(*a, **k):
            start = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                spent[name] += time.perf_counter() - start
                counts[name] += 1
        return wrapper

    recognizer = models.load(args.decoder)
    recognizer.encoder.run = timed("encoder", recognizer.encoder.run)
    recognizer.step.run = timed("decoder", recognizer.step.run)
    if pipe.layout_analyzer is not None:
        pipe.layout_analyzer.analyze = timed("layout", pipe.layout_analyzer.analyze)
    pipe.detector.detect = timed("lines", pipe.detector.detect)

    n = len(pages) * args.repeat
    start = time.perf_counter()
    for _ in range(args.repeat):
        for page in pages:
            pipe.process_document(str(page))
    total = time.perf_counter() - start

    print(f"{recognizer!r}, {n} pages")
    print(f"{'stage':<10}{'s/page':>8}{'calls/page':>12}")
    for name in ("layout", "lines", "encoder", "decoder"):
        if name in spent:
            print(f"{name:<10}{spent[name] / n:>8.3f}{counts[name] / n:>12.1f}")
    print(f"{'other':<10}{(total - sum(spent.values())) / n:>8.3f}")
    print(f"{'total':<10}{total / n:>8.3f}")


if __name__ == "__main__":
    main()
