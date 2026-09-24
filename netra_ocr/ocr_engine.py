# ocr_engine.py
"""
Khmer OCR Pipeline — pluggable text detector, plain-text output.

Detection
---------
Detectors are registered in netra_ocr/detectors/__init__.py and selected by name.
Currently available:
    yolo       (default) — YOLOv2.6s trained on Khmer documents (class 0: text, class 1: logo)
    tesseract            — KhmerLineDetector built on Tesseract + graph clustering

Add a new detector by:
  1. Implementing BaseTextDetector in netra_ocr/detectors/<name>.py
  2. Adding it to DETECTOR_REGISTRY in detectors/__init__.py

Output formats (auto-detected from extension)
---------------------------------------------
  .txt  .md    — plain UTF-8 text (text lines only)
  .json         — structured metadata (image size, per-line text + bbox)
  .docx         — Word document: text lines as paragraphs, logos as inline images
"""

import os
import shutil
import argparse
import sys
from typing import Callable

import cv2
from PIL import Image

from .detectors import get_detector
from .document import AssetStore, Document
from .io import count_pages, iter_pages, sniff_kind
from .recognition.recognize_text import recognize_batch
from .output_formatters import save_output, SUPPORTED_FORMATS
from . import structure


class KhmerOCRPipeline:
    def __init__(self, detector: str = "yolo", conf: float | None = None,
                 pad: int | None = None, decoder: str = "ar", layout: bool = False,
                 layout_conf: float = 0.2):
        """
        layout: run DocLayout-YOLO page layout analysis (titles, paragraphs,
            tables, figures, headers/footers, reading order) on top of line
            detection. Needs the 'layout' extra. Only affects
            ``process_document`` and, when True, ``process_image``.
        """
        print(f"Initializing detector: {detector}")
        kwargs = {}
        if conf is not None:
            kwargs["conf"] = conf
        if pad is not None:
            kwargs["pad"] = pad
        self.detector = get_detector(detector, **kwargs)
        self.decoder = decoder
        self.layout = layout
        self.layout_analyzer = None
        if layout:
            from .layout.doclayout import DocLayoutYoloAnalyzer
            self.layout_analyzer = DocLayoutYoloAnalyzer(conf=layout_conf)

    # ── structured documents (layout-aware, multi-page) ────────────────
    def analyze_page(self, img_bgr, page_index: int = 0, assets: AssetStore | None = None,
                     beam_width: int = 1, batch_size: int = 8,
                     include_headers_footers: bool = True) -> tuple[dict, list]:
        """Detect, recognize and structure one page (BGR array).

        Returns ``(page, blocks)`` in the ``netra_ocr.document`` schema; crops
        and the page render are added to ``assets``.
        """
        assets = assets if assets is not None else AssetStore()
        h, w = img_bgr.shape[:2]
        page_asset = assets.put(f"page-{page_index}",
                                Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
        page = {"index": page_index, "width": w, "height": h, "image": page_asset}

        detected = self.detector.detect(img_bgr)
        text_lines = [dl for dl in detected if dl.label == "text"]
        line_boxes, line_crops = structure.consolidate_lines(
            img_bgr, [tuple(int(v) for v in dl.bbox) for dl in text_lines], [dl.crop for dl in text_lines])
        logo_boxes = [tuple(int(v) for v in dl.bbox) for dl in detected if dl.label == "logo"]

        if self.layout_analyzer is None:
            texts = self._recognize(line_crops, beam_width, batch_size)
            return page, structure.plain_blocks(page_index, img_bgr, line_boxes, texts, logo_boxes, assets)

        regions = self.layout_analyzer.analyze(img_bgr)
        units = structure.plan_page((w, h), line_boxes, logo_boxes, regions, include_headers_footers,
                                    ink=structure.ink_mask(img_bgr))
        whole_crops = structure.region_crops(img_bgr, units)
        texts = self._recognize(line_crops + whole_crops, beam_width, batch_size)
        line_texts, whole_texts = texts[:len(line_boxes)], texts[len(line_boxes):]
        blocks = structure.build_blocks(page_index, img_bgr, units, line_boxes, line_texts, whole_texts, assets)
        return page, blocks

    def process_document(self, source, beam_width: int = 1, batch_size: int = 8,
                         include_headers_footers: bool = True, max_pages: int | None = None,
                         progress: Callable[[int, int], None] | None = None) -> Document:
        """Run OCR over every page of an image, multi-page TIFF or PDF.

        ``source`` is a file path or the file's bytes. ``progress(done, total)``
        is called after each page.
        """
        data = source if isinstance(source, bytes) else open(source, "rb").read()
        total = count_pages(data)
        if max_pages is not None:
            total = min(total, max_pages)
        doc = Document()
        for i, img_bgr in enumerate(iter_pages(data, max_pages=total)):
            page, blocks = self.analyze_page(img_bgr, i, doc.assets, beam_width, batch_size,
                                             include_headers_footers)
            doc.pages.append(page)
            doc.blocks.extend(blocks)
            if progress:
                progress(i + 1, total)
        return doc

    def _recognize(self, crops, beam_width, batch_size):
        if not crops:
            return []
        return recognize_batch(crops, beam_width=beam_width, batch_size=batch_size, decoder=self.decoder)

    def process_image(
        self,
        image_path: str,
        output_path: str | None = None,
        save_debug: bool = False,
        beam_width: int = 1,
        batch_size: int = 8,
        return_segments: bool = False,
    ):
        """Run detection + recognition on an image.

        Returns the joined text by default. When ``return_segments=True`` returns
        ``(text, meta)`` where ``meta`` carries the original image size and a
        JSON-serializable list of per-region segments (bbox + text/label) for the
        web overlay.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.layout_analyzer is not None:
            return self._process_image_structured(image_path, output_path, beam_width,
                                                  batch_size, return_segments)

        img = Image.open(image_path).convert("RGB")
        image_size = img.size

        def _empty_meta():
            return {"image_size": [image_size[0], image_size[1]], "segments": []}

        # STEP 1: DETECTION
        detected_lines = self.detector.detect(image_path)
        if not detected_lines:
            return ("", _empty_meta()) if return_segments else ""

        # STEP 2: SEPARATE BY LABEL
        text_lines = [dl for dl in detected_lines if dl.label == "text"]
        logo_lines  = [dl for dl in detected_lines if dl.label == "logo"]

        # STEP 3: RECOGNITION (text only)
        ocr_queue = [dl.crop for dl in text_lines]
        recognitions = recognize_batch(ocr_queue, beam_width=beam_width, batch_size=batch_size, decoder=self.decoder) \
                       if ocr_queue else []

        # STEP 4: BUILD SEGMENTS (merged, sorted top-to-bottom)
        segments = []
        for dl, text in zip(text_lines, recognitions):
            segments.append({"type": "text", "text": text, "bbox": dl.bbox})
        for dl in logo_lines:
            segments.append({"type": "logo", "crop": dl.crop, "bbox": dl.bbox})
        segments.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))

        # STEP 5: SAVE
        if save_debug:
            self._save_debug(image_path, segments)

        final_text = "\n".join(s["text"] for s in segments if s["type"] == "text")

        if output_path:
            save_output(segments, output_path, image_path=image_path, image_size=image_size)

        if return_segments:
            overlay_segments = []
            for s in segments:
                x1, y1, x2, y2 = s["bbox"]
                item = {"type": s["type"], "bbox": [int(x1), int(y1), int(x2), int(y2)]}
                if s["type"] == "text":
                    item["text"] = s["text"]
                overlay_segments.append(item)
            meta = {"image_size": [image_size[0], image_size[1]], "segments": overlay_segments}
            return final_text, meta

        return final_text

    def _process_image_structured(self, image_path, output_path, beam_width, batch_size, return_segments):
        from .exporters import save_document
        doc = self.process_document(image_path, beam_width=beam_width, batch_size=batch_size, max_pages=1)
        if output_path:
            save_document(doc, output_path)
        if not return_segments:
            return doc.text
        page = doc.pages[0]
        segments = [{"type": "logo" if b["type"] in ("figure", "formula") else "text",
                     "bbox": b["bbox"], **({"text": b["text"]} if "text" in b else {})}
                    for b in doc.blocks if b.get("bbox")]
        return doc.text, {"image_size": [page["width"], page["height"]], "segments": segments}

    def _save_debug(self, image_path: str, segments: list) -> None:
        base   = os.path.splitext(os.path.basename(image_path))[0]
        folder = f"debug_{base}"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        for i, seg in enumerate(segments):
            if seg["type"] == "text":
                with open(os.path.join(folder, f"text_{i:03d}.txt"), "w", encoding="utf-8") as fh:
                    fh.write(seg["text"])
            else:
                seg["crop"].save(os.path.join(folder, f"logo_{i:03d}.png"))

        n_text = sum(1 for s in segments if s["type"] == "text")
        n_logo = sum(1 for s in segments if s["type"] == "logo")
        print(f"  [Debug] {n_text} text + {n_logo} logo segments saved to '{folder}/'")


# ======================================================================
# CLI
# ======================================================================
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        from .server.__main__ import main as serve
        return serve(sys.argv[2:])
    parser = argparse.ArgumentParser(
        description="Khmer OCR — pluggable detector pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Supported output formats: {', '.join(sorted(SUPPORTED_FORMATS))}\n\n"
            "Examples:\n"
            "  python ocr_engine.py --image scan.jpg --output result.txt\n"
            "  python ocr_engine.py --image scan.jpg --output result.docx --detector yolo\n"
            "  python ocr_engine.py --image scan.jpg --output result.json --detector yolo --conf 0.4\n"
        ),
    )
    parser.add_argument("--image",      required=True,
                        help="Image (JPG/PNG/TIFF/BMP/WebP) or PDF")
    parser.add_argument("--detector",   default="yolo",
                        help="Text detector: yolo | tesseract | legacy")
    parser.add_argument("--output",     default="ocr_result.txt",
                        help="Extension determines format: .txt .md .json .docx (.html with --layout or PDF input)")
    parser.add_argument("--layout",     action="store_true",
                        help="Structured output via DocLayout-YOLO layout analysis (headings, paragraphs, tables, figures).")
    parser.add_argument("--conf",       type=float, default=None,
                        help="YOLO confidence threshold (default 0.25). Only applies to --detector yolo.")
    parser.add_argument("--pad",        type=int, default=None,
                        help="Pixels to pad each bbox on all sides (default 4). Only applies to --detector yolo.")
    parser.add_argument("--decoder",    default="ar", choices=["ar", "blockwise"],
                        help="Recognition decoder: ar (default) or blockwise (Stern et al. 2018 -- same output, usually faster).")
    parser.add_argument("--beam",       type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--debug",      action="store_true")
    args = parser.parse_args()

    try:
        pipeline = KhmerOCRPipeline(detector=args.detector, conf=args.conf, pad=args.pad,
                                    decoder=args.decoder, layout=args.layout)
        with open(args.image, "rb") as fh:
            is_pdf = sniff_kind(fh.read(16)) == "pdf"
        if args.layout or is_pdf:
            from .exporters import save_document
            doc = pipeline.process_document(args.image, beam_width=args.beam, batch_size=args.batch_size)
            path = save_document(doc, args.output)
            print(f"  Saved {os.path.splitext(path)[1]} output -> {path}")
            return
        pipeline.process_image(
            image_path  = args.image,
            output_path = args.output,
            save_debug  = args.debug,
            beam_width  = args.beam,
            batch_size  = args.batch_size,
        )
    except Exception as exc:
        print(f"\nPipeline error: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
