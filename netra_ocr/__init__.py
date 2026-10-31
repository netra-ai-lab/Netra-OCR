"""Netra OCR — Khmer optical character recognition.

A Squeeze-and-Excitation Transformer network for Khmer text-line recognition,
paired with pluggable text detectors (YOLO, Tesseract, classic CV).

    ocr_engine.py   KhmerOCRPipeline: page -> lines -> text -> document; the CLI
    recognition.py  recognize() for line images you already have
    models.py       which model files to load (Hugging Face or NETRA_BUNDLES)
    inference/      runs the ONNX models: pre-processing, ONNX Runtime, decoding
    detectors/      find text lines: yolo (default), tesseract, legacy
    layout/         page regions (headings, tables...), reading order, table grids
    structure.py    regions + lines + text -> heading / paragraph / table blocks
    document.py     the structured Document the editor and exporters share
    io.py           PDF / TIFF / image files -> page images
    exporters/      Word, HTML, Markdown, JSON, TXT
    server/         the web app and REST API (netra_ocr serve)
"""

__version__ = "1.2.0"

from .ocr_engine import KhmerOCRPipeline

__all__ = ["KhmerOCRPipeline", "__version__"]
