"""Netra OCR — Khmer optical character recognition.

A Squeeze-and-Excitation Transformer network for Khmer text-line recognition,
paired with pluggable text detectors (YOLO, Tesseract, classic CV).
"""

__version__ = "1.1.1"

from .ocr_engine import KhmerOCRPipeline

__all__ = ["KhmerOCRPipeline", "__version__"]
