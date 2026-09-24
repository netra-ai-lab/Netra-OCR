"""Load document pages (images, multi-page TIFF, PDF) as BGR arrays."""

import io
import os
from typing import Iterator, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

Source = Union[str, bytes]

PDF_DPI = 200
IMAGE_KINDS = {"jpeg", "png", "tiff", "bmp", "webp"}


def sniff_kind(head: bytes) -> str | None:
    """Identify a file by its magic bytes: 'pdf', an image kind, or None."""
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head.startswith(b"BM"):
        return "bmp"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


def _read(source: Source) -> bytes:
    if isinstance(source, bytes):
        return source
    with open(source, "rb") as f:
        return f.read()


def count_pages(source: Source) -> int:
    data = _read(source)
    if sniff_kind(data[:16]) == "pdf":
        import pypdfium2 as pdfium
        return len(pdfium.PdfDocument(data))
    with Image.open(io.BytesIO(data)) as im:
        return getattr(im, "n_frames", 1)


def _pil_to_bgr(im: Image.Image) -> np.ndarray:
    im = ImageOps.exif_transpose(im).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def iter_pages(source: Source, max_pages: int | None = None, dpi: int = PDF_DPI) -> Iterator[np.ndarray]:
    """Yield each page as a BGR array: PDF pages rendered at ``dpi``, image frames as-is."""
    data = _read(source)
    kind = sniff_kind(data[:16])
    if kind == "pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError as e:
            raise ImportError("PDF input requires the 'pdf' extra. Install it with: "
                              "pip install \"netra-ocr[pdf]\"") from e
        pdf = pdfium.PdfDocument(data)
        n = len(pdf) if max_pages is None else min(len(pdf), max_pages)
        for i in range(n):
            page = pdf[i]
            bitmap = page.render(scale=dpi / 72)
            yield _pil_to_bgr(bitmap.to_pil())
            page.close()
        pdf.close()
        return
    if kind is None:
        raise ValueError("Unsupported file type. Use a PDF, JPG, PNG, TIFF, BMP or WebP.")
    with Image.open(io.BytesIO(data)) as im:
        n = getattr(im, "n_frames", 1)
        if max_pages is not None:
            n = min(n, max_pages)
        for i in range(n):
            im.seek(i)
            yield _pil_to_bgr(im.copy())
