"""Exporters for structured OCR documents (``netra_ocr.document.Document``).

    data, filename, mimetype = export(doc, "docx")
    save_document(doc, "result.docx")      # format from the extension
"""

import json
import os

from ..document import Document

FORMATS = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "html": "text/html; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
    "json": "application/json",
    "txt": "text/plain; charset=utf-8",
}


def export(doc: Document, fmt: str, basename: str = "document") -> tuple[bytes, str, str]:
    """Render ``doc`` as ``fmt``. Returns ``(data, filename, mimetype)``."""
    fmt = fmt.lower().lstrip(".")
    if fmt == "docx":
        from .docx_export import to_docx
        return to_docx(doc), f"{basename}.docx", FORMATS["docx"]
    if fmt == "html":
        from .html_export import to_html
        return to_html(doc, title=basename), f"{basename}.html", FORMATS["html"]
    if fmt == "md":
        from .markdown_export import to_markdown_bytes
        data, ext = to_markdown_bytes(doc)
        return data, basename + ext, FORMATS["md"] if ext == ".md" else "application/zip"
    if fmt == "json":
        data = json.dumps(doc.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
        return data, f"{basename}.json", FORMATS["json"]
    if fmt == "txt":
        return doc.text.encode("utf-8"), f"{basename}.txt", FORMATS["txt"]
    raise ValueError(f"Unsupported format '{fmt}'. Supported: {sorted(FORMATS)}")


def save_document(doc: Document, output_path: str) -> str:
    """Write ``doc`` to ``output_path``; the extension picks the format.

    Markdown with figures is written as a .zip (document.md + images/), so
    the returned path may differ from ``output_path`` in its extension.
    """
    root, ext = os.path.splitext(output_path)
    fmt = ext.lstrip(".").lower() or "txt"
    data, filename, _ = export(doc, fmt)
    path = root + os.path.splitext(filename)[1]
    with open(path, "wb") as f:
        f.write(data)
    return path


__all__ = ["FORMATS", "export", "save_document"]
