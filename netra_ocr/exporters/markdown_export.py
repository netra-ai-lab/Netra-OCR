"""Markdown export (GFM tables). Figures go in a zip next to the .md file."""

import io
import zipfile

from ..document import Document


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def to_markdown(doc: Document, image_dir: str = "images") -> tuple[str, dict]:
    """Return ``(markdown, images)``; ``images`` maps relative path -> PNG bytes."""
    out, images = [], {}
    prev_page = None
    for b in doc.blocks:
        if prev_page is not None and b["page"] != prev_page:
            out.append("---")
        prev_page = b["page"]
        t = b["type"]
        if t == "heading":
            out.append("#" * b.get("level", 1) + " " + b["text"].replace("\n", " — "))
        elif t in ("paragraph", "page_header", "page_footer"):
            out.append(b["text"].replace("\n", "  \n"))
        elif t == "caption":
            out.append(f"*{b['text']}*")
        elif t in ("figure", "formula"):
            png = doc.assets.png_bytes(b["image"])
            if png:
                path = f"{image_dir}/{b['image']}.png"
                images[path] = png
                out.append(f"![]({path})")
        elif t == "table":
            rows = b["rows"]
            ncols = max(len(r) for r in rows)
            lines = ["| " + " | ".join(_cell(c) for c in rows[0]) + " |",
                     "|" + " --- |" * ncols]
            lines += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows[1:]]
            out.append("\n".join(lines))
    return "\n\n".join(out) + "\n", images


def to_markdown_bytes(doc: Document) -> tuple[bytes, str]:
    """Return ``(data, extension)``: a plain .md, or a .zip when there are images."""
    md, images = to_markdown(doc)
    if not images:
        return md.encode("utf-8"), ".md"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("document.md", md)
        for path, data in images.items():
            z.writestr(path, data)
    return buf.getvalue(), ".zip"
