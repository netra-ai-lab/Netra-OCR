"""Self-contained semantic HTML export (images inlined as data URIs)."""

import base64
import html

from ..document import Document

# Netra Lab brand tokens (brand-guideline/Netra Lab.html), light theme only.
_CSS = """
:root{--ground:#f6f2ec;--panel:#fff;--ink:#191014;--muted:#6b6266;--muted-2:#948b8f;
--border:#e7e1e0;--primary:#e75f1b;--maroon:#4f2724;--wash:rgba(79,39,36,.045);}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
font-family:"Kantumruy Pro","Noto Sans Khmer",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
main{max-width:46rem;margin:0 auto;padding:40px 24px 80px}
section.page{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:40px;margin-bottom:24px}
h1,h2,h3{font-weight:700;text-wrap:balance;margin:1.2em 0 .5em}
h1{font-size:28px;line-height:1.25;letter-spacing:-.01em}
h2{font-size:22px;line-height:1.3;color:var(--maroon)}
h3{font-size:18px;line-height:1.3;color:var(--maroon)}
p{font-size:17px;line-height:1.7;margin:0 0 1em;white-space:pre-wrap}
.caption,header p,footer p{font-size:13px;line-height:1.4;color:var(--muted)}
.caption{font-style:italic}
header,footer{border-color:var(--border);border-style:solid;border-width:0}
header{border-bottom-width:1px;margin-bottom:24px}
footer{border-top-width:1px;margin-top:24px;padding-top:8px}
figure{margin:1.2em 0;text-align:center}
figure img{max-width:100%;height:auto}
table{border-collapse:collapse;width:100%;margin:1.2em 0;font-size:15px;line-height:1.5}
th,td{border:1px solid var(--border);padding:8px 10px;text-align:left;vertical-align:top;white-space:pre-wrap}
th{background:var(--wash);font-weight:600}
.meta{font-size:13px;color:var(--muted-2);font-variant-numeric:tabular-nums}
"""


def _data_uri(doc: Document, asset_id: str) -> str | None:
    png = doc.assets.png_bytes(asset_id) if asset_id else None
    return "data:image/png;base64," + base64.b64encode(png).decode() if png else None


def _esc(text: str) -> str:
    return html.escape(text)


def _style(block) -> str:
    align = block.get("align", "left")
    return f' style="text-align:{align}"' if align != "left" else ""


def render_block(doc: Document, b: dict) -> str:
    t = b["type"]
    if t == "heading":
        lvl = b.get("level", 1)
        return f"<h{lvl}{_style(b)}>{_esc(b['text'])}</h{lvl}>"
    if t in ("paragraph", "page_header", "page_footer"):
        return f"<p{_style(b)}>{_esc(b['text'])}</p>"
    if t == "caption":
        return f'<p class="caption"{_style(b)}>{_esc(b["text"])}</p>'
    if t in ("figure", "formula"):
        uri = _data_uri(doc, b["image"])
        return f'<figure><img src="{uri}" alt=""></figure>' if uri else ""
    if t == "table":
        rows = b["rows"]
        out = ["<table>"]
        if b.get("caption"):
            out.append(f"<caption>{_esc(b['caption'])}</caption>")
        for ri, row in enumerate(rows):
            tag = "th" if ri == 0 and len(rows) > 1 else "td"
            out.append("<tr>" + "".join(f"<{tag}>{_esc(c)}</{tag}>" for c in row) + "</tr>")
        out.append("</table>")
        if len(rows) == 1 and len(rows[0]) == 1:
            uri = _data_uri(doc, b.get("image"))
            if uri:
                out.insert(0, f'<figure><img src="{uri}" alt=""></figure>')
        return "".join(out)
    return ""


def to_html(doc: Document, title: str = "Netra OCR document") -> bytes:
    pages: dict[int, list] = {}
    for b in doc.blocks:
        pages.setdefault(b["page"], []).append(b)
    sections = []
    for page_index in sorted(pages):
        blocks = pages[page_index]
        head = "".join(render_block(doc, b) for b in blocks if b["type"] == "page_header")
        body = "".join(render_block(doc, b) for b in blocks if b["type"] not in ("page_header", "page_footer"))
        foot = "".join(render_block(doc, b) for b in blocks if b["type"] == "page_footer")
        sections.append(
            f'<section class="page" data-page="{page_index + 1}">'
            + (f"<header>{head}</header>" if head else "")
            + body
            + (f"<footer>{foot}</footer>" if foot else "")
            + "</section>")
    page = (f'<!DOCTYPE html><html lang="km"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{_esc(title)}</title><style>{_CSS}</style></head>"
            f'<body><main>{"".join(sections)}'
            f'<p class="meta">Extracted with Netra OCR</p></main></body></html>')
    return page.encode("utf-8")
