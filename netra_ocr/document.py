"""Structured OCR document: an ordered list of semantic blocks, like an HTML DOM.

This is the single representation the pipeline produces, the web editor
edits, and every exporter (docx/html/md/json/txt) reads from.

Schema (``Document.to_dict()``)::

    {
      "version": 1,
      "revision": 0,                      # bumped on every saved edit
      "pages":  [{"index": 0, "width": W, "height": H, "image": "page-0"}],
      "blocks": [
        {"id": "p0-b3", "type": "heading", "level": 1, "text": "...",
         "align": "center", "page": 0, "bbox": [x1, y1, x2, y2], "label": "title"},
        {"id": "p0-b4", "type": "paragraph", "text": "...", "page": 0, ...},
        {"id": "p0-b5", "type": "table", "rows": [["a", "b"], ["c", "d"]],
         "image": "p0-table5", "page": 0, ...},
        {"id": "p0-b6", "type": "figure", "image": "p0-fig6", "page": 0, ...},
      ]
    }

Block types: heading, paragraph, caption, table, figure, formula,
page_header, page_footer. Text blocks keep line breaks as "\n". Blocks are
in reading order; a change of ``page`` between consecutive blocks is a page
break. ``bbox``/``label``/``lines`` record where the block came from on the
page image (for highlighting in the editor) and are informational only.
"""

import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from PIL import Image

SCHEMA_VERSION = 1
TEXT_TYPES = {"heading", "paragraph", "caption", "page_header", "page_footer"}
IMAGE_TYPES = {"figure", "formula"}
ALIGNMENTS = {"left", "center", "right", "justify"}
BLOCK_TYPES = TEXT_TYPES | IMAGE_TYPES | {"table"}

_ASSET_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_TEXT = 100_000
_MAX_BLOCKS = 20_000


class DocumentValidationError(ValueError):
    pass


class AssetStore:
    """Images referenced by blocks (page renders, figure/table crops).

    Held in memory while the pipeline runs; ``save(dir)`` writes them as
    files, and a store opened on a directory loads them lazily on access.
    """

    def __init__(self, directory: Optional[str] = None):
        self._mem: Dict[str, Image.Image] = {}
        self._dir = directory

    def put(self, asset_id: str, image: Image.Image) -> str:
        if not _ASSET_ID.match(asset_id):
            raise ValueError(f"Invalid asset id: {asset_id!r}")
        self._mem[asset_id] = image
        return asset_id

    def _path(self, asset_id: str) -> Optional[str]:
        if self._dir is None or not _ASSET_ID.match(asset_id):
            return None
        for ext in (".png", ".jpg"):
            p = os.path.join(self._dir, asset_id + ext)
            if os.path.exists(p):
                return p
        return None

    def path(self, asset_id: str) -> Optional[str]:
        """File path of a saved asset, or None if it only exists in memory."""
        return self._path(asset_id)

    def get(self, asset_id: str) -> Optional[Image.Image]:
        if asset_id in self._mem:
            return self._mem[asset_id]
        p = self._path(asset_id)
        return Image.open(p) if p else None

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self._mem or self._path(asset_id) is not None

    def png_bytes(self, asset_id: str) -> Optional[bytes]:
        img = self.get(asset_id)
        if img is None:
            return None
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for asset_id, img in self._mem.items():
            # Page renders are photos/scans: JPEG keeps them small. Crops stay
            # lossless PNG since they're embedded in exported documents.
            if asset_id.startswith("page-"):
                img.convert("RGB").save(os.path.join(directory, asset_id + ".jpg"), quality=85)
            else:
                img.save(os.path.join(directory, asset_id + ".png"))
        self._dir = directory
        self._mem.clear()


@dataclass
class Document:
    pages: List[dict] = field(default_factory=list)
    blocks: List[dict] = field(default_factory=list)
    assets: AssetStore = field(default_factory=AssetStore)
    revision: int = 0

    # ── serialization ──────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {"version": SCHEMA_VERSION, "revision": self.revision,
                "pages": self.pages, "blocks": self.blocks}

    @classmethod
    def from_dict(cls, data: dict, assets: Optional[AssetStore] = None) -> "Document":
        assets = assets or AssetStore()
        pages, blocks = validate(data, assets)
        return cls(pages=pages, blocks=blocks, assets=assets,
                   revision=int(data.get("revision", 0)))

    def save(self, directory: str, name: str = "document.json") -> None:
        os.makedirs(directory, exist_ok=True)
        self.assets.save(os.path.join(directory, "assets"))
        tmp = os.path.join(directory, name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
        os.replace(tmp, os.path.join(directory, name))

    @classmethod
    def load(cls, directory: str, name: str = "document.json") -> "Document":
        with open(os.path.join(directory, name), encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, AssetStore(os.path.join(directory, "assets")))

    # ── convenience ────────────────────────────────────────────────────
    @property
    def text(self) -> str:
        """Plain reading-order text, blank line between blocks."""
        parts = []
        for b in self.blocks:
            if b["type"] in TEXT_TYPES:
                parts.append(b["text"])
            elif b["type"] == "table":
                parts.append("\n".join("\t".join(row) for row in b["rows"]))
        return "\n\n".join(p for p in parts if p.strip())


# ── validation (untrusted input from the editor/API) ──────────────────────

def _bbox(v, where):
    if v is None:
        return None
    if (not isinstance(v, list) or len(v) != 4
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
        raise DocumentValidationError(f"{where}: bbox must be [x1, y1, x2, y2]")
    return [int(x) for x in v]


def _text(v, where):
    if not isinstance(v, str):
        raise DocumentValidationError(f"{where}: text must be a string")
    if len(v) > _MAX_TEXT:
        raise DocumentValidationError(f"{where}: text too long")
    return v


def validate(data: dict, assets: AssetStore):
    """Validate and normalize a document dict. Returns (pages, blocks)."""
    if not isinstance(data, dict):
        raise DocumentValidationError("document must be an object")
    pages_in, blocks_in = data.get("pages"), data.get("blocks")
    if not isinstance(pages_in, list) or not isinstance(blocks_in, list):
        raise DocumentValidationError("document needs 'pages' and 'blocks' lists")
    if len(blocks_in) > _MAX_BLOCKS:
        raise DocumentValidationError("too many blocks")

    pages = []
    for i, p in enumerate(pages_in):
        if not isinstance(p, dict):
            raise DocumentValidationError(f"pages[{i}] must be an object")
        pages.append({"index": i, "width": int(p.get("width", 0)), "height": int(p.get("height", 0)),
                      "image": p.get("image") if isinstance(p.get("image"), str) else None})

    def asset_ref(v, where):
        if v is None:
            return None
        if not isinstance(v, str) or v not in assets:
            raise DocumentValidationError(f"{where}: unknown image {v!r}")
        return v

    blocks, seen = [], set()
    for i, b in enumerate(blocks_in):
        where = f"blocks[{i}]"
        if not isinstance(b, dict):
            raise DocumentValidationError(f"{where} must be an object")
        btype = b.get("type")
        if btype not in BLOCK_TYPES:
            raise DocumentValidationError(f"{where}: unknown type {btype!r}")
        bid = b.get("id")
        if not isinstance(bid, str) or not bid or len(bid) > 64 or bid in seen:
            raise DocumentValidationError(f"{where}: id must be a unique non-empty string")
        seen.add(bid)
        page = b.get("page", 0)
        if not isinstance(page, int) or not (0 <= page < max(1, len(pages))):
            raise DocumentValidationError(f"{where}: page out of range")

        out = {"id": bid, "type": btype, "page": page, "bbox": _bbox(b.get("bbox"), where)}
        if isinstance(b.get("label"), str):
            out["label"] = b["label"][:32]
        if isinstance(b.get("lines"), list):
            # Source OCR lines (text + bbox) -- kept as recognized, for API
            # consumers that want line geometry; edits don't touch them.
            out["lines"] = [{"text": _text(l.get("text", ""), where), "bbox": _bbox(l.get("bbox"), where)}
                            for l in b["lines"] if isinstance(l, dict)]

        if btype in TEXT_TYPES:
            out["text"] = _text(b.get("text", ""), where)
            align = b.get("align", "left")
            if align not in ALIGNMENTS:
                raise DocumentValidationError(f"{where}: align must be one of {sorted(ALIGNMENTS)}")
            out["align"] = align
            if btype == "heading":
                level = b.get("level", 1)
                if level not in (1, 2, 3):
                    raise DocumentValidationError(f"{where}: heading level must be 1-3")
                out["level"] = level
        elif btype == "table":
            rows = b.get("rows")
            if (not isinstance(rows, list) or not rows
                    or not all(isinstance(r, list) and r for r in rows)):
                raise DocumentValidationError(f"{where}: table rows must be a non-empty list of lists")
            width = max(len(r) for r in rows)
            out["rows"] = [[_text(c, where) for c in r] + [""] * (width - len(r)) for r in rows]
            out["image"] = asset_ref(b.get("image"), where)
            if isinstance(b.get("caption"), str):
                out["caption"] = _text(b["caption"], where)
        else:
            out["image"] = asset_ref(b.get("image"), where)
            if out["image"] is None:
                raise DocumentValidationError(f"{where}: {btype} needs an image")
        blocks.append(out)
    return pages, blocks
