"""Writing the model folders the app runs (see netra_ocr/inference/hub.py).

A model folder is ``netra-model.json`` plus the ONNX graphs and side files it
names. The config's ``components`` hold the model's settings (input size,
decoder shape, labels); ``variants`` list reduced-precision copies of the
graphs and the devices that should use them; ``files`` records each file's
sha256 so a damaged download can be spotted.
"""

import hashlib
import json
from pathlib import Path

CONFIG_NAME = "netra-model.json"
# Kept so folders written before the move out of Netra-Runtime stay identical.
FORMAT = "netra-bundle"
FORMAT_VERSION = 1


def write_config(root: str | Path, config: dict) -> Path:
    """Write ``config`` into ``root``, filling in the format header and file hashes."""
    root = Path(root)
    out = {"format": FORMAT, "format_version": FORMAT_VERSION,
           **{k: v for k, v in config.items() if k not in ("format", "format_version", "files")}}
    out["files"] = {
        str(p.relative_to(root)): _sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != CONFIG_NAME
    }
    path = root / CONFIG_NAME
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_config(root: str | Path) -> dict:
    return json.loads((Path(root) / CONFIG_NAME).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
