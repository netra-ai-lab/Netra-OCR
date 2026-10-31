"""Model folders: where they come from and which of their files a device needs.

A model folder holds ``netra-model.json`` (the model's settings: input size,
decoder shape, labels, precision variants) next to its ONNX graphs and side
files such as the vocabulary. ``source`` is a local folder or
``hf://<org>/<repo>/<folder>[@revision]`` on Hugging Face.
"""

import json
from pathlib import Path

CONFIG_NAME = "netra-model.json"


def pick_variant(config: dict, device_kind: str) -> str | None:
    """The first precision variant the config prefers on ``device_kind`` (e.g. fp16 on cuda), else None."""
    for name, variant in config.get("variants", {}).items():
        if device_kind in variant.get("devices", []):
            return name
    return None


def graph_files(config: dict, variant: str | None) -> dict[str, str]:
    """Graph role -> file name, with ``variant``'s files replacing the default fp32 ones."""
    graphs = dict(config["graphs"])
    if variant is not None:
        graphs.update(config["variants"][variant].get("graphs", {}))
    return graphs


def model_file(root: Path, relative: str) -> Path:
    """``root / relative``, refusing names that point outside the folder.

    Checked on the name, not the resolved file: files in the Hugging Face cache
    are symlinks to shared blobs elsewhere.
    """
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts:
        raise ValueError(f"'{relative}' points outside the model folder")
    return root / relative


def read_config(root: Path) -> dict:
    path = root / CONFIG_NAME
    if not path.exists():
        raise FileNotFoundError(f"No {CONFIG_NAME} in {root}")
    return json.loads(path.read_text(encoding="utf-8"))


def fetch(source: str | Path, device_kind: str | None = None) -> Path:
    """A local folder for ``source``.

    For ``hf://`` sources the files are downloaded once into the Hugging Face
    cache (``HF_HOME``), so later runs work offline. With ``device_kind``, only
    the graphs that device runs are downloaded (e.g. not the fp16 copies on a
    CPU); without it, the whole folder.
    """
    text = str(source)
    if not text.startswith("hf://"):
        return Path(source)
    path, _, revision = text[len("hf://"):].partition("@")
    org, repo, *folder = path.strip("/").split("/")
    folder = "/".join(folder)
    prefix = f"{folder}/" if folder else ""
    from huggingface_hub import snapshot_download

    def download(patterns):
        return Path(snapshot_download(repo_id=f"{org}/{repo}", revision=revision or None,
                                      allow_patterns=patterns)) / folder

    if device_kind is None:
        return download([f"{folder}/*"] if folder else None)
    config = read_config(download([prefix + CONFIG_NAME]))
    every_graph = set(config["graphs"].values())
    for variant in config.get("variants", {}).values():
        every_graph.update(variant.get("graphs", {}).values())
    needed = graph_files(config, pick_variant(config, device_kind))
    side_files = [name for name in config.get("files", {}) if name not in every_graph]
    return download([prefix + name for name in (CONFIG_NAME, *side_files, *needed.values())])


def load_sessions(source: str | Path, device: str | None):
    """Fetch ``source`` and open an ONNX Runtime session per graph.

    Returns ``(root, config, variant, sessions, device)``; ``device`` is where
    the sessions actually run (a GPU that fails to load falls back to "cpu").
    """
    from .session import Session, device_kind, resolve_device

    requested = resolve_device(device)
    root = fetch(source, device_kind(requested))
    config = read_config(root)
    variant = pick_variant(config, device_kind(requested))
    sessions = {}
    for role, name in graph_files(config, variant).items():
        path = model_file(root, name)
        if not path.exists():
            raise FileNotFoundError(f"Graph file missing: {path}")
        sessions[role] = Session(path, requested)
    return root, config, variant, sessions, next(iter(sessions.values())).device
