"""Check the built wheel and sdist stay small: code, UI and fonts only.

    python -m build && python scripts/check_dist.py      # checks dist/*

Fails if a file is over MAX_BYTES, holds model weights, or ships anything
besides the netra_ocr package and the standard metadata (the sdist may also
carry the tests). The models are downloaded on first use, never packaged.
"""

import sys
import tarfile
import zipfile
from pathlib import Path

MAX_BYTES = 1_000_000
WEIGHTS = (".onnx", ".pt", ".pth", ".safetensors", ".bin", ".ckpt")
WHEEL_OK = ("netra_ocr/", "netra_ocr-")   # the package and its .dist-info
SDIST_OK = ("netra_ocr/", "netra_ocr.egg-info/", "tests/", "LICENSE", "MANIFEST.in",
            "PKG-INFO", "pyproject.toml", "readme.md", "setup.cfg")


def _names(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as z:
            return [n for n in z.namelist() if not n.endswith("/")]
    with tarfile.open(path) as t:
        # Drop the leading "netra_ocr-1.2.0/" every sdist member starts with.
        return [m.name.split("/", 1)[1] for m in t.getmembers() if m.isfile() and "/" in m.name]


def check(path: Path) -> list[str]:
    problems = []
    size = path.stat().st_size
    if size > MAX_BYTES:
        problems.append(f"{size / 1e6:.2f} MB, over the {MAX_BYTES / 1e6:.0f} MB limit")
    allowed = WHEEL_OK if path.suffix == ".whl" else SDIST_OK
    for name in _names(path):
        if name.lower().endswith(WEIGHTS):
            problems.append(f"model weights: {name}")
        elif not name.startswith(allowed) or (name.startswith("tests/") and not name.endswith(".py")):
            problems.append(f"not part of the package: {name}")
    return problems


def main(argv=None) -> int:
    dist = Path((argv or sys.argv[1:] or ["dist"])[0])
    files = sorted(dist.glob("*.whl")) + sorted(dist.glob("*.tar.gz"))
    if not files:
        print(f"no wheel or sdist in {dist}/")
        return 1
    failed = False
    for path in files:
        problems = check(path)
        failed |= bool(problems)
        print(f"{'FAIL' if problems else 'ok  '} {path.name} ({path.stat().st_size / 1e6:.2f} MB)")
        for problem in problems:
            print(f"     {problem}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
