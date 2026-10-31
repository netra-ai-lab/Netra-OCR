import importlib.util
import shutil

from .tesseract import TesseractDetector
from .yolo import YoloDetector
from .legacy import LegacyDetector

DETECTOR_REGISTRY = {
    "tesseract": TesseractDetector,
    "yolo":      YoloDetector,
    "legacy":    LegacyDetector,
}


# What each detector needs beyond the core install: Python modules from its
# extra ("legacy", "tesseract") and, for Tesseract, the system binary.
REQUIRES = {
    "yolo": (),
    "legacy": ("cv2",),
    "tesseract": ("cv2", "pytesseract", "scipy", "skimage", "networkx"),
}


def available() -> list:
    """The detectors that can run in this environment, default first."""
    found = []
    for name, modules in REQUIRES.items():
        if any(importlib.util.find_spec(m) is None for m in modules):
            continue
        if name == "tesseract" and shutil.which("tesseract") is None:
            continue
        found.append(name)
    return found


def get_detector(name: str, **kwargs):
    cls = DETECTOR_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown detector '{name}'. Available: {list(DETECTOR_REGISTRY)}"
        )
    return cls(**kwargs)
