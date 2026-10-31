"""Netra OCR's model inference on ONNX Runtime (no PyTorch).

Each model is a folder of ONNX graphs plus ``netra-model.json`` (its settings),
published in the Hugging Face repo ``Netra-Lab/Netra-OCR``. This package does
everything around the graphs: picking the device, downloading only the files
that device needs, and the model-specific pre- and post-processing.

    from netra_ocr.inference import Recognizer, Detector
    recognizer = Recognizer("hf://Netra-Lab/Netra-OCR/recognition", device="auto")
    print(recognizer("line.png").text)
"""

from .detector import Detection, Detector
from .recognizer import Recognizer, Text
from .session import available_devices, resolve_device

__all__ = ["Detection", "Detector", "Recognizer", "Text", "available_devices", "resolve_device"]
