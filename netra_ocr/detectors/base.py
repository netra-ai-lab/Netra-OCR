from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# A detector input: a file path, or an already-decoded BGR array (as returned
# by cv2.imread). Accepting arrays lets callers that already hold the page in
# memory (PDF rasterization, the server) skip a temp-file round trip.
ImageInput = Union[str, np.ndarray]


@dataclass
class DetectedLine:
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in original image pixels
    crop: Image.Image                  # PIL RGB crop ready for recognize_batch
    label: str = "text"               # "text" | "logo"


def load_bgr(image: ImageInput) -> np.ndarray:
    """Return ``image`` as a BGR ndarray, reading it from disk if it's a path."""
    if isinstance(image, np.ndarray):
        return image
    img_bgr = cv2.imread(image)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    return img_bgr


class BaseTextDetector(ABC):
    @abstractmethod
    def detect(self, image: ImageInput) -> List[DetectedLine]:
        """Return detected text lines for an image path or BGR array."""
        ...
