from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

# Layout labels produced by DocLayout-YOLO (DocStructBench). Analyzers for
# other layout models should map their classes onto these names.
TEXT_LABELS = {"title", "plain text", "figure_caption", "table_caption",
               "table_footnote", "formula_caption", "abandon"}
IMAGE_LABELS = {"figure", "isolate_formula"}
TABLE_LABELS = {"table"}


@dataclass
class LayoutRegion:
    label: str
    bbox: Box
    score: float = 1.0


class BaseLayoutAnalyzer(ABC):
    @abstractmethod
    def analyze(self, img_bgr: np.ndarray) -> List[LayoutRegion]:
        """Return the layout regions of a page (unordered)."""
        ...


def area(b: Box) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def intersection(a: Box, b: Box) -> int:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
