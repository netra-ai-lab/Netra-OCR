"""
Khmer Text Line Detection Engine
=================================
Production-grade, zero-tuning text line detection for Khmer/English/mixed documents.

Architecture:
    Image → Quality Analysis → Multi-Pipeline Preprocessing → Character Scale Estimation
    → Khmer-Aware Morphology → Multi-PSM Tesseract → CC Graph Clustering
    → Projection Splitting → Text Verification → Candidate Scoring → Best Result
"""

import cv2
import numpy as np
import pytesseract
from scipy import ndimage, signal
from scipy.spatial.distance import cdist
from skimage.filters import threshold_sauvola
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import networkx as nx
import logging
import io
import zipfile
import time

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityReport:
    resolution_score: float = 0.0
    blur_score: float = 0.0
    blur_category: str = "Unknown"
    noise_score: float = 0.0
    contrast_score: float = 0.0
    illumination_score: float = 0.0
    illumination_category: str = "Unknown"
    character_height: float = 0.0
    overall_quality: str = "Unknown"


@dataclass
class LineBox:
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    confidence: float = 0.0
    source: str = ""  # "tesseract", "graph", "projection_split"
    ink_density: float = 0.0
    component_count: int = 0
    baseline_consistency: float = 0.0
    stroke_consistency: float = 0.0


@dataclass
class PipelineResult:
    name: str = ""
    binary_image: Optional[np.ndarray] = None
    lines: List[LineBox] = field(default_factory=list)
    score: float = 0.0
    ocr_confidence: float = 0.0
    baseline_consistency: float = 0.0
    component_consistency: float = 0.0
    text_density: float = 0.0
    line_completeness: float = 0.0
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)
    timing_ms: float = 0.0


@dataclass
class DetectionResult:
    best_pipeline: str = ""
    lines: List[LineBox] = field(default_factory=list)
    lines_before_refine: List[LineBox] = field(default_factory=list)
    quality: Optional[QualityReport] = None
    all_pipelines: List[PipelineResult] = field(default_factory=list)
    line_crops: List[np.ndarray] = field(default_factory=list)
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)
    refinement_debug: Optional['RefinementDebug'] = None
    total_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: QUALITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class QualityAnalyzer:
    """Analyzes document image quality to guide preprocessing decisions."""

    def analyze(self, gray: np.ndarray) -> QualityReport:
        report = QualityReport()

        # --- Resolution / Character Scale ---
        char_h = self._estimate_character_height(gray)
        report.character_height = char_h
        report.resolution_score = min(1.0, char_h / 30.0)

        # --- Blur ---
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        report.blur_score = min(1.0, lap_var / 500.0)
        if lap_var > 300:
            report.blur_category = "Sharp"
        elif lap_var > 80:
            report.blur_category = "Moderate Blur"
        else:
            report.blur_category = "Heavy Blur"

        # --- Noise ---
        report.noise_score = self._estimate_noise(gray)

        # --- Contrast ---
        report.contrast_score = self._measure_contrast(gray)

        # --- Illumination ---
        ill_score, ill_cat = self._analyze_illumination(gray)
        report.illumination_score = ill_score
        report.illumination_category = ill_cat

        # --- Overall ---
        avg = (report.resolution_score + report.blur_score +
               report.contrast_score + report.illumination_score) / 4.0
        if avg > 0.7:
            report.overall_quality = "Good"
        elif avg > 0.4:
            report.overall_quality = "Fair"
        else:
            report.overall_quality = "Poor"

        return report

    def _estimate_character_height(self, gray: np.ndarray) -> float:
        """Estimate median character height from connected components."""
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

        if num_labels < 2:
            return 20.0  # fallback

        heights = stats[1:, cv2.CC_STAT_HEIGHT]
        areas = stats[1:, cv2.CC_STAT_AREA]
        widths = stats[1:, cv2.CC_STAT_WIDTH]

        # Filter out noise (too small) and non-text (too large)
        img_h, img_w = gray.shape
        max_dim = min(img_h, img_w) * 0.3
        mask = (heights > 3) & (heights < max_dim) & (widths > 2) & (widths < max_dim) & (areas > 10)

        filtered_heights = heights[mask]
        if len(filtered_heights) < 5:
            return 20.0

        return float(np.median(filtered_heights))

    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Estimate noise level from tiny connected component density."""
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

        if num_labels < 2:
            return 1.0

        areas = stats[1:, cv2.CC_STAT_AREA]
        tiny = np.sum(areas < 8)
        ratio = tiny / len(areas)
        return max(0.0, 1.0 - ratio * 3.0)

    def _measure_contrast(self, gray: np.ndarray) -> float:
        """Measure text-background contrast using Michelson contrast."""
        p5, p95 = np.percentile(gray, [5, 95])
        if (p5 + p95) == 0:
            return 0.0
        michelson = (p95 - p5) / (p95 + p5)
        return float(np.clip(michelson, 0, 1))

    def _analyze_illumination(self, gray: np.ndarray) -> Tuple[float, str]:
        """Detect uneven lighting via local intensity variation."""
        # Divide image into blocks and check intensity variance
        block_size = max(32, min(gray.shape) // 8)
        h, w = gray.shape
        means = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = gray[y:y + block_size, x:x + block_size]
                means.append(np.mean(block))

        if len(means) < 4:
            return 1.0, "Uniform"

        cv_val = np.std(means) / (np.mean(means) + 1e-6)
        if cv_val < 0.08:
            return 1.0, "Uniform"
        elif cv_val < 0.20:
            return 0.6, "Uneven Lighting"
        else:
            return 0.3, "Shadows Detected"


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: MULTI-PIPELINE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

class PreprocessingPipeline:
    """Generates multiple binary candidates from a grayscale image."""

    @staticmethod
    def pipeline_a(gray: np.ndarray) -> np.ndarray:
        """CLAHE + Sauvola"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        thresh = threshold_sauvola(enhanced, window_size=25, k=0.2)
        binary = (enhanced < thresh).astype(np.uint8) * 255
        return binary

    @staticmethod
    def pipeline_b(gray: np.ndarray) -> np.ndarray:
        """CLAHE + Wolf-inspired threshold (Niblack variant with stronger k)"""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Niblack-style with aggressive k as Wolf approximation
        thresh = threshold_sauvola(enhanced, window_size=31, k=0.35)
        binary = (enhanced < thresh).astype(np.uint8) * 255
        return binary

    @staticmethod
    def pipeline_c(gray: np.ndarray) -> np.ndarray:
        """CLAHE + Adaptive Gaussian"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 12
        )
        return binary

    @staticmethod
    def pipeline_d(gray: np.ndarray) -> np.ndarray:
        """Raw grayscale → Otsu"""
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def pipeline_e(gray: np.ndarray) -> np.ndarray:
        """Contrast stretch + Sauvola"""
        p2, p98 = np.percentile(gray, [2, 98])
        stretched = np.clip((gray.astype(np.float32) - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
        thresh = threshold_sauvola(stretched, window_size=25, k=0.2)
        binary = (stretched < thresh).astype(np.uint8) * 255
        return binary

    @classmethod
    def get_all_pipelines(cls):
        return [
            ("Pipeline A: CLAHE+Sauvola", cls.pipeline_a),
            ("Pipeline B: CLAHE+Wolf", cls.pipeline_b),
            ("Pipeline C: CLAHE+AdaptGauss", cls.pipeline_c),
            ("Pipeline D: Otsu", cls.pipeline_d),
            ("Pipeline E: Stretch+Sauvola", cls.pipeline_e),
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 & 4: CHARACTER-SCALED KHMER-AWARE MORPHOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

class KhmerMorphology:
    """
    Khmer-aware morphological operations for text line grouping.

    Khmer script challenges:
    - Subscripts (្ក, ្គ, etc.) extend below the baseline
    - Vowel marks (ា, ិ, ី, etc.) sit above or around consonants
    - Stacked characters (ញ្ញ, ្ស) create tall vertical extents
    - Standard horizontal dilation can merge adjacent lines

    Strategy:
    - Use wider horizontal closing to connect characters within a line
    - Use conservative vertical dilation to preserve line separation
    - RLSA for horizontal run-length smoothing
    - All kernel sizes scale from estimated character height
    """

    @staticmethod
    def apply(binary: np.ndarray, char_height: float) -> np.ndarray:
        ch = max(8, int(char_height))

        # Horizontal closing: connect characters in a line
        # Width = 4× char height captures inter-word gaps
        # Height = 0.3× char height avoids merging Khmer stacked lines
        close_w = max(3, int(ch * 4))
        close_h = max(1, int(ch * 0.3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        # RLSA: fill small horizontal gaps
        rlsa_thresh = max(3, int(ch * 2))
        rlsa_result = KhmerMorphology._rlsa_horizontal(closed, rlsa_thresh)

        # Gentle vertical dilation to capture subscripts/superscripts
        # but not so much that it merges lines
        dil_h = max(1, int(ch * 0.15))
        dil_w = max(1, int(ch * 0.5))
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (dil_w, dil_h))
        dilated = cv2.dilate(rlsa_result, kernel_dil, iterations=1)

        return dilated

    @staticmethod
    def _rlsa_horizontal(binary: np.ndarray, threshold: int) -> np.ndarray:
        """Run-Length Smoothing Algorithm (horizontal).
        Fills runs of 0s shorter than threshold."""
        result = binary.copy()
        for row_idx in range(result.shape[0]):
            row = result[row_idx]
            run_start = None
            for col_idx in range(len(row)):
                if row[col_idx] == 0:
                    if run_start is None:
                        run_start = col_idx
                else:
                    if run_start is not None:
                        run_len = col_idx - run_start
                        if run_len <= threshold:
                            row[run_start:col_idx] = 255
                        run_start = None
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: MULTI-PSM TESSERACT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class TesseractAnalyzer:
    """Run Tesseract with multiple PSM modes and extract line-level boxes."""

    PSM_MODES = [4, 6, 11, 12]
    # PSM 4: single column of variable-sized text
    # PSM 6: single uniform block of text
    # PSM 11: sparse text (find as much text as possible)
    # PSM 12: sparse text with OSD

    @staticmethod
    def analyze(gray: np.ndarray, psm_modes: List[int] = None,
                lang: str = "khm+eng") -> Dict[int, List[LineBox]]:
        """Run multiple PSMs, return line boxes per PSM."""
        if psm_modes is None:
            psm_modes = TesseractAnalyzer.PSM_MODES

        results = {}
        for psm in psm_modes:
            try:
                config = f"--psm {psm} --oem 3"
                data = pytesseract.image_to_data(
                    gray, lang=lang, config=config, output_type=pytesseract.Output.DICT
                )
                lines = TesseractAnalyzer._group_into_lines(data)
                results[psm] = lines
            except Exception as e:
                logger.warning(f"PSM {psm} failed: {e}")
                results[psm] = []
        return results

    @staticmethod
    def _group_into_lines(data: dict) -> List[LineBox]:
        """Group Tesseract word boxes into lines by block/par/line_num."""
        line_groups: Dict[Tuple, List[dict]] = {}
        n = len(data["text"])

        for i in range(n):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not text or conf < 0:
                continue

            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            if key not in line_groups:
                line_groups[key] = []
            line_groups[key].append({
                "x": data["left"][i],
                "y": data["top"][i],
                "w": data["width"][i],
                "h": data["height"][i],
                "conf": conf
            })

        lines = []
        for key, words in line_groups.items():
            if not words:
                continue
            x_min = min(w["x"] for w in words)
            y_min = min(w["y"] for w in words)
            x_max = max(w["x"] + w["w"] for w in words)
            y_max = max(w["y"] + w["h"] for w in words)
            avg_conf = np.mean([w["conf"] for w in words])

            lines.append(LineBox(
                x=x_min, y=y_min,
                w=x_max - x_min, h=y_max - y_min,
                confidence=avg_conf,
                source="tesseract",
                component_count=len(words)
            ))
        return lines


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6: CONNECTED COMPONENT GRAPH CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

class GraphClusterer:
    """
    Build a graph from connected components and cluster into text lines.

    Each CC is a node with spatial features. Edges connect components that
    likely belong to the same line (similar baseline, similar height,
    reasonable horizontal gap).

    This supplements Tesseract by catching lines Tesseract misses.
    """

    @staticmethod
    def cluster(binary: np.ndarray, char_height: float) -> List[LineBox]:
        ch = max(8, int(char_height))

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if num_labels < 2:
            return []

        # Filter components
        img_h, img_w = binary.shape
        max_dim = min(img_h, img_w) * 0.5

        valid_indices = []
        nodes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            # Remove noise and non-text
            if h < ch * 0.2 or h > ch * 5:
                continue
            if w < ch * 0.1 or w > max_dim:
                continue
            if area < ch * 2:
                continue
            cx, cy = centroids[i]
            baseline = y + h  # bottom of bounding box
            nodes.append({
                "idx": i, "x": x, "y": y, "w": w, "h": h,
                "cx": cx, "cy": cy, "baseline": baseline, "area": area
            })
            valid_indices.append(len(nodes) - 1)

        if len(nodes) < 2:
            return []

        # Build graph
        G = nx.Graph()
        for ni, node in enumerate(nodes):
            G.add_node(ni, **node)

        # Add edges based on spatial proximity and alignment
        max_h_gap = ch * 3  # max horizontal gap between components in same line
        max_baseline_diff = ch * 0.5  # baseline must be close
        max_height_ratio = 2.5  # heights must be similar

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                ni, nj = nodes[i], nodes[j]

                # Horizontal gap
                gap = 0
                if ni["x"] + ni["w"] < nj["x"]:
                    gap = nj["x"] - (ni["x"] + ni["w"])
                elif nj["x"] + nj["w"] < ni["x"]:
                    gap = ni["x"] - (nj["x"] + nj["w"])
                else:
                    gap = 0  # overlapping

                if gap > max_h_gap:
                    continue

                # Baseline alignment
                bl_diff = abs(ni["baseline"] - nj["baseline"])
                if bl_diff > max_baseline_diff:
                    continue

                # Height similarity
                h_ratio = max(ni["h"], nj["h"]) / (min(ni["h"], nj["h"]) + 1e-6)
                if h_ratio > max_height_ratio:
                    continue

                G.add_edge(i, j)

        # Extract connected components as line clusters
        lines = []
        for component in nx.connected_components(G):
            comp_nodes = [nodes[n] for n in component]
            if len(comp_nodes) < 1:
                continue

            x_min = min(n["x"] for n in comp_nodes)
            y_min = min(n["y"] for n in comp_nodes)
            x_max = max(n["x"] + n["w"] for n in comp_nodes)
            y_max = max(n["y"] + n["h"] for n in comp_nodes)

            lines.append(LineBox(
                x=x_min, y=y_min,
                w=x_max - x_min, h=y_max - y_min,
                confidence=0.0,
                source="graph",
                component_count=len(comp_nodes)
            ))

        return lines


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 7 & 8: LINE MERGING + PROJECTION-BASED SPLITTING
# ═══════════════════════════════════════════════════════════════════════════════

class LineMerger:
    """Merge overlapping lines from Tesseract and graph clustering."""

    @staticmethod
    def merge(tess_lines: List[LineBox], graph_lines: List[LineBox],
              char_height: float) -> List[LineBox]:
        all_lines = tess_lines + graph_lines
        if not all_lines:
            return []

        # Sort by y position
        all_lines.sort(key=lambda lb: lb.y)

        merged = []
        used = set()

        for i, line in enumerate(all_lines):
            if i in used:
                continue

            current = LineBox(
                x=line.x, y=line.y, w=line.w, h=line.h,
                confidence=line.confidence, source=line.source,
                component_count=line.component_count
            )

            for j in range(i + 1, len(all_lines)):
                if j in used:
                    continue
                other = all_lines[j]

                # Check vertical overlap
                overlap_y = min(current.y + current.h, other.y + other.h) - max(current.y, other.y)
                min_h = min(current.h, other.h)
                if min_h > 0 and overlap_y / min_h > 0.3:
                    # Merge
                    new_x = min(current.x, other.x)
                    new_y = min(current.y, other.y)
                    new_x2 = max(current.x + current.w, other.x + other.w)
                    new_y2 = max(current.y + current.h, other.y + other.h)
                    current.x = new_x
                    current.y = new_y
                    current.w = new_x2 - new_x
                    current.h = new_y2 - new_y
                    current.confidence = max(current.confidence, other.confidence)
                    current.component_count += other.component_count
                    used.add(j)

            merged.append(current)

        return merged


class ProjectionSplitter:
    """Split merged lines using horizontal projection profiles."""

    @staticmethod
    def split(lines: List[LineBox], binary: np.ndarray,
              char_height: float) -> List[LineBox]:
        ch = max(8, int(char_height))
        min_valley_depth = 0.7  # fraction of peak that counts as valley
        min_line_height = int(ch * 0.6)

        result = []
        for line in lines:
            # Extract the line region
            y1, y2 = max(0, line.y), min(binary.shape[0], line.y + line.h)
            x1, x2 = max(0, line.x), min(binary.shape[1], line.x + line.w)

            if y2 - y1 < min_line_height * 2 or x2 - x1 < ch:
                result.append(line)
                continue

            roi = binary[y1:y2, x1:x2]
            # Horizontal projection profile
            projection = np.sum(roi > 0, axis=1).astype(np.float64)

            if projection.max() == 0:
                result.append(line)
                continue

            # Smooth the projection
            kernel_size = max(3, ch // 4)
            if kernel_size % 2 == 0:
                kernel_size += 1
            smoothed = ndimage.uniform_filter1d(projection, kernel_size)

            # Find valleys
            peak = smoothed.max()
            threshold = peak * (1.0 - min_valley_depth)

            # Find runs below threshold
            below = smoothed < threshold
            splits = []
            in_valley = False
            valley_start = 0

            for idx in range(len(below)):
                if below[idx] and not in_valley:
                    valley_start = idx
                    in_valley = True
                elif not below[idx] and in_valley:
                    valley_mid = (valley_start + idx) // 2
                    valley_width = idx - valley_start
                    if valley_width > ch * 0.3:  # significant gap
                        splits.append(valley_mid)
                    in_valley = False

            if not splits:
                result.append(line)
                continue

            # Split into sub-lines
            boundaries = [0] + splits + [y2 - y1]
            for k in range(len(boundaries) - 1):
                sub_y1 = boundaries[k]
                sub_y2 = boundaries[k + 1]
                sub_h = sub_y2 - sub_y1
                if sub_h < min_line_height:
                    continue
                result.append(LineBox(
                    x=line.x, y=line.y + sub_y1,
                    w=line.w, h=sub_h,
                    confidence=line.confidence,
                    source="projection_split",
                    component_count=line.component_count
                ))

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 9: TEXT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TextVerifier:
    """Verify that candidate line boxes contain actual text."""

    @staticmethod
    def verify(lines: List[LineBox], binary: np.ndarray, gray: np.ndarray,
               char_height: float) -> List[LineBox]:
        ch = max(8, int(char_height))
        verified = []

        for line in lines:
            y1 = max(0, line.y)
            y2 = min(binary.shape[0], line.y + line.h)
            x1 = max(0, line.x)
            x2 = min(binary.shape[1], line.x + line.w)

            if y2 - y1 < 3 or x2 - x1 < 3:
                continue

            roi = binary[y1:y2, x1:x2]

            # Ink density
            ink_density = np.sum(roi > 0) / (roi.size + 1e-6)
            line.ink_density = ink_density

            # Reject if too sparse or too dense
            if ink_density < 0.02 or ink_density > 0.85:
                continue

            # Aspect ratio filter
            aspect = line.w / (line.h + 1e-6)
            if aspect < 0.5:  # too tall and narrow — probably not a text line
                continue

            # Height filter — reject lines shorter than 40% of char height
            if line.h < ch * 0.4:
                continue

            # CC consistency within the line
            num_l, _, stats_l, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
            if num_l > 1:
                heights_l = stats_l[1:, cv2.CC_STAT_HEIGHT]
                valid_h = heights_l[(heights_l > ch * 0.15) & (heights_l < ch * 4)]
                if len(valid_h) > 2:
                    cv_h = np.std(valid_h) / (np.mean(valid_h) + 1e-6)
                    line.baseline_consistency = max(0, 1.0 - cv_h)
                else:
                    line.baseline_consistency = 0.5
                line.component_count = max(line.component_count, len(valid_h))
            else:
                line.baseline_consistency = 0.5

            # Stroke width consistency (approximate via distance transform)
            if roi.max() > 0:
                dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
                stroke_vals = dist[roi > 0]
                if len(stroke_vals) > 10:
                    sw_cv = np.std(stroke_vals) / (np.mean(stroke_vals) + 1e-6)
                    line.stroke_consistency = max(0, 1.0 - sw_cv * 0.5)
                else:
                    line.stroke_consistency = 0.5
            else:
                line.stroke_consistency = 0.5

            verified.append(line)

        return verified


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 10: CANDIDATE SCORING
# ═══════════════════════════════════════════════════════════════════════════════

class CandidateScorer:
    """Score pipeline results and select the best one."""

    @staticmethod
    def score_pipeline(result: PipelineResult, gray: np.ndarray,
                       char_height: float) -> float:
        lines = result.lines
        if not lines:
            return 0.0

        # OCR confidence (normalized to 0-1)
        confs = [l.confidence for l in lines if l.confidence > 0]
        ocr_conf = np.mean(confs) / 100.0 if confs else 0.0

        # Baseline consistency
        baselines = [l.baseline_consistency for l in lines]
        baseline_score = np.mean(baselines) if baselines else 0.0

        # Component consistency (lines should have multiple components)
        comp_counts = [l.component_count for l in lines]
        comp_score = min(1.0, np.mean(comp_counts) / 10.0) if comp_counts else 0.0

        # Text density: total ink area vs total line area
        total_ink = sum(l.ink_density * l.w * l.h for l in lines)
        total_area = sum(l.w * l.h for l in lines) + 1e-6
        text_density = np.clip(total_ink / total_area, 0, 1)

        # Line completeness: what fraction of image height is covered by lines
        img_h = gray.shape[0]
        covered = sum(l.h for l in lines)
        completeness = min(1.0, covered / (img_h * 0.8 + 1e-6))

        score = (
            0.40 * ocr_conf +
            0.20 * baseline_score +
            0.15 * comp_score +
            0.15 * text_density +
            0.10 * completeness
        )

        result.ocr_confidence = ocr_conf
        result.baseline_consistency = baseline_score
        result.component_consistency = comp_score
        result.text_density = text_density
        result.line_completeness = completeness
        result.score = score

        return score


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 11: LINE BOUNDARY REFINEMENT (Post-Detection)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RefinementDebug:
    """Debug information from the boundary refinement stage."""
    pre_refine_boxes: List[LineBox] = field(default_factory=list)
    post_refine_boxes: List[LineBox] = field(default_factory=list)
    pre_refine_viz: Optional[np.ndarray] = None
    post_refine_viz: Optional[np.ndarray] = None
    component_ownership_viz: Optional[np.ndarray] = None
    gap_split_viz: Optional[np.ndarray] = None
    horizontal_trim_viz: Optional[np.ndarray] = None
    vertical_trim_viz: Optional[np.ndarray] = None
    neighbor_adjust_viz: Optional[np.ndarray] = None
    per_line_stats: List[Dict] = field(default_factory=list)
    split_count: int = 0  # how many lines were split


class LineBoundaryRefiner:
    """
    Post-detection boundary refinement for tight, OCR-quality line crops.

    Solves four problems:
      1) Page-wide boxes merging disjoint text clusters → gap-aware splitting
      2) Trailing/leading whitespace → ink-envelope horizontal trim
      3) Vertical over-expansion into neighbors → projection-based vertical trim
      4) Neighbor intrusion → valley-based overlap resolution

    Pipeline (order matters):
        Detected Boxes
        → Component Ownership (assign CCs to lines)
        → Gap-Aware Segmentation (split lines at large horizontal gaps)  ← NEW
        → Ink-Based Horizontal Trim (shrink-wrap to component envelope)
        → Projection-Based Vertical Trim (shrink-wrap vertically)
        → Neighbor-Aware Overlap Resolution (valley-split overlapping boxes)
        → Adaptive Padding (tiny safety margin)
        → Final Tight Boxes

    All thresholds scale from estimated character height. Zero manual tuning.

    Khmer-specific handling:
        - Subscripts (្ក, ្គ) extend below baseline → vertical trim respects
          components whose centroids fall within the line's vertical core
        - Vowel marks (ា, ិ, ី) sit above/around consonants → small detached
          components near the line are claimed, not discarded
        - Stacked chars (ញ្ញ) create tall extents → ownership uses centroid
          proximity, not bounding-box overlap
        - Khmer word spacing is typically tighter than Latin → gap threshold
          adapts to actual observed spacing, not a fixed Latin-biased default
    """

    def refine(self, lines: List[LineBox], binary: np.ndarray,
               gray: np.ndarray, char_height: float) -> Tuple[List[LineBox], RefinementDebug]:
        debug = RefinementDebug()
        debug.pre_refine_boxes = [self._copy_box(l) for l in lines]

        if not lines:
            return lines, debug

        ch = max(8, int(char_height))
        img_h, img_w = binary.shape[:2]

        debug.pre_refine_viz = self._draw_boxes(gray, lines, (0, 0, 255), "PRE")

        # ── Step 1: Extract connected components from binary ──
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        components = []
        for i in range(1, num_labels):
            cx, cy, cw, ch_c, area = stats[i]
            ccx, ccy = centroids[i]
            if area < 4 or ch_c < 2 or cw < 2:
                continue
            if ch_c > ch * 6 or cw > img_w * 0.8:
                continue
            components.append({
                "id": i, "x": cx, "y": cy, "w": cw, "h": ch_c,
                "cx": ccx, "cy": ccy, "area": area
            })

        # ── Step 2: Component ownership ──
        line_components = [[] for _ in lines]
        unassigned = []

        for comp in components:
            best_line = -1
            best_dist = float("inf")

            for li, line in enumerate(lines):
                line_cy = line.y + line.h / 2.0
                line_top = line.y - ch * 0.4
                line_bot = line.y + line.h + ch * 0.4

                h_overlap = (comp["x"] < line.x + line.w + ch * 0.3 and
                             comp["x"] + comp["w"] > line.x - ch * 0.3)
                if not h_overlap:
                    continue
                if comp["cy"] < line_top or comp["cy"] > line_bot:
                    continue

                dist = abs(comp["cy"] - line_cy)
                if dist < best_dist:
                    best_dist = dist
                    best_line = li

            if best_line >= 0:
                line_components[best_line].append(comp)
            else:
                unassigned.append(comp)

        debug.component_ownership_viz = self._draw_component_ownership(
            gray, lines, line_components, unassigned
        )

        # ── Step 3: Gap-aware segmentation ──
        # This is the critical step that prevents page-wide boxes.
        # For each line, sort its components left-to-right, measure gaps,
        # and split where gaps are abnormally large.
        split_lines = []
        split_components = []
        debug.split_count = 0

        for li, line in enumerate(lines):
            comps = line_components[li]
            sub_lines, sub_comps = self._gap_aware_split(line, comps, ch)
            if len(sub_lines) > 1:
                debug.split_count += len(sub_lines) - 1
            split_lines.extend(sub_lines)
            split_components.extend(sub_comps)

        debug.gap_split_viz = self._draw_boxes(
            gray, split_lines, (200, 100, 255), "SPLIT"
        )

        # ── Step 4: Horizontal trim ──
        h_trimmed = []
        h_trimmed_comps = []
        for li, line in enumerate(split_lines):
            trimmed = self._horizontal_trim(line, split_components[li], binary, ch)
            h_trimmed.append(trimmed)
            h_trimmed_comps.append(split_components[li])

        debug.horizontal_trim_viz = self._draw_boxes(gray, h_trimmed, (255, 165, 0), "H-TRIM")

        # ── Step 5: Vertical trim ──
        v_trimmed = []
        for li, line in enumerate(h_trimmed):
            trimmed = self._vertical_trim(line, h_trimmed_comps[li], binary, ch)
            v_trimmed.append(trimmed)

        debug.vertical_trim_viz = self._draw_boxes(gray, v_trimmed, (0, 200, 200), "V-TRIM")

        # ── Step 6: Neighbor-aware overlap resolution ──
        adjusted = self._neighbor_aware_adjust(v_trimmed, binary, ch)

        debug.neighbor_adjust_viz = self._draw_boxes(gray, adjusted, (0, 200, 0), "NBR-ADJ")

        # ── Step 7: Adaptive padding ──
        pad = max(2, int(ch * 0.08))
        final = []
        for line in adjusted:
            fl = LineBox(
                x=max(0, line.x - pad),
                y=max(0, line.y - pad),
                w=min(img_w - max(0, line.x - pad), line.w + 2 * pad),
                h=min(img_h - max(0, line.y - pad), line.h + 2 * pad),
                confidence=line.confidence, source=line.source,
                ink_density=line.ink_density, component_count=line.component_count,
                baseline_consistency=line.baseline_consistency,
                stroke_consistency=line.stroke_consistency,
            )
            if fl.w > 0 and fl.h > 0:
                final.append(fl)

        # ── Per-line stats ──
        # Because splitting can change the line count, we report stats for
        # each output line, referencing its original parent where possible.
        for li, post in enumerate(final):
            n_comps = len(split_components[li]) if li < len(split_components) else 0
            stat = {
                "line": li + 1,
                "post_w": int(post.w),
                "post_h": int(post.h),
                "components_owned": n_comps,
                "source": post.source,
            }
            # Try to find corresponding pre-refine box for delta reporting
            if li < len(debug.pre_refine_boxes):
                pre = debug.pre_refine_boxes[li]
                stat["pre_w"] = pre.w
                stat["pre_h"] = pre.h
                stat["w_reduction"] = pre.w - int(post.w)
                stat["w_reduction_pct"] = ((pre.w - int(post.w)) / (pre.w + 1e-6)) * 100
                stat["h_reduction"] = pre.h - int(post.h)
                stat["h_reduction_pct"] = ((pre.h - int(post.h)) / (pre.h + 1e-6)) * 100
            else:
                stat["pre_w"] = "—"
                stat["pre_h"] = "—"
                stat["w_reduction_pct"] = 0.0
                stat["h_reduction_pct"] = 0.0
            debug.per_line_stats.append(stat)

        debug.post_refine_boxes = final
        debug.post_refine_viz = self._draw_boxes(gray, final, (0, 200, 0), "REFINED")

        return final, debug

    # ─────────────────────────────────────────────────────────────────────────
    # GAP-AWARE SEGMENTATION
    # ─────────────────────────────────────────────────────────────────────────

    def _gap_aware_split(self, line: LineBox, components: List[Dict],
                         ch: int) -> Tuple[List[LineBox], List[List[Dict]]]:
        """
        Split a line into sub-lines at abnormally large horizontal gaps.

        Algorithm:
          1. Sort owned components left-to-right by x position.
          2. Compute sequential gaps between adjacent components.
          3. Estimate expected spacing: median gap = typical word spacing.
          4. Any gap > max(3× median_gap, 2× char_height) triggers a split.
             The dual threshold avoids splitting normal word spaces while
             catching genuine disjoint clusters.
          5. Each cluster becomes a new sub-line box.

        If no split is needed, returns the original line as-is.
        If the line has fewer than 3 components, no split is attempted
        (too few data points to estimate normal spacing).
        """
        if len(components) < 3:
            return [self._copy_box(line)], [components]

        # Sort left-to-right
        sorted_comps = sorted(components, key=lambda c: c["x"])

        # Compute gaps between consecutive components
        gaps = []
        for i in range(len(sorted_comps) - 1):
            right_edge = sorted_comps[i]["x"] + sorted_comps[i]["w"]
            left_edge = sorted_comps[i + 1]["x"]
            gap = left_edge - right_edge
            gaps.append(gap)

        if not gaps:
            return [self._copy_box(line)], [components]

        # Estimate normal spacing from the distribution of gaps
        # Use median (robust to outliers — the big gaps we want to detect)
        gap_arr = np.array(gaps, dtype=np.float64)
        median_gap = float(np.median(gap_arr))

        # Estimate median component width as a proxy for character width
        widths = [c["w"] for c in sorted_comps]
        median_width = float(np.median(widths))

        # Split threshold:
        #   - At least 3× the median gap (adapts to actual document spacing)
        #   - At least 2× char_height (absolute minimum — prevents splitting
        #     normal Khmer word breaks which can be tight)
        #   - At least 3× median component width (ensures we're not splitting
        #     between two characters of a word)
        split_threshold = max(
            median_gap * 3.0,
            ch * 2.0,
            median_width * 3.0
        )

        # Find split points
        split_indices = []  # indices into sorted_comps where splits occur
        for i, gap in enumerate(gaps):
            if gap > split_threshold:
                split_indices.append(i + 1)  # split AFTER component i

        if not split_indices:
            return [self._copy_box(line)], [components]

        # Create sub-groups
        boundaries = [0] + split_indices + [len(sorted_comps)]
        sub_lines = []
        sub_comps_list = []

        for k in range(len(boundaries) - 1):
            group = sorted_comps[boundaries[k]:boundaries[k + 1]]
            if not group:
                continue

            x_min = min(c["x"] for c in group)
            y_min = min(c["y"] for c in group)
            x_max = max(c["x"] + c["w"] for c in group)
            y_max = max(c["y"] + c["h"] for c in group)

            sub_lines.append(LineBox(
                x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min,
                confidence=line.confidence, source=line.source,
                ink_density=line.ink_density,
                component_count=len(group),
                baseline_consistency=line.baseline_consistency,
                stroke_consistency=line.stroke_consistency,
            ))
            sub_comps_list.append(group)

        return sub_lines, sub_comps_list

    # ─────────────────────────────────────────────────────────────────────────
    # HORIZONTAL TRIMMING
    # ─────────────────────────────────────────────────────────────────────────

    def _horizontal_trim(self, line: LineBox, components: List[Dict],
                         binary: np.ndarray, ch: int) -> LineBox:
        """
        Shrink-wrap horizontal extent to the component envelope.

        When ≥2 components are owned, the envelope IS the answer — no safety
        floor needed because over-expansion is exactly the problem we solve.

        Falls back to vertical projection scanning when components are scarce.
        """
        img_h, img_w = binary.shape

        if components and len(components) >= 2:
            x_min = min(c["x"] for c in components)
            x_max = max(c["x"] + c["w"] for c in components)
            new_w = x_max - x_min
            if new_w < ch:
                return self._copy_box(line)
            return self._box_with(line, x=x_min, w=new_w)

        if components and len(components) == 1:
            c = components[0]
            return self._box_with(line, x=c["x"], w=c["w"])

        # Projection fallback
        y1, y2 = max(0, line.y), min(img_h, line.y + line.h)
        x1, x2 = max(0, line.x), min(img_w, line.x + line.w)
        if y2 <= y1 or x2 <= x1:
            return self._copy_box(line)

        roi = binary[y1:y2, x1:x2]
        col_proj = np.sum(roi > 0, axis=0)
        if col_proj.max() == 0:
            return self._copy_box(line)

        thresh = col_proj.max() * 0.05
        left_off = int(np.argmax(col_proj > thresh))
        right_off = len(col_proj) - 1 - int(np.argmax(col_proj[::-1] > thresh))

        x_min = line.x + left_off
        x_max = line.x + right_off + 1
        new_w = x_max - x_min
        if new_w < ch or new_w < line.w * 0.15:
            return self._copy_box(line)

        return self._box_with(line, x=x_min, w=new_w)

    # ─────────────────────────────────────────────────────────────────────────
    # VERTICAL TRIMMING
    # ─────────────────────────────────────────────────────────────────────────

    def _vertical_trim(self, line: LineBox, components: List[Dict],
                       binary: np.ndarray, ch: int) -> LineBox:
        """
        Shrink-wrap vertical extent to the component envelope.

        Because component ownership already assigned diacritics and subscripts
        to their correct line, the envelope automatically includes them.
        """
        img_h, img_w = binary.shape

        if components and len(components) >= 2:
            y_min = min(c["y"] for c in components)
            y_max = max(c["y"] + c["h"] for c in components)
            new_h = y_max - y_min
            if new_h < ch * 0.5:
                center = (y_min + y_max) / 2.0
                y_min = int(center - ch * 0.25)
                y_max = int(center + ch * 0.25)
                new_h = y_max - y_min
            if new_h < 3:
                return self._copy_box(line)
            return self._box_with(line, y=y_min, h=new_h)

        if components and len(components) == 1:
            c = components[0]
            return self._box_with(line, y=c["y"], h=c["h"])

        # Projection fallback
        y1, y2 = max(0, line.y), min(img_h, line.y + line.h)
        x1, x2 = max(0, line.x), min(img_w, line.x + line.w)
        if y2 <= y1 or x2 <= x1:
            return self._copy_box(line)

        roi = binary[y1:y2, x1:x2]
        row_proj = np.sum(roi > 0, axis=1)
        if row_proj.max() == 0:
            return self._copy_box(line)

        thresh = row_proj.max() * 0.05
        top_off = int(np.argmax(row_proj > thresh))
        bot_off = len(row_proj) - 1 - int(np.argmax(row_proj[::-1] > thresh))

        y_min = line.y + top_off
        y_max = line.y + bot_off + 1
        new_h = y_max - y_min

        if new_h < ch * 0.5:
            center = (y_min + y_max) / 2.0
            y_min = int(center - ch * 0.25)
            y_max = int(center + ch * 0.25)
            new_h = y_max - y_min

        if new_h < line.h * 0.3:
            return self._copy_box(line)

        return self._box_with(line, y=y_min, h=new_h)

    # ─────────────────────────────────────────────────────────────────────────
    # NEIGHBOR-AWARE OVERLAP RESOLUTION
    # ─────────────────────────────────────────────────────────────────────────

    def _neighbor_aware_adjust(self, lines: List[LineBox],
                               binary: np.ndarray, ch: int) -> List[LineBox]:
        """
        Resolve vertical overlaps between adjacent lines by finding the
        projection valley in the overlap region and splitting there.
        """
        if len(lines) < 2:
            return [self._copy_box(l) for l in lines]

        sorted_pairs = sorted(enumerate(lines), key=lambda x: x[1].y)
        result = [self._copy_box(l) for l in lines]

        for si in range(len(sorted_pairs) - 1):
            idx_a = sorted_pairs[si][0]
            idx_b = sorted_pairs[si + 1][0]
            a, b = result[idx_a], result[idx_b]

            overlap = (a.y + a.h) - b.y
            if overlap <= max(1, int(ch * 0.05)):
                continue

            # Find projection valley in overlap zone
            ov_y1 = max(0, b.y)
            ov_y2 = min(binary.shape[0], a.y + a.h)
            ov_x1 = max(0, min(a.x, b.x))
            ov_x2 = min(binary.shape[1], max(a.x + a.w, b.x + b.w))

            if ov_y2 <= ov_y1 or ov_x2 <= ov_x1:
                continue

            ov_roi = binary[ov_y1:ov_y2, ov_x1:ov_x2]
            row_proj = np.sum(ov_roi > 0, axis=1).astype(np.float64)

            if len(row_proj) > 0 and row_proj.max() > 0:
                k = max(3, ch // 6)
                if k % 2 == 0:
                    k += 1
                smoothed = ndimage.uniform_filter1d(row_proj, k)
                split_offset = int(np.argmin(smoothed))
            else:
                split_offset = (ov_y2 - ov_y1) // 2

            split_y = ov_y1 + split_offset

            new_a_h = split_y - a.y
            if new_a_h >= ch * 0.4:
                a.h = new_a_h

            new_b_h = (b.y + b.h) - split_y
            if new_b_h >= ch * 0.4:
                b.y = split_y
                b.h = new_b_h

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _copy_box(lb: LineBox) -> LineBox:
        return LineBox(
            x=lb.x, y=lb.y, w=lb.w, h=lb.h,
            confidence=lb.confidence, source=lb.source,
            ink_density=lb.ink_density, component_count=lb.component_count,
            baseline_consistency=lb.baseline_consistency,
            stroke_consistency=lb.stroke_consistency,
        )

    @staticmethod
    def _box_with(lb: LineBox, **overrides) -> LineBox:
        """Copy a LineBox, overriding specific spatial fields."""
        return LineBox(
            x=overrides.get("x", lb.x),
            y=overrides.get("y", lb.y),
            w=overrides.get("w", lb.w),
            h=overrides.get("h", lb.h),
            confidence=lb.confidence, source=lb.source,
            ink_density=lb.ink_density, component_count=lb.component_count,
            baseline_consistency=lb.baseline_consistency,
            stroke_consistency=lb.stroke_consistency,
        )

    def _draw_boxes(self, gray: np.ndarray, lines: List[LineBox],
                    color: Tuple[int, int, int], label_prefix: str) -> np.ndarray:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for i, lb in enumerate(lines):
            cv2.rectangle(vis, (lb.x, lb.y), (lb.x + lb.w, lb.y + lb.h), color, 2)
            cv2.putText(vis, f"{label_prefix}{i + 1}",
                        (lb.x, max(12, lb.y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return vis

    def _draw_component_ownership(self, gray: np.ndarray, lines: List[LineBox],
                                   line_components: List[List[Dict]],
                                   unassigned: List[Dict]) -> np.ndarray:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        n = max(len(lines), 1)
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            c = cv2.cvtColor(np.uint8([[[hue, 200, 220]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(v) for v in c))

        for li, comps in enumerate(line_components):
            col = colors[li % len(colors)]
            for c in comps:
                cv2.rectangle(vis, (c["x"], c["y"]),
                              (c["x"] + c["w"], c["y"] + c["h"]), col, 1)
            lb = lines[li]
            cv2.rectangle(vis, (lb.x, lb.y), (lb.x + lb.w, lb.y + lb.h), col, 2)

        for c in unassigned:
            cv2.rectangle(vis, (c["x"], c["y"]),
                          (c["x"] + c["w"], c["y"] + c["h"]), (128, 128, 128), 1)
        return vis


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class KhmerLineDetector:
    """Main orchestrator for the Khmer text line detection pipeline."""

    def __init__(self):
        self.quality_analyzer = QualityAnalyzer()
        self.graph_clusterer = GraphClusterer()
        self.text_verifier = TextVerifier()
        self.candidate_scorer = CandidateScorer()
        self.boundary_refiner = LineBoundaryRefiner()

    def detect(self, image: np.ndarray, progress_callback=None) -> DetectionResult:
        t_start = time.time()
        result = DetectionResult()

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Deskew
        gray = self._deskew(gray)
        result.debug_images["01_grayscale"] = gray.copy()

        if progress_callback:
            progress_callback(0.05, "Analyzing document quality...")

        # Stage 1: Quality Analysis
        quality = self.quality_analyzer.analyze(gray)
        result.quality = quality
        char_height = quality.character_height

        if progress_callback:
            progress_callback(0.10, "Running Tesseract analysis...")

        # Run Tesseract ONCE on the grayscale (it doesn't use the binary)
        # This is the main bottleneck, so we cache and share across pipelines
        tess_results = TesseractAnalyzer.analyze(gray, lang="khm+eng")
        all_tess_lines = []
        for psm, lines in tess_results.items():
            all_tess_lines.extend(lines)

        if progress_callback:
            progress_callback(0.40, "Running preprocessing pipelines...")

        # Stage 2-10: Run each pipeline independently
        pipelines = PreprocessingPipeline.get_all_pipelines()
        all_results = []

        for idx, (name, pipeline_fn) in enumerate(pipelines):
            if progress_callback:
                frac = 0.40 + 0.50 * (idx / len(pipelines))
                progress_callback(frac, f"Processing {name}...")

            try:
                pr = self._run_pipeline(name, pipeline_fn, gray, char_height,
                                        cached_tess_lines=all_tess_lines)
                all_results.append(pr)
            except Exception as e:
                logger.warning(f"Pipeline {name} failed: {e}")
                all_results.append(PipelineResult(name=name, score=0.0))

        result.all_pipelines = all_results

        if progress_callback:
            progress_callback(0.90, "Selecting best result...")

        # Select best pipeline
        best = max(all_results, key=lambda p: p.score) if all_results else None
        if best and best.lines:
            result.best_pipeline = best.name
            result.lines = best.lines
            result.lines_before_refine = [LineBox(x=l.x, y=l.y, w=l.w, h=l.h,
                                                   confidence=l.confidence, source=l.source,
                                                   ink_density=l.ink_density,
                                                   component_count=l.component_count,
                                                   baseline_consistency=l.baseline_consistency,
                                                   stroke_consistency=l.stroke_consistency)
                                           for l in best.lines]
            result.debug_images.update(best.debug_images)
        else:
            result.best_pipeline = "None"
            result.lines = []
            result.lines_before_refine = []

        # ── Stage 11: Line Boundary Refinement ──
        # Tighten boxes horizontally (remove trailing whitespace) and
        # vertically (prevent overlap with neighbors), using the binary
        # from the best pipeline.
        if result.lines:
            if progress_callback:
                progress_callback(0.92, "Refining line boundaries...")

            best_binary = best.binary_image if best else None
            if best_binary is None:
                # Fallback: generate binary from best pipeline
                _, best_binary = cv2.threshold(gray, 0, 255,
                                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            refined, ref_debug = self.boundary_refiner.refine(
                result.lines, best_binary, gray, char_height
            )
            result.lines = refined
            result.refinement_debug = ref_debug

        # Generate line crops
        if progress_callback:
            progress_callback(0.95, "Generating line crops...")

        result.line_crops = self._generate_crops(image, result.lines)

        result.total_time_ms = (time.time() - t_start) * 1000

        if progress_callback:
            progress_callback(1.0, "Done!")

        return result

    def _run_pipeline(self, name: str, pipeline_fn, gray: np.ndarray,
                      char_height: float,
                      cached_tess_lines: List[LineBox] = None) -> PipelineResult:
        t0 = time.time()
        pr = PipelineResult(name=name)

        # Preprocessing
        binary = pipeline_fn(gray)
        pr.binary_image = binary
        pr.debug_images[f"{name}_binary"] = binary.copy()

        # Khmer-aware morphology
        morphed = KhmerMorphology.apply(binary, char_height)
        pr.debug_images[f"{name}_morphed"] = morphed.copy()

        # Use cached Tesseract lines (already run once on grayscale)
        all_tess_lines = cached_tess_lines if cached_tess_lines else []

        # Graph clustering on morphed image (this IS pipeline-specific)
        graph_lines = self.graph_clusterer.cluster(morphed, char_height)

        # Merge
        merged = LineMerger.merge(all_tess_lines, graph_lines, char_height)

        # Projection splitting
        split = ProjectionSplitter.split(merged, binary, char_height)

        # Text verification
        verified = self.text_verifier.verify(split, binary, gray, char_height)

        pr.lines = verified

        # Generate debug visualization
        debug_viz = self._draw_lines(gray, verified, all_tess_lines, graph_lines)
        pr.debug_images[f"{name}_detections"] = debug_viz

        # Score
        self.candidate_scorer.score_pipeline(pr, gray, char_height)

        pr.timing_ms = (time.time() - t0) * 1000
        return pr

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """Deskew using Hough transform on edges."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                minLineLength=gray.shape[1] // 8,
                                maxLineGap=10)
        if lines is None or len(lines) < 3:
            return gray

        angles = []
        # OpenCV 4 returns (N, 1, 4), OpenCV 5 (N, 4): flatten to one row per segment.
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            if abs(x2 - x1) > 10:  # not vertical
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) < 15:  # only small skew
                    angles.append(angle)

        if not angles:
            return gray

        median_angle = np.median(angles)
        if abs(median_angle) < 0.3:
            return gray

        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def _draw_lines(self, gray: np.ndarray, final: List[LineBox],
                    tess: List[LineBox], graph: List[LineBox]) -> np.ndarray:
        """Draw detection debug visualization."""
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Tesseract lines in blue (thin)
        for lb in tess:
            cv2.rectangle(vis, (lb.x, lb.y), (lb.x + lb.w, lb.y + lb.h),
                          (255, 180, 0), 1)

        # Graph lines in cyan (thin)
        for lb in graph:
            cv2.rectangle(vis, (lb.x, lb.y), (lb.x + lb.w, lb.y + lb.h),
                          (0, 255, 255), 1)

        # Final lines in green (thick)
        for i, lb in enumerate(final):
            cv2.rectangle(vis, (lb.x, lb.y), (lb.x + lb.w, lb.y + lb.h),
                          (0, 255, 0), 2)
            cv2.putText(vis, f"L{i + 1}", (lb.x, lb.y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return vis

    def _generate_crops(self, image: np.ndarray,
                        lines: List[LineBox]) -> List[np.ndarray]:
        """Generate cropped line images with padding."""
        crops = []
        img_h, img_w = image.shape[:2]
        pad = 2

        for lb in lines:
            y1 = max(0, lb.y - pad)
            y2 = min(img_h, lb.y + lb.h + pad)
            x1 = max(0, lb.x - pad)
            x2 = min(img_w, lb.x + lb.w + pad)
            crop = image[y1:y2, x1:x2].copy()
            if crop.size > 0:
                crops.append(crop)
        return crops

    @staticmethod
    def crops_to_zip(crops: List[np.ndarray]) -> bytes:
        """Package line crops into a ZIP file."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, crop in enumerate(crops):
                _, png_data = cv2.imencode(".png", crop)
                zf.writestr(f"line_{i + 1:03d}.png", png_data.tobytes())
        return buf.getvalue()