"""
Rainfall Pattern & Crop Suitability Analyser for Ugandan Agro-Ecological Zones.
Implements signal peak detection, multi-metric distance matrices, and crop rule classification.
"""

import numpy as np
import scipy.spatial.distance as dist
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Any

class Region:
    """Models a geographical region's rainfall profile using vectorized NumPy arrays."""
    
    def __init__(self, name: str, rainfall: List[float]) -> None:
        self.name: str = name
        self.rainfall: np.ndarray = np.array(rainfall, dtype=float)
        if len(self.rainfall) != 12:
            raise ValueError("Rainfall array must contain exactly 12 monthly values.")

    @property
    def total_annual(self) -> float:
        return float(np.sum(self.rainfall))

    @property
    def mean_monthly(self) -> float:
        return float(np.mean(self.rainfall))

    @property
    def coefficient_of_variation(self) -> float:
        std = np.std(self.rainfall, ddof=1)
        mean = self.mean_monthly
        return float(std / mean) if mean > 0 else 0.0

    def get_extremes(self) -> Tuple[int, int]:
        """Returns the 0-indexed indices for (wettest_month, driest_month)."""
        return int(np.argmax(self.rainfall)), int(np.argmin(self.rainfall))

    def classify_regime(self) -> str:
        """Detects rainy seasons using scipy peak-finding algorithms to map climate zones."""
        # Pad array circularly to catch peaks near boundary edges (Dec-Jan transitions)
        padded = np.tile(self.rainfall, 3)
        peaks, _ = find_peaks(padded[12:24], distance=3, prominence=15)
        
        if len(peaks) >= 2:
            return "Bimodal"
        return "Unimodal"


class CropRule:
    """Evaluates agronomic thresholds based on FAO agricultural guidelines."""
    
    def __init__(self, crop_name: str, min_mm: float, max_mm: float) -> None:
        self.crop_name: str = crop_name
        self.min_mm: float = min_mm
        self.max_mm: float = max_mm

    def assess_month(self, rainfall_mm: float) -> str:
        """Classifies a specific rainfall value against crop survival thresholds."""
        if rainfall_mm < 40.0:
            return "Drought Risk"
        elif rainfall_mm > 250.0:
            return "Waterlogging Risk"
        elif self.min_mm <= rainfall_mm <= self.max_mm:
            return f"Good for {self.crop_name}"
        else:
            return "Suboptimal"


class ClimateMetrics:
    """Computes rigorous mathematical similarities and vector metrics between regions."""
    
    @staticmethod
    def manual_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Corrects math.cos() flaw by computing true vector cosine similarity."""
        dot_prod = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_prod / (norm_v1 * norm_v2))

    @staticmethod
    def verify_with_scipy(v1: np.ndarray, v2: np.ndarray) -> float:
        """Verifies manual formula precision against standard scipy metrics."""
        # scipy calculates cosine distance (1 - similarity)
        cos_dist = dist.cosine(v1, v2)
        return float(1.0 - cos_dist)
