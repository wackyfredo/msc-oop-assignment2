import pytest
import numpy as np
from src.rainfall_analyser import Region, CropRule, ClimateMetrics

def test_region_statistical_metrics():
    """Tests basic summary statistics calculations on rainfall data arrays."""
    reg = Region("Test", [100]*12)
    assert reg.total_annual == 1200.0
    assert reg.mean_monthly == 100.0
    assert reg.coefficient_of_variation == 0.0

def test_cosine_similarity_correction():
    """Ensures manual vector computation matches scipy validation precisely."""
    v1 = np.array([120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130], dtype=float)
    v2 = np.array([70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90], dtype=float)
    
    manual = ClimateMetrics.manual_cosine_similarity(v1, v2)
    scipy_val = ClimateMetrics.verify_with_scipy(v1, v2)
    assert pytest.approx(manual, rel=1e-5) == scipy_val

def test_crop_rule_edge_cases():
    """Edge Case: Validates extreme boundaries (drought vs waterlogging anomalies)."""
    rule = CropRule("Maize", 60, 180)
    assert rule.assess_month(10.0) == "Drought Risk"
    assert rule.assess_month(300.0) == "Waterlogging Risk"
    assert rule.assess_month(100.0) == "Good for Maize"
