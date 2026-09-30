import pytest
import numpy as np
from src.fish_model import FishStock, PriceModel, RiskAssessor

def test_fibonacci_length():
    """Verifies baseline generation works accurately."""
    seq = FishStock.generate_fibonacci_baseline(15)
    assert len(seq) == 15
    assert seq[0] == 1
    assert seq[-1] == 610

def test_logistic_growth_bounds():
    """Edge Case: Ensures fish stock never drops below zero even with overharvesting."""
    stock_model = FishStock(r=0.4, K=10000, n0=4000)
    res = stock_model.simulate_trajectory(h=0.90, weeks=10) # 90% harvest rate is unsustainably high
    assert np.all(res["stock"] >= 0)

def test_price_bounds():
    """Ensures price walk strictly respects bounding limits."""
    pm = PriceModel(start_price=12000, lower_bound=9000, upper_bound=16000)
    path = pm.simulate_path(weeks=52, seed=42)
    assert np.all(path >= 9000)
    assert np.all(path <= 16000)
