import pytest
import numpy as np
from src.taxi_planner import Route, MovingAverageForecaster, ExponentialSmoothingForecaster, LinearTrendForecaster

def test_market_equilibrium():
    """Validates systemic calculations for structural supply-demand equations."""
    eq = Route.solve_equilibrium()
    assert pytest.approx(eq["P_equilibrium"]) == 2200.0
    assert pytest.approx(eq["Q_equilibrium"]) == 76.0

def test_moving_average_prediction():
    """Confirms window mechanics generate correct rolling forecasts."""
    history = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    forecaster = MovingAverageForecaster(window=3)
    # Expected average of 20, 30, 40 is 30
    assert pytest.approx(forecaster.predict(history)) == 30.0

def test_forecaster_edge_cases():
    """Edge Case: Verifies models degrade gracefully when given minimal history profiles."""
    history = np.array([50.0], dtype=float)
    ses = ExponentialSmoothingForecaster(alpha=0.3)
    trend = LinearTrendForecaster()
    
    assert pytest.approx(ses.predict(history)) == 50.0
    assert pytest.approx(trend.predict(history)) == 50.0
