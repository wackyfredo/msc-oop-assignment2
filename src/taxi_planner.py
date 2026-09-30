"""
Taxi Route Revenue, Pricing & Fleet Planner for Kampala Matatu Association.
Implements multi-model forecasting pipelines, microeconomic equilibrium solvers, and fleet sizing optimizations.
"""

import numpy as np
import scipy.linalg as la
import statistics as stats
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Route:
    """Models a single matatu transit route with passenger counts and pricing structures."""
    
    def __init__(self, name: str, passengers: List[int], fare: float) -> None:
        self.name: str = name
        self.passengers: np.ndarray = np.array(passengers, dtype=float)
        self.fare: float = fare

    @property
    def daily_revenue(self) -> np.ndarray:
        return self.passengers * self.fare

    @property
    def total_revenue(self) -> float:
        return float(np.sum(self.daily_revenue))

    def compute_statistics(self) -> Dict[str, float]:
        """Calculates central tendency and dispersion metrics using statistics package."""
        data = list(self.passengers)
        return {
            "mean": stats.mean(data),
            "variance": stats.variance(data) if len(data) > 1 else 0.0,
            "stdev": stats.stdev(data) if len(data) > 1 else 0.0
        }

    @staticmethod
    def solve_equilibrium() -> Dict[str, float]:
        """
        Solves the supply-demand linear system for the Ntinda route:
        Q + 0.02P = 120  (Demand)
        Q - 0.03P = 10   (Supply)
        """
        A = np.array([[1.0, 0.02], 
                      [1.0, -0.03]], dtype=float)
        b = np.array([120.0, 10.0], dtype=float)
        sol = la.solve(A, b)
        return {"Q_equilibrium": float(sol[0]), "P_equilibrium": float(sol[1])}


class Forecaster(ABC):
    """Abstract base class modeling time-series forecasting frameworks."""
    
    @abstractmethod
    def predict(self, history: np.ndarray) -> float:
        """Generates a one-step-ahead forward forecast based on history."""
        pass


class MovingAverageForecaster(Forecaster):
    """Implements a standard rolling rolling-window moving average model."""
    
    def __init__(self, window: int = 3) -> None:
        self.window = window

    def predict(self, history: np.ndarray) -> float:
        if len(history) < self.window:
            return float(np.mean(history))
        return float(np.mean(history[-self.window:]))


class ExponentialSmoothingForecaster(Forecaster):
    """Implements Simple Exponential Smoothing (SES) with a smoothing parameter alpha."""
    
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = np.clip(alpha, 0.0, 1.0)

    def predict(self, history: np.ndarray) -> float:
        if len(history) == 0:
            return 0.0
        level = history[0]
        for val in history[1:]:
            level = self.alpha * val + (1.0 - self.alpha) * level
        return float(level)


class LinearTrendForecaster(Forecaster):
    """Implements a linear trend forecast model using a least-squares polynomial fit."""
    
    def predict(self, history: np.ndarray) -> float:
        n = len(history)
        if n < 2:
            return float(history[-1]) if n == 1 else 0.0
        t = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(t, history, 1)
        return float(slope * n + intercept)


class Backtester:
    """Evaluates time-series models via rolling-origin walk-forward validations."""
    
    @staticmethod
    def evaluate_mae(forecaster: Forecaster, data: np.ndarray, start_day: int = 4) -> float:
        errors = []
        # Walk-forward loop over days 4-10 (0-indexed indices 3 to 9)
        for i in range(start_day - 1, len(data)):
            history = data[:i]
            actual = data[i]
            pred = forecaster.predict(history)
            errors.append(abs(actual - pred))
        return float(np.mean(errors)) if errors else 0.0
