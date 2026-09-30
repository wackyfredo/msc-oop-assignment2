"""
Lake Victoria Fish Stock & Export Risk Model (Jinja Context).
Implements logistic population dynamics, random walk pricing models, and Value-at-Risk optimization.
"""

import numpy as np
import statistics as stats
from typing import List, Dict, Any

class FishStock:
    """Simulates discrete logistic biological growth with proportional harvesting."""
    
    def __init__(self, r: float = 0.4, K: float = 10000.0, n0: float = 4000.0) -> None:
        self.r = r  # Intrinsic growth rate
        self.K = K  # Carrying capacity (tonnes)
        self.n0 = n0  # Initial stock (tonnes)

    @staticmethod
    def generate_fibonacci_baseline(n: int = 15) -> List[int]:
        """Generates a raw Fibonacci sequence to serve as a flawed baseline."""
        seq = [1, 1]
        for _ in range(2, n):
            seq.append(seq[-1] + seq[-2])
        return seq

    def simulate_trajectory(self, h: float, weeks: int = 52, closed_weeks: int = 0) -> Dict[str, np.ndarray]:
        """Simulates 52-week population dynamics under variable harvest rates."""
        stock = np.zeros(weeks + 1)
        harvests = np.zeros(weeks)
        stock[0] = self.n0
        
        for t in range(weeks):
            current_n = stock[t]
            # Manage closed season (e.g., no harvesting for the first 8 weeks if closed_weeks=8)
            current_h = 0.0 if t < closed_weeks else h
            
            # Logistic growth formula with active harvesting constraint
            growth = self.r * current_n * (1.0 - (current_n / self.K))
            harvest = current_h * current_n
            
            harvests[t] = harvest
            stock[t+1] = max(0.0, current_n + growth - harvest)
            
        return {"stock": stock, "harvests_tonnes": harvests}


class PriceModel:
    """Models a bounded random walk for Nile Perch export prices in UGX/kg."""
    
    def __init__(self, start_price: float = 12000.0, lower_bound: float = 9000.0, upper_bound: float = 16000.0) -> None:
        self.start_price = start_price
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def simulate_path(self, weeks: int = 52, seed: int = None) -> np.ndarray:
        """Generates a seeded weekly price random walk trajectory."""
        rng = np.random.default_rng(seed=seed)
        prices = np.zeros(weeks)
        current_price = self.start_price
        
        for t in range(weeks):
            # Price shocks vary by +/- UGX 500 per week
            shock = rng.normal(0, 500)
            current_price = np.clip(current_price + shock, self.lower_bound, self.upper_bound)
            prices[t] = current_price
            
        return prices


class RiskAssessor:
    """Evaluates risk profiling using Coefficient of Variation and Value-at-Risk."""
    
    @staticmethod
    def classify_risk(cv: float) -> str:
        """Classifies operational risk dynamically based on Coefficient of Variation."""
        if cv < 0.10:
            return "Low Risk"
        elif cv <= 0.25:
            return "Moderate Risk"
        return "High Risk"

    @staticmethod
    def calculate_var_95(revenues: np.ndarray) -> float:
        """Computes the 5% Value-at-Risk (VaR) threshold for total revenue distributions."""
        return float(np.percentile(revenues, 5))
