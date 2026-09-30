"""
Micro-Grid Dispatch Planner for Kasese Health Centre.
Provides linear system solvers, optimization, data validation, and sensitivity analyses.
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import nnls
from typing import Tuple, Union, Dict, Any, List

class MicroGrid:
    """Models a 2D Solar and Battery Micro-Grid system."""
    
    def __init__(self, solar_cost: float = 150.0, battery_cost: float = 450.0) -> None:
        """Initializes the grid with a static coefficient matrix and tariff costs."""
        # 3x + 2y = D1, 4x + y = D2
        self.A: np.ndarray = np.array([[3.0, 2.0], 
                                        [4.0, 1.0]], dtype=float)
        self.solar_cost: float = solar_cost
        self.battery_cost: float = battery_cost

    @property
    def determinant(self) -> float:
        """Computes the determinant of the coefficient matrix."""
        return float(la.det(self.A))

    @property
    def condition_number(self) -> float:
        """Computes the L2 condition number of the coefficient matrix."""
        return float(np.linalg.cond(self.A))

    def solve_day(self, d1: float, d2: float, strategy: str = "nnls") -> Dict[str, Any]:
        """Solves the energy allocation for a single day given D1 and D2 constraints."""
        b = np.array([d1, d2], dtype=float)
        
        if strategy == "exact":
            x, y = la.solve(self.A, b)
            feasible = (x >= 0 and y >= 0)
        elif strategy == "nnls":
            sol, rnorm = nnls(self.A, b)
            x, y = sol, sol
            feasible = np.allclose(self.A @ sol, b, rtol=1e-5)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        cost = (x * self.solar_cost) + (y * self.battery_cost)
        return {"solar": x, "battery": y, "feasible": feasible, "cost": cost}

    def solve_vectorized(self, B: np.ndarray) -> np.ndarray:
        """Solves a 2xN matrix of demands efficiently using a single vectorized linear call."""
        return la.solve(self.A, B)

    @staticmethod
    def validate_interactive_input() -> Tuple[float, float]:
        """Provides robust terminal/interactive validation for user inputs."""
        while True:
            try:
                d1_str = input("Enter Daytime Load D1 (kWh): ").strip()
                d2_str = input("Enter Critical Equipment Load D2 (kWh): ").strip()
                if not d1_str or not d2_str:
                    print("Error: Inputs cannot be empty.")
                    continue
                d1, d2 = float(d1_str), float(d2_str)
                if d1 < 0 or d2 < 0:
                    print("Error: Energy demands cannot be negative.")
                    continue
                return d1, d2
            except ValueError:
                print("Error: Non-numeric values entered.")

    def run_sensitivity(self, d1: float, d2: float, draws: int = 1000, seed: int = 2026) -> Dict[str, np.ndarray]:
        """Performs a Monte Carlo sensitivity analysis with a fixed seed."""
        rng = np.random.default_rng(seed=seed)
        p_d1 = d1 * rng.uniform(0.95, 1.05, draws)
        p_d2 = d2 * rng.uniform(0.95, 1.05, draws)
        B_perturbed = np.vstack([p_d1, p_d2])
        X_perturbed = self.solve_vectorized(B_perturbed)
        return {"solar": X_perturbed[0, :], "battery": X_perturbed[1, :]}


class HybridMicroGrid(MicroGrid):
    """Extends MicroGrid to model a 3D system including a Diesel Generator (z)."""
    
    def __init__(self, solar_cost: float = 150.0, battery_cost: float = 450.0, diesel_cost: float = 1200.0) -> None:
        super().__init__(solar_cost, battery_cost)
        self.diesel_cost: float = diesel_cost
        self.A_3d: np.ndarray = np.array([
            [3.0, 2.0, 1.0],
            [4.0, 1.0, 2.0],
            [1.0, 1.0, 3.0]
        ], dtype=float)

    def solve_hybrid_day(self, d1: float, d2: float, d3: float) -> Dict[str, Any]:
        """Solves the 3x3 linear system for the hybrid configuration."""
        b = np.array([d1, d2, d3], dtype=float)
        try:
            det = float(la.det(self.A_3d))
            if np.isclose(det, 0.0):
                raise la.LinAlgError("Matrix is singular.")
            x, y, z = la.solve(self.A_3d, b)
            cost = (x * self.solar_cost) + (y * self.battery_cost) + (z * self.diesel_cost)
            return {"solar": x, "battery": y, "diesel": z, "cost": cost, "det": det}
        except la.LinAlgError as e:
            return {"error": str(e), "det": 0.0}
