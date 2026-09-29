import numpy as np
import statistics
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# ==========================================
# 1. CLASS DESIGN & DATA SETUP
# ==========================================
class DistrictPopulation:
    def __init__(self, name: str, years: np.ndarray, populations: np.ndarray):
        self.name = name
        self.years = np.asarray(years, dtype=int)
        self.populations = np.asarray(populations, dtype=float)
        self._validate()

    def _validate(self):
        if len(self.years) != len(self.populations):
            raise ValueError("Years and populations arrays must have equal length.")
        if np.any(self.populations < 0):
            raise ValueError("Population values cannot be negative.")

    def __len__(self):
        return len(self.years)

    def __repr__(self):
        return f"DistrictPopulation(name='{self.name}', years={len(self)} records)"

# Dataset Initialization (Values in thousands)
years_data = np.arange(2015, 2025)

# Added Mbarara and Jinja as two extra districts
districts_data = {
    "Kampala": [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800],
    "Wakiso":  [950, 1000, 1070, 1150, 1220, 1300, 1390, 1480, 1570, 1670],
    "Gulu":    [320, 330, 345, 360, 375, 390, 410, 430, 455, 480],
    "Mbarara": [400, 415, 430, 450, 470, 490, 515, 540, 570, 600],
    "Jinja":   [280, 285, 290, 295, 300, 304, 309, 314, 319, 324]
}

districts = {name: DistrictPopulation(name, years_data, pops) for name, pops in districts_data.items()}

# ==========================================
# 4. FORECASTING SYSTEM ARCHITECTURE
# ==========================================
class Forecaster(ABC):
    @abstractmethod
    def fit(self, years: np.ndarray, pops: np.ndarray):
        pass

    @abstractmethod
    def predict(self, horizon_years: np.ndarray) -> np.ndarray:
        pass

class LinearTrendForecaster(Forecaster):
    def fit(self, years, pops):
        self.coeffs = np.polyfit(years, pops, 1)
    def predict(self, horizon_years):
        return np.polyval(self.coeffs, horizon_years)

class ExponentialCAGRForecaster(Forecaster):
    def fit(self, years, pops):
        self.t0 = years[0]
        self.p0 = pops[0]
        # Calculate CAGR over fit window
        self.cagr = (pops[-1] / pops[0]) ** (1 / (years[-1] - years[0])) - 1
    def predict(self, horizon_years):
        return self.p0 * ((1 + self.cagr) ** (horizon_years - self.t0))

class FibonacciRatioForecaster(Forecaster):
    def fit(self, years, pops):
        self.last_year = years[-1]
        self.last_pop = pops[-1]
        # Traditional golden ratio steps sequence
        self.fib_ratios = [1.01, 1.02, 1.03, 1.05, 1.08, 1.13, 1.21]
    def predict(self, horizon_years):
        predictions = []
        for yr in horizon_years:
            idx = int(yr - self.last_year) - 1
            idx = min(max(0, idx), len(self.fib_ratios) - 1)
            predictions.append(self.last_pop * self.fib_ratios[idx])
        return np.array(predictions)

# ==========================================
# 5. VALIDATION ENGINE (Train: 2015-2021, Test: 2022-2024)
# ==========================================
train_mask = years_data <= 2021
test_mask = years_data > 2021

train_years, test_years = years_data[train_mask], years_data[test_mask]
future_years = np.arange(2025, 2030)

best_models = {}

print("--- MODEL BACKTESTING METRICS ---")
for name, dist in districts.items():
    y_train, y_test = dist.populations[train_mask], dist.populations[test_mask]
    
    models = {
        "Linear Trend": LinearTrendForecaster(),
        "Exponential/CAGR": ExponentialCAGRForecaster(),
        "Fibonacci-Ratio": FibonacciRatioForecaster()
    }
    
    best_mape = float('inf')
    best_model_name = ""
    
    print(f"\nDistrict: {name}")
    print(f"{'Model':<20} | {'MAE':<8} | {'RMSE':<8} | {'MAPE (%)':<8}")
    print("-" * 52)
    
    for m_name, model in models.items():
        model.fit(train_years, y_train)
        preds = model.predict(test_years)
        
        mae = np.mean(np.abs(y_test - preds))
        rmse = np.sqrt(np.mean((y_test - preds) **2))
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        
        print(f"{m_name:<20} | {mae:<8.2f} | {rmse:<8.2f} | {mape:<8.2f}%")
        
        if mape < best_mape:
            best_mape = mape
            best_model_name = m_name
            
    best_models[name] = best_model_name
    print(f"Selected Model: {best_model_name}")

# ==========================================
# 6 & 7. FUTURE FORECAST & VISUALIZATION (With Bootstrapping)
# ==========================================
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
axes = axes.flatten()

print("\n--- 2029 CLASSROOM INFRASTRUCTURE REQUIREMENT PLANNING ---")

for idx, (name, dist) in enumerate(districts.items()):
    ax = axes[idx]
    
    # Refit chosen architecture on complete 2015-2024 tracking layout
    chosen_name = best_models[name]
    if chosen_name == "Linear Trend":
        model = LinearTrendForecaster()
    elif chosen_name == "Exponential/CAGR":
        model = ExponentialCAGRForecaster()
    else:
        model = FibonacciRatioForecaster()
        
    model.fit(dist.years, dist.populations)
    fitted_vals = model.predict(dist.years)
    forecast_vals = model.predict(future_years)
    
    # Extension: Residual Bootstrap Resampling (1000 Iterations)
    residuals = dist.populations - fitted_vals
    bootstrap_forecasts = []
    np.random.seed(42)
    
    for _ in range(1000):
        boot_res = np.random.choice(residuals, size=len(future_years), replace=True)
        bootstrap_forecasts.append(forecast_vals + boot_res)
        
    lower_bound = np.percentile(bootstrap_forecasts, 2.5, axis=0)
    upper_bound = np.percentile(bootstrap_forecasts, 97.5, axis=0)
    
    # Plotting Code Execution
    ax.plot(dist.years, dist.populations, 'ko-', label='Actual Historical Data', linewidth=2)
    ax.plot(dist.years, fitted_vals, 'b--', label='Model Fit Curve')
    ax.plot(future_years, forecast_vals, 'r-^', label='5-Year Forward Forecast')
    ax.fill_between(future_years, lower_bound, upper_bound, color='r', alpha=0.15, label='95% Bootstrap Confidence Interval')
    
    ax.axvline(x=2024, color='gray', linestyle=':', label='Forecast Threshold Window')
    ax.set_title(f"{name} Population Overview ({chosen_name})")
    ax.set_xlabel("Planning Calendar Year")
    ax.set_ylabel("Population Metrics (Thousands)")
    ax.legend(loc="upper left", fontsize='small')
    ax.grid(True, alpha=0.3)
    
    # 8. Classroom Infrastructure Calculations
    final_2024_pop = dist.populations[-1] * 1000
    forecast_2029_pop = forecast_vals[-1] * 1000
    
    added_population = forecast_2029_pop - final_2024_pop
    additional_pupils = added_population * 0.18
    classrooms_needed = int(np.ceil(additional_pupils / 53))
    
    print(f"District: {name:<10} | Added Pop: {added_population/1000:<6.1f}k | Classrooms Required by 2029: {max(0, classrooms_needed)}")

# Clean up empty subplots frame elements
fig.delaxes(axes[-1])
plt.tight_layout()
plt.show()

