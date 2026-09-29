# ubos-population-forecaster
: 5-year population forecasting pipeline for UBOS district planning using NumPy and object-oriented Python.


## Written Project Analysis

### 📊 Task 2: Variance Discrepancy Breakdown (`statistics` vs. `NumPy`)
* **`statistics.variance`**: Computes **sample variance** using a divisor of N - 1. This is used when data is a sample of a larger population to prevent underestimating variance (Bessel's correction).
* **`np.var`**: Computes **population variance** by default, using a divisor of N. It assumes you possess the complete, absolute dataset.
* **The `ddof` (Delta Degrees of Freedom) Parameter**: This argument modifies the divisor to N - ddof. To force NumPy to produce the exact same sample variance as Python's native statistics module, you must set `ddof=1` inside the function: `np.var(array, ddof=1)`.

### 📈 Task 6: Evaluation of Actual vs. Forecast Variance
Comparing historical variance against projected model variance tells us how much "uncertainty" or structural trend change is embedded in our future estimates:
* **Linear Models**: Typically display **lower variance** than actual history because regression lines capture only the clean, deterministic trend while completely smoothing out historical "noise" and random yearly fluctuations.
* **Exponential/CAGR Models**: Display **much higher variance** over long time horizons. Because the values scale up exponentially, the absolute differences between points grow rapidly, causing variance to inflate quadratically over time.

### 🧐 Extension: Critical Assessment of the Fibonacci Model
The Fibonacci-ratio model scales forward projections strictly on an abstract mathematical integer chain (\(1, 2, 3, 5, 8, 13 \dots\)) or geometric golden ratios (1.618). 

* **Verdict**: It is **entirely indefensible** for professional civic engineering or municipal planning.
* **Why?**: Human populations grow based on clear, underlying biological and socio-economic variables: crude birth rates, local mortality rates, health infrastructure, and economic rural-to-urban migration. Population steps never mathematically conform to a rigid, recursive sequence of integers. Relying on it for infrastructure risks severe over-building or drastic under-funding.
A.I generated
