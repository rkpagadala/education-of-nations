"""
Compute the education integral at every lag length (0-50 years) and report
the coefficient of variation (CV) across six countries at each lag.

The integral at lag L for a country with crossing date D is:
    sum of annual lower-secondary completion values from year (D-L) to year (D-1).
At lag 0: completion at the crossing year (single point).

Data sources (5-year intervals, linearly interpolated to annual):
    - cohort_lower_sec_both.csv  (1870-2015)
    - lower_sec_both.csv         (1950-2025)
The cohort file is preferred; the modern file extends it where needed.
"""

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────
COUNTRIES = {
    "Taiwan Province of China": {"crossing": 1972, "label": "Taiwan"},
    "Republic of Korea":        {"crossing": 1987, "label": "South Korea"},
    "Cuba":                     {"crossing": 1974, "label": "Cuba"},
    "Bangladesh":               {"crossing": 2014, "label": "Bangladesh"},
    "Sri Lanka":                {"crossing": 1993, "label": "Sri Lanka"},
    "China":                    {"crossing": 1994, "label": "China"},
}

MAX_LAG = 50

# ── Load and merge data ───────────────────────────────────────────────────
cohort = pd.read_csv(
    ""
    "wcde/data/processed/cohort_lower_sec_both.csv"
)
modern = pd.read_csv(
    ""
    "wcde/data/processed/lower_sec_both.csv"
)

def build_annual_series(country_name):
    """Build an annual interpolated series for a country from both CSVs."""
    # Get row from cohort file
    row_c = cohort[cohort["country"] == country_name]
    row_m = modern[modern["country"] == country_name]

    # Collect all (year, value) pairs from both sources
    points = {}
    for df in [row_c, row_m]:
        if df.empty:
            continue
        for col in df.columns:
            if col == "country":
                continue
            try:
                yr = int(col)
                val = float(df[col].values[0])
                # Cap at 100 (some data has >100 artifacts)
                val = min(val, 100.0)
                if yr not in points or df is row_c:
                    # Prefer cohort where overlapping, but take modern for new years
                    if yr not in points:
                        points[yr] = val
            except (ValueError, IndexError):
                continue

    # Also add modern-only years
    if not row_m.empty:
        for col in row_m.columns:
            if col == "country":
                continue
            try:
                yr = int(col)
                val = min(float(row_m[col].values[0]), 100.0)
                if yr not in points:
                    points[yr] = val
            except (ValueError, IndexError):
                continue

    years = sorted(points.keys())
    vals = [points[y] for y in years]

    # Interpolate to annual
    all_years = np.arange(years[0], years[-1] + 1)
    annual_vals = np.interp(all_years, years, vals)
    return dict(zip(all_years, annual_vals))


# ── Build annual series for each country ──────────────────────────────────
series = {}
for cname in COUNTRIES:
    series[cname] = build_annual_series(cname)

# ── Compute integrals at each lag ─────────────────────────────────────────
results = {}  # lag -> {country_label: integral}
for lag in range(0, MAX_LAG + 1):
    row = {}
    for cname, info in COUNTRIES.items():
        D = info["crossing"]
        s = series[cname]
        if lag == 0:
            # Single point: completion at crossing year
            integral = s.get(D, np.nan)
        else:
            # Sum from (D-lag) to (D-1)
            integral = 0.0
            for yr in range(D - lag, D):
                val = s.get(yr, None)
                if val is None:
                    integral = np.nan
                    break
                integral += val
        row[info["label"]] = integral
    results[lag] = row

# ── Compute CV at each lag ────────────────────────────────────────────────
labels = [info["label"] for info in COUNTRIES.values()]
lag_cv = {}
for lag in range(0, MAX_LAG + 1):
    vals = np.array([results[lag][lb] for lb in labels])
    if np.any(np.isnan(vals)) or np.mean(vals) == 0:
        lag_cv[lag] = np.nan
    else:
        lag_cv[lag] = np.std(vals, ddof=0) / np.mean(vals)

# ── Output ────────────────────────────────────────────────────────────────

# 1. Table every 5 years
print("=" * 60)
print("LAG (years)  vs  CV across 6 countries")
print("=" * 60)
print(f"{'Lag':>4s}  {'CV':>8s}")
print("-" * 14)
for lag in range(0, MAX_LAG + 1, 5):
    cv = lag_cv[lag]
    if np.isnan(cv):
        print(f"{lag:4d}       NaN")
    else:
        print(f"{lag:4d}  {cv:8.4f}")

# 2. Minimum CV
valid_lags = {k: v for k, v in lag_cv.items() if not np.isnan(v)}
if valid_lags:
    best_lag = min(valid_lags, key=valid_lags.get)
    print(f"\n{'=' * 60}")
    print(f"MINIMUM CV = {lag_cv[best_lag]:.4f}  at lag = {best_lag} years")
    print(f"{'=' * 60}")

    # 3. Integral values at optimal lag
    print(f"\nIntegral values at lag {best_lag} (pp-years):")
    print(f"{'Country':<15s} {'Integral':>12s}")
    print("-" * 28)
    for lb in labels:
        print(f"{lb:<15s} {results[best_lag][lb]:12.1f}")
    vals_best = np.array([results[best_lag][lb] for lb in labels])
    print(f"{'Mean':<15s} {np.mean(vals_best):12.1f}")
    print(f"{'Std':<15s} {np.std(vals_best, ddof=0):12.1f}")
    print(f"{'CV':<15s} {lag_cv[best_lag]:12.4f}")

# 4. CV at lag 0 for comparison
print(f"\n{'=' * 60}")
print(f"CV at lag 0 (simple threshold) = {lag_cv[0]:.4f}")
print(f"{'=' * 60}")
print(f"\nThreshold values at lag 0 (completion % at crossing year):")
print(f"{'Country':<15s} {'Value':>8s}")
print("-" * 24)
for lb in labels:
    print(f"{lb:<15s} {results[0][lb]:8.1f}")

# 5. Full table (every year) for reference
print(f"\n{'=' * 60}")
print("FULL TABLE: lag vs CV (every year)")
print(f"{'=' * 60}")
print(f"{'Lag':>4s}  {'CV':>8s}")
print("-" * 14)
for lag in range(0, MAX_LAG + 1):
    cv = lag_cv[lag]
    if np.isnan(cv):
        print(f"{lag:4d}       NaN")
    else:
        print(f"{lag:4d}  {cv:8.4f}")
