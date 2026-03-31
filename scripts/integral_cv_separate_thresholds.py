"""
Compute the education INTEGRAL at every lag length from 0 to 50 years,
separately for TFR < 3.65 crossing and LE > 69.8 crossing.

Integral at lag L for country with crossing date D:
    sum of annual lower-secondary completion from year (D-L) to (D-1).

Reports CV across 6 countries at each lag, finds the minimum-CV lag,
and compares with point-estimate CV at the crossing date.
"""

import pandas as pd
import numpy as np

# ── Data ────────────────────────────────────────────────────────────────────
DATA = "wcde/data/processed/cohort_lower_sec_both.csv"

df = pd.read_csv(DATA)
df = df.set_index("country")
df.columns = df.columns.astype(int)  # year columns as int

# ── Country name mapping ────────────────────────────────────────────────────
NAME_MAP = {
    "Taiwan": "Taiwan Province of China",
    "South Korea": "Republic of Korea",
    "Cuba": "Cuba",
    "Bangladesh": "Bangladesh",
    "Sri Lanka": "Sri Lanka",
    "China": "China",
}

COUNTRIES = list(NAME_MAP.keys())

# ── Crossing dates ──────────────────────────────────────────────────────────
TFR_CROSSING = {
    "Taiwan": 1972,
    "South Korea": 1975,
    "Cuba": 1972,
    "Bangladesh": 1995,
    "Sri Lanka": 1981,
    "China": 1975,
}

LE_CROSSING = {
    "Taiwan": 1972,
    "South Korea": 1987,
    "Cuba": 1974,
    "Bangladesh": 2014,
    "Sri Lanka": 1993,
    "China": 1994,
}

# ── Build annual interpolated series for each country ───────────────────────
def interpolate_annual(row):
    """Linearly interpolate 5-year data to annual."""
    years = row.index.values
    vals = row.values.astype(float)
    mask = ~np.isnan(vals)
    years_valid = years[mask]
    vals_valid = vals[mask]
    annual_years = np.arange(years_valid.min(), years_valid.max() + 1)
    annual_vals = np.interp(annual_years, years_valid, vals_valid)
    return pd.Series(annual_vals, index=annual_years)

annual = {}
for short, full in NAME_MAP.items():
    annual[short] = interpolate_annual(df.loc[full])

# ── Compute integral at lag L for a country with crossing date D ────────────
def compute_integral(country, crossing, lag):
    """Sum of annual completion from year (crossing-lag) to (crossing-1)."""
    if lag == 0:
        return 0.0
    s = annual[country]
    start = crossing - lag
    end = crossing - 1
    # Clip to available range
    start = max(start, s.index.min())
    end = min(end, s.index.max())
    if start > end:
        return np.nan
    return s.loc[start:end].sum()

def point_estimate(country, crossing):
    """Completion at crossing year."""
    s = annual[country]
    if crossing in s.index:
        return s.loc[crossing]
    return np.nan

# ── Compute CV across 6 countries at each lag ──────────────────────────────
MAX_LAG = 50

def compute_cv_table(crossings, label):
    results = []  # (lag, cv, {country: integral})
    for lag in range(1, MAX_LAG + 1):
        vals = {}
        for c in COUNTRIES:
            vals[c] = compute_integral(c, crossings[c], lag)
        arr = np.array(list(vals.values()))
        if np.any(np.isnan(arr)) or np.std(arr) == 0:
            cv = np.nan
        else:
            cv = np.std(arr, ddof=0) / np.mean(arr)
        results.append((lag, cv, vals))
    return results

def print_results(crossings, label):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    # Point estimate CV
    point_vals = {c: point_estimate(c, crossings[c]) for c in COUNTRIES}
    arr_p = np.array(list(point_vals.values()))
    cv_point = np.std(arr_p, ddof=0) / np.mean(arr_p)
    print(f"\nPoint-estimate CV (completion at crossing year): {cv_point:.4f}")
    for c in COUNTRIES:
        print(f"  {c:15s}  crossing={crossings[c]}  completion={point_vals[c]:.1f}%")

    # Integral CV table
    results = compute_cv_table(crossings, label)

    # Print every 5 years
    print(f"\n{'Lag':>5s}  {'CV':>8s}  ", end="")
    for c in COUNTRIES:
        print(f"{c:>12s}", end="")
    print()
    print("-" * 85)
    for lag, cv, vals in results:
        if lag % 5 == 0 or lag == 1:
            print(f"{lag:5d}  {cv:8.4f}  ", end="")
            for c in COUNTRIES:
                print(f"{vals[c]:12.1f}", end="")
            print()

    # Minimum CV lag
    valid = [(lag, cv, vals) for lag, cv, vals in results if not np.isnan(cv)]
    if valid:
        best_lag, best_cv, best_vals = min(valid, key=lambda x: x[1])
        print(f"\nMinimum CV = {best_cv:.4f} at lag = {best_lag} years")
        print(f"Country integrals at optimal lag ({best_lag} years):")
        for c in COUNTRIES:
            print(f"  {c:15s}  integral = {best_vals[c]:8.1f}  "
                  f"(window {crossings[c]-best_lag}–{crossings[c]-1})")
    return cv_point, best_cv, best_lag

cv_point_tfr, cv_int_tfr, lag_tfr = print_results(TFR_CROSSING, "TFR < 3.65 crossing")
cv_point_le, cv_int_le, lag_le = print_results(LE_CROSSING, "LE > 69.8 crossing")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  SUMMARY: Which combination produces the tightest convergence?")
print(f"{'='*80}")
print(f"\n{'Threshold':<12s}  {'Method':<18s}  {'CV':>8s}  {'Lag':>5s}")
print("-" * 50)
print(f"{'TFR<3.65':<12s}  {'point estimate':<18s}  {cv_point_tfr:8.4f}  {'  0':>5s}")
print(f"{'TFR<3.65':<12s}  {'integral':<18s}  {cv_int_tfr:8.4f}  {lag_tfr:5d}")
print(f"{'LE>69.8':<12s}  {'point estimate':<18s}  {cv_point_le:8.4f}  {'  0':>5s}")
print(f"{'LE>69.8':<12s}  {'integral':<18s}  {cv_int_le:8.4f}  {lag_le:5d}")

best_combo = min([
    ("TFR point", cv_point_tfr),
    ("TFR integral", cv_int_tfr),
    ("LE point", cv_point_le),
    ("LE integral", cv_int_le),
], key=lambda x: x[1])

print(f"\nTightest convergence: {best_combo[0]} with CV = {best_combo[1]:.4f}")
