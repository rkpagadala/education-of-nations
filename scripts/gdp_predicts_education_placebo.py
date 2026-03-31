"""
gdp_predicts_education_placebo.py
==================================
Reverse direction test: does GDP predict future education?

If education is upstream of GDP, then:
  - education(T) → GDP(T+25) should be strong
  - GDP(T) → education(T+25) should be weak

This is essentially a Granger-style causality test with country FE.
Also tests GDP(T) → LE/TFR/U5MR(T+25) vs education(T) → same.

Lower secondary, entry=10%, ceilings 60/90.
"""

import os, sys, json
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared import *

T_YEARS = list(range(1960, 1995, 5))
LAG = 25
CEILINGS = [60, 90]
COL_NAME = "lower_sec"

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df = load_wb("life_expectancy_years.csv")
tfr_df = load_wb("children_per_woman_total_fertility.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

edu_annual = interpolate_to_annual(edu_raw, COL_NAME)
entry_years = precompute_entry_years(edu_annual)

results = {}

# ── Test 1: edu(T) → GDP(T+25) vs GDP(T) → edu(T+25) ───────────────

print(f"\n{'=' * 90}")
print(f"DIRECTION TEST: Who predicts whom 25 years later?")
print(f"{'=' * 90}")

# Build panel with both future GDP and future education
rows = []
for c in sorted(edu_annual.keys()):
    s = edu_annual[c]
    for t in T_YEARS:
        if t not in s.index or (t + LAG) not in s.index:
            continue
        edu_t = s[t]
        edu_future = s[t + LAG]
        gdp_t = get_wb_val(gdp_df, c, t)
        gdp_future = get_wb_val(gdp_df, c, t + LAG)
        le_future = get_wb_val(le_df, c, t + LAG)
        tfr_future = get_wb_val(tfr_df, c, t + LAG)
        u5mr_future = get_wb_val(u5mr_df, c, t + LAG)
        if np.isnan(edu_t):
            continue
        rows.append({
            "country": c, "t": t, "edu_t": edu_t,
            "log_gdp_t": np.log(gdp_t) if not np.isnan(gdp_t) and gdp_t > 0 else np.nan,
            "edu_future": edu_future,
            "log_gdp_future": np.log(gdp_future) if not np.isnan(gdp_future) and gdp_future > 0 else np.nan,
            "le_future": le_future,
            "tfr_future": tfr_future,
            "u5mr_future": u5mr_future,
        })

panel = pd.DataFrame(rows)
print(f"Panel: {len(panel)} obs, {panel['country'].nunique()} countries")

for ceiling in CEILINGS:
    print(f"\n  Ceiling = {ceiling}%")
    print(f"  {'Direction':<40} {'R²':>7} {'n':>5} {'Ctry':>5}")
    print(f"  {'-' * 65}")

    cohort = entry_years.get(10, {})
    sub = filter_panel(panel, cohort, ceiling)

    ceil_results = {}

    # Forward: edu(T) → outcomes(T+25)
    for label, col in [("edu(T) → edu(T+25)", "edu_future"),
                       ("edu(T) → GDP(T+25)", "log_gdp_future"),
                       ("edu(T) → LE(T+25)", "le_future"),
                       ("edu(T) → TFR(T+25)", "tfr_future"),
                       ("edu(T) → U5MR(T+25)", "u5mr_future")]:
        r2, n, nc = fe_r2("edu_t", col, sub)
        def fmt(v): return f"{v:.3f}" if not np.isnan(v) else "n/a"
        print(f"  {label:<40} {fmt(r2):>7} {n:>5} {nc:>5}")
        ceil_results[label] = round(r2, 3) if not np.isnan(r2) else None

    print()

    # Reverse: GDP(T) → outcomes(T+25)
    for label, col in [("GDP(T) → edu(T+25)", "edu_future"),
                       ("GDP(T) → GDP(T+25)", "log_gdp_future"),
                       ("GDP(T) → LE(T+25)", "le_future"),
                       ("GDP(T) → TFR(T+25)", "tfr_future"),
                       ("GDP(T) → U5MR(T+25)", "u5mr_future")]:
        r2, n, nc = fe_r2("log_gdp_t", col, sub)
        def fmt(v): return f"{v:.3f}" if not np.isnan(v) else "n/a"
        print(f"  {label:<40} {fmt(r2):>7} {n:>5} {nc:>5}")
        ceil_results[label] = round(r2, 3) if not np.isnan(r2) else None

    results[str(ceiling)] = ceil_results

checkin = {
    "script": "scripts/gdp_predicts_education_placebo.py",
    "produced": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "method": "Country FE. Granger-style direction test. edu(T)→X(T+25) vs GDP(T)→X(T+25). Lower secondary, entry=10%.",
    "results": results,
}
os.makedirs(CHECKIN, exist_ok=True)
with open(os.path.join(CHECKIN, "gdp_predicts_education_placebo.json"), "w") as f:
    json.dump(checkin, f, indent=2)
print(f"\nCheckin written.")
