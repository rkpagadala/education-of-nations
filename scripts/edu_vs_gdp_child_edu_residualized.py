"""
edu_vs_gdp_child_edu_residualized.py
=====================================
Does GDP independently predict intergenerational education transmission?

Parent education(T) → Child education(T+25): the PTE mechanism.
GDP(T) → Child education(T+25): the income-led hypothesis.
Residualized GDP(T) → Child education(T+25): GDP's independent effect.

If residualized GDP ≈ 0, education transmits itself without income.
Completes the triangle: LE (done), TFR (done), child education.
"""

import os, sys, json
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared import *

EDU_LEVELS = {"primary": "primary", "lower_secondary": "lower_sec", "upper_secondary": "upper_sec"}
T_YEARS = list(range(1960, 1995, 5))
LAG = 25
CEILINGS = [50, 60, 70, 80, 90]

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")

all_results = {}

for level_name, col_name in EDU_LEVELS.items():
    print(f"\n{'#' * 90}")
    print(f"# {level_name.upper().replace('_', ' ')}  →  CHILD EDUCATION(T+25)")
    print(f"{'#' * 90}")

    edu_annual = interpolate_to_annual(edu_raw, col_name)

    # Build panel: parent edu at T, child edu at T+25 (same column, later year)
    rows = []
    for c in sorted(edu_annual.keys()):
        s = edu_annual[c]
        for t in T_YEARS:
            if t not in s.index or (t + LAG) not in s.index:
                continue
            parent_edu = s[t]
            child_edu = s[t + LAG]
            gdp_t = get_wb_val(gdp_df, c, t)
            if np.isnan(parent_edu) or np.isnan(child_edu):
                continue
            rows.append({
                "country": c, "t": t, "edu_t": parent_edu,
                "log_gdp_t": np.log(gdp_t) if not np.isnan(gdp_t) and gdp_t > 0 else np.nan,
                "child_edu": child_edu,
            })

    panel = pd.DataFrame(rows)
    print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")

    entry_years = precompute_entry_years(edu_annual)
    results = run_residualized_sweep(panel, entry_years, "child_edu", CEILINGS, label="ChildEdu")
    print_summary(results, CEILINGS, label="ChildEdu")
    all_results[level_name] = results

# Cross-level
for ceiling in [60, 90]:
    print(f"\n{'=' * 90}")
    print(f"CROSS-LEVEL: entry=10%, ceiling={ceiling}%  →  Child Education(T+25)")
    print(f"{'=' * 90}")
    print(f"{'Level':<20} {'Edu→CE':>7} {'GDP→CE':>7} {'Resid→CE':>9} {'Edu→GDP':>8}")
    print("-" * 55)
    for level_name in EDU_LEVELS:
        r = all_results[level_name].get(str(ceiling), {}).get("10", {})
        if r and r.get("edu_r2") is not None:
            def fmt(v): return f"{v:.3f}" if v is not None else "n/a"
            print(f"{level_name:<20} {fmt(r.get('edu_r2')):>7} {fmt(r.get('raw_gdp_r2')):>7} "
                  f"{fmt(r.get('resid_gdp_r2')):>9} {fmt(r.get('edu_gdp_r2')):>8}")

checkin = {
    "script": "scripts/edu_vs_gdp_child_edu_residualized.py",
    "produced": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "method": "Country FE, residualized GDP. Parent edu(T) → child edu(T+25). Entry-cohort + ceiling. T=1960-1990, lag=25.",
    "levels": all_results,
}
os.makedirs(CHECKIN, exist_ok=True)
with open(os.path.join(CHECKIN, "edu_vs_gdp_child_edu_residualized.json"), "w") as f:
    json.dump(checkin, f, indent=2)
print(f"\nCheckin written.")
