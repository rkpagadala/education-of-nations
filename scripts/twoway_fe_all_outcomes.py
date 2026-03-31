"""
twoway_fe_all_outcomes.py
==========================
Two-way fixed effects (country + time) residualization for all four outcomes:
  - Life expectancy (T+25)
  - TFR (T+25)
  - Child education (T+25)
  - Child mortality U5 (T+25)

Adding time FE rules out global trends (medical technology, green revolution)
as confounders. If the zero GDP result survives two-way FE, it's bulletproof.
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
le_df = load_wb("life_expectancy_years.csv")
tfr_df = load_wb("children_per_woman_total_fertility.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

OUTCOMES = {
    "LE": ("le_tp25", le_df),
    "TFR": ("tfr_tp25", tfr_df),
    "U5MR": ("u5mr_tp25", u5mr_df),
}

all_results = {}

for outcome_label, (outcome_col, outcome_df) in OUTCOMES.items():
    print(f"\n\n{'*' * 90}")
    print(f"* OUTCOME: {outcome_label}  (two-way FE: country + time)")
    print(f"{'*' * 90}")

    outcome_results = {}

    for level_name, col_name in EDU_LEVELS.items():
        print(f"\n{'#' * 90}")
        print(f"# {level_name.upper().replace('_', ' ')}  →  {outcome_label}(T+25)")
        print(f"{'#' * 90}")

        edu_annual = interpolate_to_annual(edu_raw, col_name)
        panel = build_panel(edu_annual, outcome_df, gdp_df, T_YEARS, LAG, outcome_col)
        print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")

        entry_years = precompute_entry_years(edu_annual)
        results = run_residualized_sweep(
            panel, entry_years, outcome_col, CEILINGS,
            fe_func=fe_twoway_r2, resid_func=fe_residualize_gdp_twoway,
            label=outcome_label, print_every=10
        )
        print_summary(results, CEILINGS, label=outcome_label)
        outcome_results[level_name] = results

    # Cross-level for this outcome
    for ceiling in [60, 90]:
        print(f"\n{'=' * 90}")
        print(f"CROSS-LEVEL (two-way FE): entry=10%, ceiling={ceiling}%  →  {outcome_label}")
        print(f"{'=' * 90}")
        print(f"{'Level':<20} {'Edu':>7} {'GDP':>7} {'Resid':>9} {'Edu→GDP':>8}")
        print("-" * 55)
        for level_name in EDU_LEVELS:
            r = outcome_results[level_name].get(str(ceiling), {}).get("10", {})
            if r and r.get("edu_r2") is not None:
                def fmt(v): return f"{v:.3f}" if v is not None else "n/a"
                print(f"{level_name:<20} {fmt(r.get('edu_r2')):>7} {fmt(r.get('raw_gdp_r2')):>7} "
                      f"{fmt(r.get('resid_gdp_r2')):>9} {fmt(r.get('edu_gdp_r2')):>8}")

    all_results[outcome_label] = outcome_results

# Also add child education (parent→child, not WB data)
print(f"\n\n{'*' * 90}")
print(f"* OUTCOME: CHILD EDUCATION  (two-way FE: country + time)")
print(f"{'*' * 90}")

child_edu_results = {}
for level_name, col_name in EDU_LEVELS.items():
    print(f"\n# {level_name.upper().replace('_', ' ')}  →  Child Edu(T+25)")
    edu_annual = interpolate_to_annual(edu_raw, col_name)
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
    results = run_residualized_sweep(
        panel, entry_years, "child_edu", CEILINGS,
        fe_func=fe_twoway_r2, resid_func=fe_residualize_gdp_twoway,
        label="ChildEdu", print_every=10
    )
    print_summary(results, CEILINGS, label="ChildEdu")
    child_edu_results[level_name] = results

all_results["ChildEdu"] = child_edu_results

# Final cross-outcome summary
print(f"\n\n{'=' * 90}")
print(f"GRAND SUMMARY: Two-way FE, entry=10%, ceiling=90%, lower_secondary")
print(f"{'=' * 90}")
print(f"{'Outcome':<15} {'Edu':>7} {'GDP':>7} {'Resid':>9} {'Edu→GDP':>8}")
print("-" * 50)
for outcome_label in ["LE", "TFR", "U5MR", "ChildEdu"]:
    r = all_results[outcome_label].get("lower_secondary", {}).get("90", {}).get("10", {})
    if r and r.get("edu_r2") is not None:
        def fmt(v): return f"{v:.3f}" if v is not None else "n/a"
        print(f"{outcome_label:<15} {fmt(r.get('edu_r2')):>7} {fmt(r.get('raw_gdp_r2')):>7} "
              f"{fmt(r.get('resid_gdp_r2')):>9} {fmt(r.get('edu_gdp_r2')):>8}")

checkin = {
    "script": "scripts/twoway_fe_all_outcomes.py",
    "produced": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "method": "Two-way FE (country + time). Residualized GDP. Entry-cohort + ceiling. T=1960-1990, lag=25. Four outcomes × three education levels.",
    "results": {k: {lev: v for lev, v in out.items()} for k, out in all_results.items()},
}
os.makedirs(CHECKIN, exist_ok=True)
with open(os.path.join(CHECKIN, "twoway_fe_all_outcomes.json"), "w") as f:
    json.dump(checkin, f, indent=2)
print(f"\nCheckin written.")
