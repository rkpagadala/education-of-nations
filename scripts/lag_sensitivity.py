"""
lag_sensitivity.py
===================
Does the zero GDP result hold at different lags?

Runs the residualized analysis at lags 15, 20, 25, 30 for all four outcomes
(LE, TFR, U5MR, child education). Lower secondary, entry=10%.

If the result holds across lags, it's not an artifact of the 25-year choice.
"""

import os, sys, json
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared import *

T_YEARS_BASE = list(range(1960, 2000, 5))
LAGS = [15, 20, 25, 30]
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

all_results = {}

for lag in LAGS:
    print(f"\n\n{'*' * 90}")
    print(f"* LAG = {lag} years")
    print(f"{'*' * 90}")

    # Adjust T_YEARS so outcomes don't exceed data
    # LE/TFR go to 2022, U5MR to 2015
    max_t_le = min(2022 - lag, 1990)
    max_t_u5 = min(2015 - lag, 1990)
    t_years_le = [t for t in T_YEARS_BASE if t <= max_t_le]
    t_years_u5 = [t for t in T_YEARS_BASE if t <= max_t_u5]

    lag_results = {}

    # WB outcomes
    for outcome_label, outcome_col, outcome_df, t_years in [
        ("LE", "le", le_df, t_years_le),
        ("TFR", "tfr", tfr_df, t_years_le),
        ("U5MR", "u5mr", u5mr_df, t_years_u5),
    ]:
        panel = build_panel(edu_annual, outcome_df, gdp_df, t_years, lag, outcome_col)
        if len(panel) < 10:
            print(f"  {outcome_label}: insufficient data at lag={lag}")
            continue

        print(f"\n  {outcome_label} (n={len(panel)}, countries={panel['country'].nunique()})")
        for ceiling in CEILINGS:
            cohort = entry_years.get(10, {})
            sub = filter_panel(panel, cohort, ceiling)
            if len(sub) < 10:
                continue
            r2_e, n_e, c_e = fe_r2("edu_t", outcome_col, sub)
            r2_g, _, _ = fe_r2("log_gdp_t", outcome_col, sub)
            resid = fe_residualize_gdp(sub)
            r2_resid = np.nan
            if resid is not None:
                sub_r, _ = resid
                r2_resid, _, _ = fe_r2("gdp_resid", outcome_col, sub_r)

            def fmt(v): return f"{v:.3f}" if not np.isnan(v) else "n/a"
            print(f"    ceil={ceiling}%: Edu={fmt(r2_e)}  GDP={fmt(r2_g)}  Resid={fmt(r2_resid)}  n={n_e} ctry={c_e}")

            lag_results[f"{outcome_label}_ceil{ceiling}"] = {
                "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
                "raw_gdp_r2": round(r2_g, 3) if not np.isnan(r2_g) else None,
                "resid_gdp_r2": round(r2_resid, 3) if not np.isnan(r2_resid) else None,
                "n": n_e, "countries": c_e,
            }

    # Child education
    rows = []
    t_years_edu = [t for t in T_YEARS_BASE if t <= 1990]
    for c in sorted(edu_annual.keys()):
        s = edu_annual[c]
        for t in t_years_edu:
            if t not in s.index or (t + lag) not in s.index:
                continue
            parent_edu = s[t]
            child_edu = s[t + lag]
            gdp_t = get_wb_val(gdp_df, c, t)
            if np.isnan(parent_edu) or np.isnan(child_edu):
                continue
            rows.append({
                "country": c, "t": t, "edu_t": parent_edu,
                "log_gdp_t": np.log(gdp_t) if not np.isnan(gdp_t) and gdp_t > 0 else np.nan,
                "child_edu": child_edu,
            })
    panel_ce = pd.DataFrame(rows)
    if len(panel_ce) >= 10:
        print(f"\n  ChildEdu (n={len(panel_ce)}, countries={panel_ce['country'].nunique()})")
        for ceiling in CEILINGS:
            cohort = entry_years.get(10, {})
            sub = filter_panel(panel_ce, cohort, ceiling)
            if len(sub) < 10:
                continue
            r2_e, n_e, c_e = fe_r2("edu_t", "child_edu", sub)
            r2_g, _, _ = fe_r2("log_gdp_t", "child_edu", sub)
            resid = fe_residualize_gdp(sub)
            r2_resid = np.nan
            if resid is not None:
                sub_r, _ = resid
                r2_resid, _, _ = fe_r2("gdp_resid", "child_edu", sub_r)

            def fmt(v): return f"{v:.3f}" if not np.isnan(v) else "n/a"
            print(f"    ceil={ceiling}%: Edu={fmt(r2_e)}  GDP={fmt(r2_g)}  Resid={fmt(r2_resid)}  n={n_e} ctry={c_e}")

            lag_results[f"ChildEdu_ceil{ceiling}"] = {
                "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
                "raw_gdp_r2": round(r2_g, 3) if not np.isnan(r2_g) else None,
                "resid_gdp_r2": round(r2_resid, 3) if not np.isnan(r2_resid) else None,
                "n": n_e, "countries": c_e,
            }

    all_results[str(lag)] = lag_results

# Summary
print(f"\n\n{'=' * 90}")
print(f"SUMMARY: lower secondary, entry=10%, ceiling=90%")
print(f"{'=' * 90}")
print(f"{'Lag':<6} {'LE Edu':>7} {'LE Res':>7} | {'TFR Edu':>8} {'TFR Res':>8} | {'U5 Edu':>7} {'U5 Res':>7} | {'CE Edu':>7} {'CE Res':>7}")
print("-" * 85)
for lag in LAGS:
    lr = all_results[str(lag)]
    def g(key, field):
        v = lr.get(key, {}).get(field)
        return f"{v:.3f}" if v is not None else "n/a"
    print(f"  {lag}yr {g('LE_ceil90','edu_r2'):>7} {g('LE_ceil90','resid_gdp_r2'):>7} | "
          f"{g('TFR_ceil90','edu_r2'):>8} {g('TFR_ceil90','resid_gdp_r2'):>8} | "
          f"{g('U5MR_ceil90','edu_r2'):>7} {g('U5MR_ceil90','resid_gdp_r2'):>7} | "
          f"{g('ChildEdu_ceil90','edu_r2'):>7} {g('ChildEdu_ceil90','resid_gdp_r2'):>7}")

checkin = {
    "script": "scripts/lag_sensitivity.py",
    "produced": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "method": "Country FE, residualized GDP. Lags 15/20/25/30. Lower secondary, entry=10%. Four outcomes.",
    "results": all_results,
}
os.makedirs(CHECKIN, exist_ok=True)
with open(os.path.join(CHECKIN, "lag_sensitivity.json"), "w") as f:
    json.dump(checkin, f, indent=2)
print(f"\nCheckin written.")
