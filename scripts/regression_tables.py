"""
regression_tables.py
=====================
Formal regression output with clustered standard errors.

For each outcome (LE, TFR, U5MR, child education), reports:
  - β coefficient for education
  - β coefficient for GDP
  - β coefficient for residualized GDP
  - Standard errors clustered by country
  - p-values
  - Within-R²

Uses statsmodels PanelOLS (if available) or manual clustering.
Lower secondary, entry=10%, ceilings 60/90.
"""

import os, sys, json
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared import *

try:
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("WARNING: linearmodels not installed. Using manual clustered SEs.")
    print("  pip install linearmodels")

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


def manual_clustered_se(x_col, y_col, data):
    """
    Country FE regression with manually computed clustered SEs.
    Returns (beta, se, pval, r2, n, n_countries).
    """
    from scipy import stats

    sub = data.dropna(subset=[x_col, y_col]).copy()
    counts = sub.groupby("country").size()
    sub = sub[sub["country"].isin(counts[counts >= 2].index)]
    n_countries = sub["country"].nunique()
    if n_countries < 3 or len(sub) < 10:
        return None

    # Demean
    xdm = (sub[x_col] - sub.groupby("country")[x_col].transform("mean")).values
    ydm = (sub[y_col] - sub.groupby("country")[y_col].transform("mean")).values

    ok = ~np.isnan(xdm) & ~np.isnan(ydm)
    xdm, ydm = xdm[ok], ydm[ok]
    countries = sub["country"].values[ok]
    n = len(xdm)

    # OLS (no intercept, already demeaned)
    beta = np.sum(xdm * ydm) / np.sum(xdm ** 2)
    resid = ydm - beta * xdm
    r2 = 1 - np.sum(resid ** 2) / np.sum(ydm ** 2)

    # Clustered SE (Cameron, Gelbach, Miller 2008)
    unique_c = np.unique(countries)
    G = len(unique_c)
    meat = 0.0
    for c in unique_c:
        idx = countries == c
        score_c = np.sum(xdm[idx] * resid[idx])
        meat += score_c ** 2

    bread = 1.0 / np.sum(xdm ** 2)
    # Small-sample correction: G/(G-1) * (N-1)/(N-K)
    correction = (G / (G - 1)) * ((n - 1) / (n - 1))  # K=1 for demeaned
    var_beta = bread ** 2 * meat * correction
    se = np.sqrt(var_beta)
    t_stat = beta / se
    pval = 2 * stats.t.sf(np.abs(t_stat), df=G - 1)

    return {"beta": beta, "se": se, "pval": pval, "r2": r2, "n": n, "countries": G}


def panel_ols_result(x_col, y_col, data):
    """Use linearmodels PanelOLS if available."""
    if not HAS_LINEARMODELS:
        return manual_clustered_se(x_col, y_col, data)

    sub = data.dropna(subset=[x_col, y_col]).copy()
    counts = sub.groupby("country").size()
    sub = sub[sub["country"].isin(counts[counts >= 2].index)]
    if sub["country"].nunique() < 3 or len(sub) < 10:
        return None

    sub = sub.set_index(["country", "t"])
    try:
        mod = PanelOLS(sub[y_col], sub[[x_col]], entity_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        return {
            "beta": float(res.params[x_col]),
            "se": float(res.std_errors[x_col]),
            "pval": float(res.pvalues[x_col]),
            "r2": float(res.rsquared_within),
            "n": int(res.nobs),
            "countries": int(sub.index.get_level_values(0).nunique()),
        }
    except Exception as e:
        print(f"    PanelOLS failed: {e}, falling back to manual")
        sub = sub.reset_index()
        return manual_clustered_se(x_col, y_col, sub)


# ── Build panels ─────────────────────────────────────────────────────

all_results = {}

# WB outcomes
for outcome_label, outcome_col, outcome_df in [
    ("LE", "le", le_df),
    ("TFR", "tfr", tfr_df),
    ("U5MR", "u5mr", u5mr_df),
]:
    panel = build_panel(edu_annual, outcome_df, gdp_df, T_YEARS, LAG, outcome_col)
    outcome_results = {}

    for ceiling in CEILINGS:
        print(f"\n{'=' * 80}")
        print(f"{outcome_label}, ceiling={ceiling}%, entry=10%")
        print(f"{'=' * 80}")

        cohort = entry_years.get(10, {})
        sub = filter_panel(panel, cohort, ceiling)

        print(f"{'Predictor':<25} {'β':>10} {'SE':>10} {'p':>10} {'R²':>8} {'n':>6} {'Ctry':>5}")
        print("-" * 80)

        ceil_results = {}
        for pred_label, x_col, use_data in [
            ("Education", "edu_t", sub),
            ("GDP (raw)", "log_gdp_t", sub),
        ]:
            res = panel_ols_result(x_col, outcome_col, use_data)
            if res:
                print(f"{pred_label:<25} {res['beta']:>10.4f} {res['se']:>10.4f} "
                      f"{res['pval']:>10.4f} {res['r2']:>8.3f} {res['n']:>6} {res['countries']:>5}")
                ceil_results[pred_label] = res

        # Residualized GDP
        resid = fe_residualize_gdp(sub)
        if resid is not None:
            sub_r, edu_gdp_r2 = resid
            res = panel_ols_result("gdp_resid", outcome_col, sub_r)
            if res:
                print(f"{'GDP (residualized)':<25} {res['beta']:>10.4f} {res['se']:>10.4f} "
                      f"{res['pval']:>10.4f} {res['r2']:>8.3f} {res['n']:>6} {res['countries']:>5}")
                ceil_results["GDP (residualized)"] = res

        outcome_results[str(ceiling)] = ceil_results

    all_results[outcome_label] = outcome_results

# Child education
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
panel_ce = pd.DataFrame(rows)

ce_results = {}
for ceiling in CEILINGS:
    print(f"\n{'=' * 80}")
    print(f"Child Education, ceiling={ceiling}%, entry=10%")
    print(f"{'=' * 80}")

    cohort = entry_years.get(10, {})
    sub = filter_panel(panel_ce, cohort, ceiling)

    print(f"{'Predictor':<25} {'β':>10} {'SE':>10} {'p':>10} {'R²':>8} {'n':>6} {'Ctry':>5}")
    print("-" * 80)

    ceil_res = {}
    for pred_label, x_col, use_data in [
        ("Parent Education", "edu_t", sub),
        ("GDP (raw)", "log_gdp_t", sub),
    ]:
        res = panel_ols_result(x_col, "child_edu", use_data)
        if res:
            print(f"{pred_label:<25} {res['beta']:>10.4f} {res['se']:>10.4f} "
                  f"{res['pval']:>10.4f} {res['r2']:>8.3f} {res['n']:>6} {res['countries']:>5}")
            ceil_res[pred_label] = res

    resid = fe_residualize_gdp(sub)
    if resid is not None:
        sub_r, _ = resid
        res = panel_ols_result("gdp_resid", "child_edu", sub_r)
        if res:
            print(f"{'GDP (residualized)':<25} {res['beta']:>10.4f} {res['se']:>10.4f} "
                  f"{res['pval']:>10.4f} {res['r2']:>8.3f} {res['n']:>6} {res['countries']:>5}")
            ceil_res["GDP (residualized)"] = res

    ce_results[str(ceiling)] = ceil_res

all_results["ChildEdu"] = ce_results

checkin = {
    "script": "scripts/regression_tables.py",
    "produced": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "method": "Country FE with clustered SEs. β, SE, p-value for education, raw GDP, residualized GDP. Lower secondary, entry=10%.",
    "results": {k: {c: {p: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in pred.items()} for p, pred in ceil.items()} for c, ceil in out.items()} for k, out in all_results.items()},
}
os.makedirs(CHECKIN, exist_ok=True)
with open(os.path.join(CHECKIN, "regression_tables.json"), "w") as f:
    json.dump(checkin, f, indent=2)
print(f"\nCheckin written.")
