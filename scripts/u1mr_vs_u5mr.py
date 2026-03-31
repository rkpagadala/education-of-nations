"""
u1mr_vs_u5mr.py
================
Compare education's predictive power for infant mortality (U1MR) vs
under-5 mortality (U5MR).

Hypothesis: education should explain MORE of infant mortality than U5MR
because infant death is dominated by household behavior (breastfeeding,
hygiene, birth spacing) that requires an educated mother. Deaths at
ages 1-4 are more amenable to mass vaccination campaigns that bypass
the household. The MDG-era residualized GDP signal should be WEAKER
for U1MR than U5MR.

Entry-cohort design (entry >= 10%, ceiling <= 90%), country FE,
lower secondary completion, T=1960-1990, lag=25.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared import (
    load_education, load_wb, interpolate_to_annual, precompute_entry_years,
    build_panel, filter_panel, fe_residualize_gdp,
    CHECKIN
)

T_YEARS = list(range(1960, 1995, 5))
LAG = 25
COL_NAME = "lower_sec"
CEILING = 90
ENTRY_THRESHOLD = 10


def clustered_fe(x_col, y_col, data):
    """Country FE regression with clustered SEs."""
    sub = data.dropna(subset=[x_col, y_col]).copy()
    counts = sub.groupby("country").size()
    sub = sub[sub["country"].isin(counts[counts >= 2].index)]
    n_countries = sub["country"].nunique()
    if n_countries < 3 or len(sub) < 10:
        return None

    xdm = (sub[x_col] - sub.groupby("country")[x_col].transform("mean")).values
    ydm = (sub[y_col] - sub.groupby("country")[y_col].transform("mean")).values
    countries = sub["country"].values

    ok = ~np.isnan(xdm) & ~np.isnan(ydm)
    xdm, ydm, countries = xdm[ok], ydm[ok], countries[ok]
    n = len(xdm)
    if n < 10:
        return None

    beta = np.sum(xdm * ydm) / np.sum(xdm ** 2)
    resid = ydm - beta * xdm
    r2 = 1 - np.sum(resid ** 2) / np.sum(ydm ** 2)

    unique_c = np.unique(countries)
    G = len(unique_c)
    meat = 0.0
    for c in unique_c:
        idx = countries == c
        score_c = np.sum(xdm[idx] * resid[idx])
        meat += score_c ** 2

    bread = 1.0 / np.sum(xdm ** 2)
    correction = (G / (G - 1))
    var_beta = bread ** 2 * meat * correction
    se = np.sqrt(var_beta)
    t_stat = beta / se
    pval = 2 * stats.t.sf(np.abs(t_stat), df=G - 1)

    return {"beta": beta, "se": se, "pval": pval, "r2": r2, "n": n, "countries": G}


# ── Load data ─────────────────────────────────────────────────────

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
u5mr_df = load_wb("child_mortality_u5.csv")
u1mr_df = load_wb("infant_mortality_u1.csv")

edu_annual = interpolate_to_annual(edu_raw, COL_NAME)
entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[ENTRY_THRESHOLD]

# ── Build panels ──────────────────────────────────────────────────

panel_u5 = build_panel(edu_annual, u5mr_df, gdp_df, T_YEARS, LAG, "mortality")
panel_u1 = build_panel(edu_annual, u1mr_df, gdp_df, T_YEARS, LAG, "mortality")

sub_u5 = filter_panel(panel_u5, cohort, CEILING)
sub_u1 = filter_panel(panel_u1, cohort, CEILING)

print(f"U5MR panel: {len(sub_u5)} obs, {sub_u5['country'].nunique()} countries")
print(f"U1MR panel: {len(sub_u1)} obs, {sub_u1['country'].nunique()} countries")

# ── Full sample comparison ────────────────────────────────────────

print("\n" + "=" * 90)
print("FULL SAMPLE: U5MR vs U1MR (INFANT MORTALITY)")
print("=" * 90)
print(f"Entry >= {ENTRY_THRESHOLD}%, ceiling <= {CEILING}%, T=1960-1990, lag={LAG}\n")

for label, sub in [("U5MR (under-5)", sub_u5), ("U1MR (under-1)", sub_u1)]:
    res_e = clustered_fe("edu_t", "mortality", sub)
    res_g = clustered_fe("log_gdp_t", "mortality", sub)

    resid = fe_residualize_gdp(sub)
    res_r = None
    if resid is not None:
        sub_r, edu_gdp_r2 = resid
        res_r = clustered_fe("gdp_resid", "mortality", sub_r)

    print(f"  {label}:")
    if res_e:
        print(f"    Education R²     = {res_e['r2']:.3f}  (p={res_e['pval']:.4f}, n={res_e['n']})")
    if res_g:
        print(f"    Raw GDP R²       = {res_g['r2']:.3f}  (p={res_g['pval']:.4f})")
    if res_r:
        print(f"    Resid GDP R²     = {res_r['r2']:.3f}  (p={res_r['pval']:.4f})")
    if resid:
        print(f"    Edu → GDP R²     = {edu_gdp_r2:.3f}")
    print()

# ── By outcome year sweep ─────────────────────────────────────────

print("=" * 90)
print("SWEEP BY OUTCOME YEAR: U5MR vs U1MR")
print("=" * 90)
print(f"\n{'Year':>6}  {'--- U5MR ---':^28}  {'--- U1MR ---':^28}")
print(f"{'':>6}  {'Edu R²':>7} {'Res R²':>7} {'p':>6} {'n':>5}  {'Edu R²':>7} {'Res R²':>7} {'p':>6} {'n':>5}")
print("-" * 75)

for year in range(1985, 2021):
    print(f"  {year}  ", end="")

    for sub_full in [sub_u5, sub_u1]:
        mask = (sub_full["t"] + LAG) <= year
        sub = sub_full[mask].copy()

        if len(sub) < 10 or sub["country"].nunique() < 3:
            print(f"{'---':>7} {'---':>7} {'---':>6} {'---':>5}  ", end="")
            continue

        res_e = clustered_fe("edu_t", "mortality", sub)
        resid = fe_residualize_gdp(sub)
        res_r = None
        if resid is not None:
            sub_r, _ = resid
            res_r = clustered_fe("gdp_resid", "mortality", sub_r)

        r2_e = res_e["r2"] if res_e else np.nan
        r2_r = res_r["r2"] if res_r else np.nan
        p_r = res_r["pval"] if res_r else np.nan
        n = res_e["n"] if res_e else 0

        def fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "---"
        def fmtp(v):
            return f"{v:.2f}" if not np.isnan(v) else "---"

        print(f"{fmt(r2_e):>7} {fmt(r2_r):>7} {fmtp(p_r):>6} {n:>5}  ", end="")

    print()

# ── Before/after 2000 split ───────────────────────────────────────

print("\n" + "=" * 90)
print("BEFORE vs AFTER 2000: U5MR vs U1MR")
print("=" * 90)

for label, sub_full in [("U5MR", sub_u5), ("U1MR", sub_u1)]:
    print(f"\n  {label}:")
    for period, mask in [
        ("Before 2000", (sub_full["t"] + LAG) < 2000),
        ("After 2000",  (sub_full["t"] + LAG) >= 2000),
        ("All years",   pd.Series(True, index=sub_full.index)),
    ]:
        sub = sub_full[mask].copy()
        if len(sub) < 10 or sub["country"].nunique() < 3:
            continue

        res_e = clustered_fe("edu_t", "mortality", sub)
        resid = fe_residualize_gdp(sub)
        res_r = None
        if resid is not None:
            sub_r, _ = resid
            res_r = clustered_fe("gdp_resid", "mortality", sub_r)

        r2_e = res_e["r2"] if res_e else np.nan
        r2_r = res_r["r2"] if res_r else np.nan
        p_r = res_r["pval"] if res_r else np.nan
        n = res_e["n"] if res_e else 0

        print(f"    {period:<15}  Edu R²={r2_e:.3f}  Resid R²={r2_r:.3f}  p={p_r:.4f}  n={n}")

# ── Checkin ────────────────────────────────────────────────────────

checkin = {
    "script": "scripts/u1mr_vs_u5mr.py",
    "produced": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "method": (
        "Infant mortality (U1MR) vs under-5 mortality (U5MR) comparison. "
        f"Entry >= {ENTRY_THRESHOLD}%, ceiling <= {CEILING}%, T=1960-1990, lag={LAG}. "
        "Clustered SEs by country."
    ),
}

os.makedirs(CHECKIN, exist_ok=True)
out_path = os.path.join(CHECKIN, "u1mr_vs_u5mr.json")
with open(out_path, "w") as f:
    json.dump(checkin, f, indent=2)
print(f"\nCheckin written to {out_path}")
