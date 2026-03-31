"""
u5mr_residual_by_year.py
========================
Verify paper claim: residualized GDP → U5MR p-value changes over time.

Lutz argues that at low education levels, recent health interventions
(vaccines, oral rehydration, bed nets) since the MDG era genuinely
reduce child mortality independent of domestic education.

Test: split the residualization analysis by outcome year cutoff.
For each cutoff year Y, restrict to observations where T+lag <= Y,
then report residualized GDP R² and p-value. If Lutz is right,
the p-value should be high (no GDP signal) for early cutoffs and
drop as post-2000 MDG-era outcomes enter the sample.

Uses clustered standard errors (matching regression_tables.py).
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# ── Parameters (match Table 2b) ────────────────────────────────────

T_YEARS = list(range(1960, 1995, 5))
LAG = 25
COL_NAME = "lower_sec"
CEILING = 90
ENTRY_THRESHOLD = 10


def clustered_fe(x_col, y_col, data):
    """
    Country FE regression with clustered standard errors.
    Matches regression_tables.py methodology exactly.
    Returns dict with beta, se, pval, r2, n, countries — or None.
    """
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
    correction = (G / (G - 1)) * ((n - 1) / (n - 1))  # K=1 for demeaned
    var_beta = bread ** 2 * meat * correction
    se = np.sqrt(var_beta)
    t_stat = beta / se
    pval = 2 * stats.t.sf(np.abs(t_stat), df=G - 1)

    return {
        "beta": beta, "se": se, "pval": pval, "r2": r2,
        "n": n, "countries": G,
    }


# ── Load data ───────────────────────────────────────────────────────

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

edu_annual = interpolate_to_annual(edu_raw, COL_NAME)
entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[ENTRY_THRESHOLD]

# ── Build full panel ────────────────────────────────────────────────

panel = build_panel(edu_annual, u5mr_df, gdp_df, T_YEARS, LAG, "u5mr")
sub_full = filter_panel(panel, cohort, CEILING)

print(f"Full panel: {len(sub_full)} obs, {sub_full['country'].nunique()} countries")
print(f"Outcome years range: {(sub_full['t'] + LAG).min()} to {(sub_full['t'] + LAG).max()}")

# ── Verify against regression_tables.py (full sample) ──────────────

print("\n── Verification: full sample should match Table 2b ──")
resid_full = fe_residualize_gdp(sub_full)
if resid_full is not None:
    sub_r_full, _ = resid_full
    res = clustered_fe("gdp_resid", "u5mr", sub_r_full)
    if res:
        print(f"  Resid R²={res['r2']:.3f}  p={res['pval']:.4f}  "
              f"n={res['n']}  countries={res['countries']}")
        print(f"  Paper Table 2b: R²=0.023, p=0.11")

# ── Sweep by outcome year cutoff ────────────────────────────────────

print("\n" + "=" * 90)
print("RESIDUALIZED GDP → U5MR: SWEEP BY OUTCOME YEAR CUTOFF (CLUSTERED SEs)")
print("=" * 90)
print(f"Entry >= {ENTRY_THRESHOLD}%, ceiling <= {CEILING}%, "
      f"T=1960-1990, lag={LAG}")
print()
print(f"{'Outcome ≤':<12} {'Edu R²':>7} {'Edu p':>8} {'Raw GDP R²':>11} "
      f"{'Resid R²':>9} {'Resid p':>9} {'n':>5} {'Ctry':>5}")
print("-" * 75)

results = []

for cutoff in range(1990, 2021, 5):
    mask = (sub_full["t"] + LAG) <= cutoff
    sub = sub_full[mask].copy()

    if len(sub) < 10 or sub["country"].nunique() < 3:
        continue

    # Education
    res_e = clustered_fe("edu_t", "u5mr", sub)

    # Raw GDP
    res_g = clustered_fe("log_gdp_t", "u5mr", sub)

    # Residualized GDP
    resid = fe_residualize_gdp(sub)
    res_r = None
    if resid is not None:
        sub_r, edu_gdp_r2 = resid
        res_r = clustered_fe("gdp_resid", "u5mr", sub_r)

    def fmt(v):
        return f"{v:.3f}" if v is not None and not np.isnan(v) else "n/a"

    def fmtp(v):
        return f"{v:.4f}" if v is not None and not np.isnan(v) else "n/a"

    r2_e = res_e["r2"] if res_e else np.nan
    p_e = res_e["pval"] if res_e else np.nan
    r2_g = res_g["r2"] if res_g else np.nan
    r2_r = res_r["r2"] if res_r else np.nan
    p_r = res_r["pval"] if res_r else np.nan
    n = res_e["n"] if res_e else 0
    ctry = res_e["countries"] if res_e else 0

    print(f"  ≤ {cutoff:<7} {fmt(r2_e):>7} {fmtp(p_e):>8} {fmt(r2_g):>11} "
          f"{fmt(r2_r):>9} {fmtp(p_r):>9} {n:>5} {ctry:>5}")

    results.append({
        "outcome_year_cutoff": cutoff,
        "edu_r2": round(r2_e, 4) if not np.isnan(r2_e) else None,
        "edu_pval": round(p_e, 4) if not np.isnan(p_e) else None,
        "raw_gdp_r2": round(r2_g, 4) if not np.isnan(r2_g) else None,
        "resid_gdp_r2": round(r2_r, 4) if not np.isnan(r2_r) else None,
        "resid_pvalue": round(p_r, 4) if not np.isnan(p_r) else None,
        "n_obs": n,
        "n_countries": ctry,
    })

# ── Before / after 2000 split ──────────────────────────────────────

print("\n" + "=" * 90)
print("BEFORE vs AFTER 2000 (MDG ERA) — CLUSTERED SEs")
print("=" * 90)

split_results = {}

for label, mask in [
    ("Before 2000", (sub_full["t"] + LAG) < 2000),
    ("After 2000",  (sub_full["t"] + LAG) >= 2000),
    ("All years",   pd.Series(True, index=sub_full.index)),
]:
    sub = sub_full[mask].copy()
    if len(sub) < 10 or sub["country"].nunique() < 3:
        print(f"  {label}: insufficient data")
        continue

    res_e = clustered_fe("edu_t", "u5mr", sub)

    resid = fe_residualize_gdp(sub)
    res_r = None
    if resid is not None:
        sub_r, _ = resid
        res_r = clustered_fe("gdp_resid", "u5mr", sub_r)

    r2_e = res_e["r2"] if res_e else np.nan
    r2_r = res_r["r2"] if res_r else np.nan
    p_r = res_r["pval"] if res_r else np.nan
    n = res_e["n"] if res_e else 0
    ctry = res_e["countries"] if res_e else 0

    def fmt(v):
        return f"{v:.3f}" if v is not None and not np.isnan(v) else "n/a"

    def fmtp(v):
        return f"{v:.4f}" if v is not None and not np.isnan(v) else "n/a"

    print(f"  {label:<15}  Edu R²={fmt(r2_e)}  Resid R²={fmt(r2_r)}  "
          f"p={fmtp(p_r)}  n={n}  countries={ctry}")

    split_results[label] = {
        "edu_r2": round(r2_e, 4) if not np.isnan(r2_e) else None,
        "resid_gdp_r2": round(r2_r, 4) if not np.isnan(r2_r) else None,
        "resid_pvalue": round(p_r, 4) if not np.isnan(p_r) else None,
        "n_obs": n,
        "n_countries": ctry,
    }

# ── Paper number verification ───────────────────────────────────────

print("\n" + "=" * 90)
print("PAPER VERIFICATION")
print("=" * 90)

before = split_results.get("Before 2000", {})
after = split_results.get("After 2000", {})

print(f"\n  Paper claims (Section 6.2.1):")
print(f"    'before 2000, residualized GDP explains 0.3% of child mortality'")
print(f"    'after 2000, it rises to 2.3%'")
print(f"\n  Script produces:")
print(f"    Before 2000: Resid R² = {before.get('resid_gdp_r2', 'n/a')}")
print(f"    After 2000:  Resid R² = {after.get('resid_gdp_r2', 'n/a')}")

b_r2 = before.get("resid_gdp_r2")
a_r2 = after.get("resid_gdp_r2")
if b_r2 is not None and a_r2 is not None:
    b_pct = f"{b_r2 * 100:.1f}%"
    a_pct = f"{a_r2 * 100:.1f}%"
    b_match = "MATCH" if abs(b_r2 - 0.003) < 0.002 else "MISMATCH"
    a_match = "MATCH" if abs(a_r2 - 0.023) < 0.005 else "MISMATCH"
    print(f"\n  Before 2000: {b_pct} vs paper 0.3%  → {b_match}")
    print(f"  After 2000:  {a_pct} vs paper 2.3%  → {a_match}")

# ── Checkin ─────────────────────────────────────────────────────────

checkin = {
    "script": "scripts/u5mr_residual_by_year.py",
    "produced": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "method": (
        "Residualized GDP → U5MR swept by outcome year cutoff. "
        "Clustered SEs by country (matching regression_tables.py). "
        "Tests Lutz hypothesis: MDG-era health interventions create "
        "GDP signal for child mortality independent of education. "
        f"Entry >= {ENTRY_THRESHOLD}%, ceiling <= {CEILING}%, "
        f"T=1960-1990, lag={LAG}."
    ),
    "sweep": results,
    "before_after_2000": split_results,
}

os.makedirs(CHECKIN, exist_ok=True)
out_path = os.path.join(CHECKIN, "u5mr_residual_by_year.json")
with open(out_path, "w") as f:
    json.dump(checkin, f, indent=2)
print(f"\nCheckin written to {out_path}")
