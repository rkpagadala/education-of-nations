"""
backfill_income_lag_test.py

Robustness test for Figure A1: does backfilling pre-1960 income at subsistence
levels change the lag-decay result?

Motivation:
  A reviewer objects that income data "thins out" at long lags, making the
  education-vs-income comparison unfair. The defense: pre-1960 income wasn't
  missing — it was uniformly low (~$400–600 per capita in constant 2017 USD).
  If we backfill it, income's R² should still collapse at long lags because
  subsistence-level income has no cross-country variance to predict anything.

What it does:
  For each lag L from 0 to 100 years (step 5):
    - Outcome: life expectancy at year T (WDI, 1960–2015)
    - Predictor A: lower secondary completion at year T-L  (WCDE v3, 1870–2015)
    - Predictor B: log GDP per capita at year T-L
    - Country fixed effects: demean each variable by country mean
    - Record within-country R² for each predictor separately

  Run TWICE:
    (1) Original GDP data (WDI, starts ~1960; GDP capped at lag 25 per fig_a1)
    (2) Backfilled GDP: pre-1960 values set to $500/capita (constant 2017 USD)
        for all countries, extending GDP back to 1870 — NO lag cap

  Prints both GDP R² curves side by side, plus education R² for reference.

Data sources:
  - Education: wcde/data/processed/cohort_lower_sec_both.csv (WCDE v3)
  - GDP: data/gdppercapita_us_inflation_adjusted.csv (World Bank, constant 2017 USD)
  - Life expectancy: data/life_expectancy_years.csv (World Bank)

Key parameters:
  SUBSISTENCE_GDP = 500   # constant 2017 USD, approximate pre-industrial income
  BACKFILL_START  = 1870  # earliest year to backfill GDP
  BACKFILL_END    = 1959  # last year before WDI GDP data begins
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
WCDE_PROC  = os.path.join(REPO_ROOT, "wcde", "data", "processed")
WB_DIR     = os.path.join(REPO_ROOT, "data")

# ── parameters ────────────────────────────────────────────────────────────────
LAG_MIN     = 0
LAG_MAX     = 100
LAG_STEP    = 5
PANEL_START = 1960   # WDI life expectancy starts here
PANEL_END   = 2015   # last outcome year

SUBSISTENCE_GDP = 500    # constant 2017 USD — approximate pre-industrial income
BACKFILL_START  = 1870   # earliest year to backfill (matches WCDE education data)
BACKFILL_END    = 1959   # last year before WDI GDP data begins

MIN_OBS       = 500   # minimum observations for a valid R² estimate
MIN_OBS_PER_C = 3     # drop countries with < this many obs
GDP_LAG_MAX   = 25    # cap for original (non-backfilled) GDP, per fig_a1

# ── load data ─────────────────────────────────────────────────────────────────
def load_wide(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.str.lower().str.strip()
    df.columns = df.columns.astype(int)
    return df.clip(lower=0)


# WCDE v3 cohort completion (5-year intervals), interpolated to annual
_edu_raw = pd.read_csv(os.path.join(WCDE_PROC, "cohort_lower_sec_both.csv"),
                       index_col="country")
_edu_raw.columns = [int(c) for c in _edu_raw.columns]
_all_yrs = list(range(min(_edu_raw.columns), max(_edu_raw.columns) + 1))
_edu_raw = _edu_raw.reindex(columns=_all_yrs).interpolate(axis=1) \
                    .bfill(axis=1).ffill(axis=1)
_edu_raw.index = _edu_raw.index.str.lower().str.strip()
edu = _edu_raw.clip(lower=0)

gdp_orig = load_wide(os.path.join(WB_DIR, "gdppercapita_us_inflation_adjusted.csv"))
le        = load_wide(os.path.join(WB_DIR, "life_expectancy_years.csv"))

# ── build backfilled GDP ──────────────────────────────────────────────────────
# Start from original GDP, then prepend columns 1870–1959 filled with $500
gdp_back = gdp_orig.copy()
backfill_years = list(range(BACKFILL_START, BACKFILL_END + 1))
for yr in backfill_years:
    if yr not in gdp_back.columns:
        gdp_back[yr] = SUBSISTENCE_GDP
    else:
        # fill NaNs in existing columns too
        gdp_back[yr] = gdp_back[yr].fillna(SUBSISTENCE_GDP)

# also fill NaN cells in 1960+ with subsistence where WDI data is missing
for yr in gdp_back.columns:
    if yr >= 1960:
        gdp_back[yr] = gdp_back[yr].fillna(SUBSISTENCE_GDP)

# sort columns chronologically
gdp_back = gdp_back[sorted(gdp_back.columns)]

# common countries across all three datasets
countries = sorted(set(edu.index) & set(gdp_orig.index) & set(le.index))
print(f"N countries (intersection of edu, GDP, LE): {len(countries)}")
print(f"GDP original columns: {min(gdp_orig.columns)}–{max(gdp_orig.columns)}")
print(f"GDP backfilled columns: {min(gdp_back.columns)}–{max(gdp_back.columns)}")
print(f"Education columns: {min(edu.columns)}–{max(edu.columns)}")
print(f"Subsistence backfill value: ${SUBSISTENCE_GDP}/capita (constant 2017 USD)")
print()


# ── within-country R² ────────────────────────────────────────────────────────
def within_r2(predictor_col, outcome_col, rows):
    """Within-country (FE-demeaned) R² using OLS on demeaned data."""
    if len(rows) < MIN_OBS:
        return np.nan, 0, 0
    p = pd.DataFrame(rows)
    counts = p.groupby("country")[outcome_col].transform("count")
    p = p[counts >= MIN_OBS_PER_C].copy()
    if len(p) < MIN_OBS:
        return np.nan, 0, 0
    n_obs = len(p)
    n_countries = p["country"].nunique()
    for col in [predictor_col, outcome_col]:
        p[col] = p[col] - p.groupby("country")[col].transform("mean")
    reg = LinearRegression(fit_intercept=False).fit(
        p[[predictor_col]], p[outcome_col]
    )
    return reg.score(p[[predictor_col]], p[outcome_col]), n_obs, n_countries


# ── compute R² at each lag ────────────────────────────────────────────────────
lags = list(range(LAG_MIN, LAG_MAX + 1, LAG_STEP))

r2_edu       = []
r2_gdp_orig  = []  # original GDP, capped at lag 25
r2_gdp_back  = []  # backfilled GDP, no lag cap

print(f"Computing within-country R² at {len(lags)} lag values...\n")

for lag in lags:
    edu_rows       = []
    gdp_orig_rows  = []
    gdp_back_rows  = []

    for country in countries:
        for yr in range(PANEL_START, PANEL_END + 1):
            le_val = le.loc[country, yr] if yr in le.columns else np.nan
            if np.isnan(le_val):
                continue
            yr_pred = yr - lag

            # education predictor (WCDE, available 1870–2015)
            if yr_pred in edu.columns:
                edu_val = edu.loc[country, yr_pred]
                if not np.isnan(edu_val):
                    edu_rows.append({"country": country, "edu": edu_val,
                                     "le": le_val})

            # original GDP predictor (WDI, 1960+; capped at lag 25)
            if lag <= GDP_LAG_MAX and yr_pred in gdp_orig.columns:
                gdp_val = gdp_orig.loc[country, yr_pred]
                if not np.isnan(gdp_val) and gdp_val > 0:
                    gdp_orig_rows.append({"country": country,
                                          "log_gdp": np.log(gdp_val),
                                          "le": le_val})

            # backfilled GDP predictor (1870+; no lag cap)
            if yr_pred in gdp_back.columns:
                gdp_val = gdp_back.loc[country, yr_pred]
                if not np.isnan(gdp_val) and gdp_val > 0:
                    gdp_back_rows.append({"country": country,
                                          "log_gdp": np.log(gdp_val),
                                          "le": le_val})

    re, ne, nce  = within_r2("edu",     "le", edu_rows)
    ro, no, nco  = within_r2("log_gdp", "le", gdp_orig_rows)
    rb, nb, ncb  = within_r2("log_gdp", "le", gdp_back_rows)

    r2_edu.append(re)
    r2_gdp_orig.append(ro)
    r2_gdp_back.append(rb)

    print(f"  lag={lag:3d}  edu R²={re:>6.3f} (n={ne:5d}, {nce:3d} ctry)  "
          f"gdp_orig R²={'   nan' if np.isnan(ro) else f'{ro:>5.3f}'} "
          f"(n={no:5d}, {nco:3d} ctry)  "
          f"gdp_back R²={rb:>5.3f} (n={nb:5d}, {ncb:3d} ctry)")


# ── summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY: Backfill test — does filling pre-1960 income at $500/capita change result?")
print("=" * 80)
print(f"\n{'Lag':>5}  {'Edu R²':>8}  {'GDP orig':>10}  {'GDP backfill':>12}  {'Delta':>8}")
print("-" * 55)

for lag, re, ro, rb in zip(lags, r2_edu, r2_gdp_orig, r2_gdp_back):
    marker = {25: " <- 1 gen", 50: " <- 2 gen", 75: " <- 3 gen"}.get(lag, "")
    re_s = f"{re:>8.3f}" if not np.isnan(re) else "     nan"
    ro_s = f"{ro:>10.3f}" if not np.isnan(ro) else "       nan"
    rb_s = f"{rb:>12.3f}" if not np.isnan(rb) else "         nan"
    if not np.isnan(ro) and not np.isnan(rb):
        delta = rb - ro
        d_s = f"{delta:>+8.3f}"
    elif np.isnan(ro) and not np.isnan(rb):
        d_s = "  (new)"
    else:
        d_s = "       "
    print(f"{lag:>5}  {re_s}  {ro_s}  {rb_s}  {d_s}{marker}")

print("\n" + "-" * 55)
print("Interpretation:")
print("  - 'GDP orig' = original WDI data, capped at lag 25 (per Figure A1)")
print("  - 'GDP backfill' = pre-1960 income set to $500/capita, NO lag cap")
print("  - If backfilling doesn't rescue income's R², the reviewer's objection fails:")
print("    the data wasn't missing, it was uniformly low and uninformative.")
print("  - Education R² shown for comparison: it persists across all lags.")

# ── key comparison: education vs backfilled GDP at generational lags ──────────
print("\n" + "=" * 80)
print("KEY COMPARISON: Education vs Backfilled GDP at generational horizons")
print("=" * 80)
for gen, lag_val in [("1 generation", 25), ("2 generations", 50), ("3 generations", 75)]:
    idx = lags.index(lag_val)
    print(f"  {gen} (lag {lag_val}):  Edu R² = {r2_edu[idx]:.3f}  |  "
          f"GDP backfill R² = {r2_gdp_back[idx]:.3f}  |  "
          f"ratio = {r2_edu[idx] / r2_gdp_back[idx]:.1f}x"
          if r2_gdp_back[idx] > 0.001 else
          f"  {gen} (lag {lag_val}):  Edu R² = {r2_edu[idx]:.3f}  |  "
          f"GDP backfill R² = {r2_gdp_back[idx]:.3f}  |  "
          f"ratio = inf")

print("\nDone.")
