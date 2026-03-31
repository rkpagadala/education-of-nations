# =============================================================================
# PAPER REFERENCE
# Script:  scripts/education_vs_gdp_by_cutoff.py
# Paper:   "Education of Nations"
#
# Produces:
#   Side-by-side comparison of education R² vs GDP R² at each parental
#   education cutoff (10% to 90% and no cutoff).
#   Key finding: education leads GDP by 2–3.4x across all cutoffs;
#               education R² improves with lower cutoffs (S-curve),
#               GDP R² stays flat (~0.21–0.29) — no S-curve structure
#
# Inputs:
#   wcde/data/processed/cohort_completion_both_long.csv
#   data/gdppercapita_us_inflation_adjusted.csv
#
# Key parameters:
#   GENERATIONAL_LAG = 25
#   CUTOFFS = 10, 20, ..., 90
# =============================================================================
"""
education_vs_gdp_by_cutoff.py

Compares education vs GDP as predictors of child education (within-country FE)
at each parental education cutoff. Shows that the paper's headline comparison
(R² 0.455 vs 0.256 = 1.7x) is contaminated by ceiling observations. The
proper comparison — excluding countries already at ceiling — is 3.4x at the
30% cutoff where most of the remaining 22% sits.

Section 6.1 of the paper.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "../wcde/data/processed")
DATA = os.path.join(SCRIPT_DIR, "../data")

# ── Load education data ─────────────────────────────────────────────────────
long = pd.read_csv(os.path.join(PROC, "cohort_completion_both_long.csv"))
low_w = long.pivot(index="country", columns="cohort_year", values="lower_sec")

# ── Load GDP data ────────────────────────────────────────────────────────────
gdp = pd.read_csv(os.path.join(DATA, "gdppercapita_us_inflation_adjusted.csv"))
gdp_long = gdp.melt(id_vars="Country", var_name="year", value_name="gdp")
gdp_long["year"] = pd.to_numeric(gdp_long["year"], errors="coerce")
gdp_long = gdp_long.dropna(subset=["year", "gdp"])
gdp_long["year"] = gdp_long["year"].astype(int)
gdp_long["country_lc"] = gdp_long["Country"].str.lower().str.strip()

# ── Country name mapping: WCDE → GDP ────────────────────────────────────────
MANUAL_MAP = {
    "republic of korea": "south korea",
    "united states of america": "united states",
    "united kingdom of great britain and northern ireland": "united kingdom",
    "taiwan province of china": "taiwan",
    "hong kong special administrative region of china": "hong kong, china",
    "democratic people's republic of korea": "north korea",
    "iran (islamic republic of)": "iran",
    "bolivia (plurinational state of)": "bolivia",
    "venezuela (bolivarian republic of)": "venezuela",
    "lao people's democratic republic": "lao",
    "viet nam": "vietnam",
    "russian federation": "russia",
    "syrian arab republic": "syria",
    "republic of moldova": "moldova",
    "united republic of tanzania": "tanzania",
    "democratic republic of the congo": "congo, dem. rep.",
}

gdp_lc_set = set(gdp_long["country_lc"].unique())
wcde_to_lc = {}
for wc in low_w.index:
    wl = wc.lower()
    if wl in gdp_lc_set:
        wcde_to_lc[wc] = wl
        continue
    if wl in MANUAL_MAP:
        wcde_to_lc[wc] = MANUAL_MAP[wl]
        continue
    for gc in gdp_lc_set:
        if wl.split()[0] == gc.split()[0] and len(wl.split()[0]) > 4:
            wcde_to_lc[wc] = gc
            break


def v(df_w, country, year):
    """Look up a value from wide-format DataFrame."""
    try:
        val = float(df_w.loc[country, int(year)])
        return val if not np.isnan(val) else np.nan
    except (KeyError, ValueError):
        return np.nan


def get_gdp(country_lc, year):
    """Look up GDP for a country-year."""
    rows = gdp_long[(gdp_long["country_lc"] == country_lc) & (gdp_long["year"] == year)]
    if len(rows) == 0:
        return np.nan
    val = rows["gdp"].iloc[0]
    return val if pd.notna(val) and val > 0 else np.nan


# ── Build post-1975 panel with GDP ──────────────────────────────────────────
CHILD_COHORTS = list(range(1975, 2016, 5))
panel_rows = []
for c in low_w.index:
    gdp_lc = wcde_to_lc.get(c)
    for child_yr in CHILD_COHORTS:
        parent_yr = child_yr - 25
        child_low = v(low_w, c, child_yr)
        parent_low = v(low_w, c, parent_yr)
        if np.isnan(child_low) or np.isnan(parent_low):
            continue
        row = {
            "country": c,
            "cohort_year": child_yr,
            "child_low": child_low,
            "parent_low": parent_low,
            "log_gdp": np.nan,
        }
        if gdp_lc:
            gval = get_gdp(gdp_lc, child_yr)
            if not np.isnan(gval):
                row["log_gdp"] = np.log(gval)
        panel_rows.append(row)

panel = pd.DataFrame(panel_rows)
panel_gdp = panel.dropna(subset=["log_gdp"]).copy()

print(f"Post-1975 panel with GDP: {len(panel_gdp)} obs, "
      f"{panel_gdp['country'].nunique()} countries")


def run_fe(df, x_col, y_col="child_low"):
    """Run country fixed-effects regression via demeaning."""
    sub = df.copy()
    counts = sub.groupby("country").size()
    valid = counts[counts > 1].index
    sub = sub[sub["country"].isin(valid)]
    if len(sub) < 10:
        return np.nan, np.nan, 0, 0
    sub["y_dm"] = sub[y_col] - sub.groupby("country")[y_col].transform("mean")
    sub["x_dm"] = sub[x_col] - sub.groupby("country")[x_col].transform("mean")
    reg = LinearRegression(fit_intercept=False).fit(
        sub[["x_dm"]].values, sub["y_dm"].values
    )
    r2 = reg.score(sub[["x_dm"]].values, sub["y_dm"].values)
    return reg.coef_[0], r2, len(sub), sub["country"].nunique()


# ── Print comparison table ───────────────────────────────────────────────────
CUTOFFS = list(range(10, 100, 10))

print(f"\n{'Cutoff':>10} {'Edu B':>8} {'Edu R2':>8} "
      f"{'GDP B':>10} {'GDP R2':>8} {'Ratio':>7} {'n':>6} {'Ctry':>5}")
print("-" * 70)

for cutoff in CUTOFFS:
    sub = panel_gdp[panel_gdp["parent_low"] < cutoff]
    eb, er2, n, nc = run_fe(sub, "parent_low")
    gb, gr2, _, _ = run_fe(sub, "log_gdp")
    if n == 0 or np.isnan(er2) or np.isnan(gr2) or gr2 == 0:
        continue
    ratio = er2 / gr2
    print(f"    <{cutoff:3d}%  {eb:8.3f} {er2:8.3f} "
          f"{gb:10.3f} {gr2:8.3f} {ratio:7.1f}x {n:6d} {nc:5d}")

eb, er2, n, nc = run_fe(panel_gdp, "parent_low")
gb, gr2, _, _ = run_fe(panel_gdp, "log_gdp")
ratio = er2 / gr2
print(f"  no cut  {eb:8.3f} {er2:8.3f} "
      f"{gb:10.3f} {gr2:8.3f} {ratio:7.1f}x {n:6d} {nc:5d}")

print("\nDone.")
