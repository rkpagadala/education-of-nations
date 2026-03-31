# =============================================================================
# PAPER REFERENCE
# Script:  scripts/asian_financial_crisis.py
# Paper:   "Education of Nations"
#
# Produces:
#   Income-removal test: GDP collapsed 1997-98 across five AFC-affected
#   countries; education trajectories continued undisturbed.
#   Key finding: Indonesia lost 14.5% of GDP; education gained 5.4pp.
#               Thailand accelerated through the crisis.
#               Education is embodied in people, not stored in budgets.
#
# Inputs:
#   wcde/data/processed/cohort_completion_both_long.csv
#   data/gdppercapita_us_inflation_adjusted.csv
#
# Section 4 / Section 6.2 of the paper.
# =============================================================================
"""
asian_financial_crisis.py

The Asian Financial Crisis (1997-98) as a natural experiment: abrupt,
exogenous income removal across five countries with dense educational data.
GDP collapsed; education trajectories were undisturbed. The cleanest
income-removal test in the dataset.
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "../wcde/data/processed")
DATA = os.path.join(SCRIPT_DIR, "../data")

# ── Load data ────────────────────────────────────────────────────────────────
long = pd.read_csv(os.path.join(PROC, "cohort_completion_both_long.csv"))
low_w = long.pivot(index="country", columns="cohort_year", values="lower_sec")

gdp = pd.read_csv(os.path.join(DATA, "gdppercapita_us_inflation_adjusted.csv"))

AFC_COUNTRIES = {
    "Republic of Korea": "South Korea",
    "Indonesia": "Indonesia",
    "Thailand": "Thailand",
    "Malaysia": "Malaysia",
    "Philippines": "Philippines",
}

EDU_YEARS = [1985, 1990, 1995, 2000, 2005, 2010, 2015]
GDP_YEARS = [1995, 1996, 1997, 1998, 1999, 2000, 2005]


def get_gdp_series(country_name):
    """Get GDP per capita for selected years."""
    row = gdp[gdp["Country"].str.lower() == country_name.lower()]
    if len(row) == 0:
        return {}
    row = row.iloc[0]
    result = {}
    for yr in GDP_YEARS:
        val = row.get(str(yr))
        if pd.notna(val):
            result[yr] = float(val)
    return result


def get_edu(wcde_name):
    """Get education completion for selected years."""
    if wcde_name not in low_w.index:
        return {}
    result = {}
    for yr in EDU_YEARS:
        try:
            val = float(low_w.loc[wcde_name, yr])
            if not np.isnan(val):
                result[yr] = val
        except (KeyError, ValueError):
            pass
    return result


# ── Print results by level ───────────────────────────────────────────────────
LEVELS = ["lower_sec", "upper_sec", "college"]

print("=" * 70)
print("  ASIAN FINANCIAL CRISIS: Income Removal Test")
print("  GDP collapsed 1997-98; education trajectory undisturbed")
print("  All levels: lower secondary, upper secondary, college")
print("=" * 70)

for level in LEVELS:
    wide = long.pivot(index="country", columns="cohort_year", values=level)
    print(f"\n  --- {level.upper().replace('_', ' ')} ---")
    print(f"  {'Country':<15} " + " ".join(f"{yr:>7}" for yr in EDU_YEARS))
    for wcde_name, gdp_name in AFC_COUNTRIES.items():
        if wcde_name not in wide.index:
            continue
        vals = []
        for yr in EDU_YEARS:
            try:
                v = float(wide.loc[wcde_name, yr])
                vals.append(f"{v:6.1f}%" if not np.isnan(v) else "    n/a")
            except (KeyError, ValueError):
                vals.append("    n/a")
        print(f"  {gdp_name:<15} " + " ".join(vals))

print()
print("=" * 70)
print("  DETAIL BY COUNTRY (lower secondary + GDP)")
print("=" * 70)
print()

for wcde_name, gdp_name in AFC_COUNTRIES.items():
    edu = get_edu(wcde_name)
    gdp_vals = get_gdp_series(gdp_name)

    if not edu:
        print(f"  {gdp_name}: not found in WCDE\n")
        continue

    print(f"  {gdp_name}")
    print(f"  {'Year':>6}  {'Education':>10}  {'GDP/cap':>10}")
    print(f"  {'-' * 32}")

    all_years = sorted(set(list(edu.keys()) + list(gdp_vals.keys())))
    for yr in all_years:
        e = f"{edu[yr]:.1f}%" if yr in edu else ""
        g = f"${gdp_vals[yr]:,.0f}" if yr in gdp_vals else ""
        print(f"  {yr:>6}  {e:>10}  {g:>10}")

    if 1997 in gdp_vals and 1998 in gdp_vals:
        drop = (gdp_vals[1998] - gdp_vals[1997]) / gdp_vals[1997] * 100
        print(f"  GDP drop 1997-1998: {drop:+.1f}%")

    if 1990 in edu and 1995 in edu and 2000 in edu:
        rate_pre = (edu[1995] - edu[1990]) / 5
        predicted_2000 = edu[1995] + rate_pre * 5
        actual_2000 = edu[2000]
        print(f"  Pre-crisis edu rate: {rate_pre:+.2f} pp/yr")
        print(f"  Predicted 2000: {predicted_2000:.1f}%")
        print(f"  Actual 2000: {actual_2000:.1f}%")
        print(f"  Deviation: {actual_2000 - predicted_2000:+.1f} pp")

    print()

print("Done.")
