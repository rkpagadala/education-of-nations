# =============================================================================
# PAPER REFERENCE
# Script:  scripts/beta_by_ceiling_cutoff.py
# Paper:   "Education of Nations"
#
# Produces:
#   Table showing β and R² at parental education cutoffs from <10% to no cutoff
#   Two panels: (A) all countries 1900–2015, (B) post-1975 panel
#   Key finding: β > 1 at every cutoff below 90% — education amplifies
#               across generations, dragged below unity only by ceiling noise
#
# Inputs:
#   wcde/data/processed/cohort_completion_both_long.csv
#
# Key parameters:
#   GENERATIONAL_LAG = 25
#   CUTOFFS = 10, 20, ..., 90
# =============================================================================
"""
beta_by_ceiling_cutoff.py

Shows that the pooled β is mechanically suppressed by ceiling observations.
When countries near 100% completion are excluded, β exceeds unity at every
cutoff — education gains amplify across generations rather than depreciating.

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

# ── Load data ────────────────────────────────────────────────────────────────
long = pd.read_csv(os.path.join(PROC, "cohort_completion_both_long.csv"))
low_w = long.pivot(index="country", columns="cohort_year", values="lower_sec")


def v(df_w, country, year):
    """Look up a value from wide-format DataFrame."""
    try:
        val = float(df_w.loc[country, int(year)])
        return val if not np.isnan(val) else np.nan
    except (KeyError, ValueError):
        return np.nan


def build_panel(child_cohorts):
    """Build parent-child panel for all countries."""
    rows = []
    for c in low_w.index:
        for child_yr in child_cohorts:
            parent_yr = child_yr - 25
            child_low = v(low_w, c, child_yr)
            parent_low = v(low_w, c, parent_yr)
            if np.isnan(child_low) or np.isnan(parent_low):
                continue
            rows.append({
                "country": c,
                "cohort_year": child_yr,
                "child_low": child_low,
                "parent_low": parent_low,
            })
    return pd.DataFrame(rows)


def run_fe(df):
    """Run country fixed-effects regression via demeaning."""
    sub = df.copy()
    counts = sub.groupby("country").size()
    valid = counts[counts > 1].index
    sub = sub[sub["country"].isin(valid)]
    if len(sub) < 10:
        return np.nan, np.nan, 0, 0
    sub["child_dm"] = sub["child_low"] - sub.groupby("country")["child_low"].transform("mean")
    sub["parent_dm"] = sub["parent_low"] - sub.groupby("country")["parent_low"].transform("mean")
    reg = LinearRegression(fit_intercept=False).fit(
        sub[["parent_dm"]].values, sub["child_dm"].values
    )
    r2 = reg.score(sub[["parent_dm"]].values, sub["child_dm"].values)
    return reg.coef_[0], r2, len(sub), sub["country"].nunique()


CUTOFFS = list(range(10, 100, 10))


def print_panel(panel, label):
    """Print β and R² at each cutoff for a panel."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {len(panel)} obs, {panel['country'].nunique()} countries")
    print(f"{'='*60}")
    print(f"{'Cutoff':>10} {'beta':>8} {'R2':>8} {'n':>8} {'Countries':>10}")
    print("-" * 50)

    for cutoff in CUTOFFS:
        sub = panel[panel["parent_low"] < cutoff]
        beta, r2, n, nc = run_fe(sub)
        if n == 0:
            continue
        print(f"    <{cutoff:3d}%  {beta:8.3f} {r2:8.3f} {n:8d} {nc:10d}")

    beta, r2, n, nc = run_fe(panel)
    print(f"  no cut  {beta:8.3f} {r2:8.3f} {n:8d} {nc:10d}")


# ── Panel A: All countries, 1900–2015 ────────────────────────────────────────
panel_full = build_panel(list(range(1900, 2016, 5)))
print_panel(panel_full, "Panel A: All countries, 1900-2015")

# ── Panel B: Post-1975 (matching Table 1 time range) ────────────────────────
panel_post75 = build_panel(list(range(1975, 2016, 5)))
print_panel(panel_post75, "Panel B: Post-1975 (Table 1 comparison)")

print("\nDone.")
