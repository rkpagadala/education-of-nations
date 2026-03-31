"""
grandmother_effect.py
=====================
Test whether grandmother's education (T-50) independently predicts
fertility and other outcomes, controlling for mother's education (T-25).

If the PT mechanism transmits across two generations simultaneously,
grandmother's education should have a residual effect on:
  (a) granddaughter's fertility (TFR at T)
  (b) granddaughter's education (child edu at T)
  (c) life expectancy (LE at T)

This effect will be subtle — mother's education absorbs most of the
grandmother's influence. The test asks whether the grandmother adds
anything beyond what the mother already transmits.

Panel structure:
  - Grandmother education: E_{i, t-50}  (WCDE lower sec, both sexes, 20-24)
  - Mother education:      E_{i, t-25}  (WCDE lower sec, both sexes, 20-24)
  - Outcomes at time T:    TFR, LE, child education (WCDE at T+25)

Country fixed effects throughout.

Sources:
  - Education: WCDE v3, lower secondary completion, both sexes, age 20-24
  - TFR: World Bank WDI (SP.DYN.TFRT.IN)
  - LE: World Bank WDI (SP.DYN.LE00.IN)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from _shared import PROC, DATA, CHECKIN, REGIONS, NAME_MAP, WB_REGIONS

# ── Load data ─────────────────────────────────────────────────────────
edu_raw = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"))
le_raw = pd.read_csv(os.path.join(DATA, "life_expectancy_years.csv"))
tfr_raw = pd.read_csv(os.path.join(DATA, "children_per_woman_total_fertility.csv"))

# ── Filter to sovereign countries ─────────────────────────────────────
edu = edu_raw[~edu_raw["country"].isin(REGIONS)].copy()

# Map WCDE names to lowercase for WDI join
edu["country_lc"] = edu["country"].map(NAME_MAP).fillna(edu["country"].str.lower())

# WDI: lowercase country names
le_raw["Country"] = le_raw["Country"].str.lower()
tfr_raw["Country"] = tfr_raw["Country"].str.lower()

# Filter WDI regions
le = le_raw[~le_raw["Country"].isin(WB_REGIONS)].copy()
tfr = tfr_raw[~tfr_raw["Country"].isin(WB_REGIONS)].copy()


def build_panel(edu_df, outcome_df, outcome_name, lag_gm=50, lag_m=25):
    """Build three-generation panel.

    For each country-year observation:
      - grandmother_edu = education at T - lag_gm
      - mother_edu      = education at T - lag_m
      - outcome         = outcome variable at T

    Education is at 5-year intervals; outcomes are annual.
    We match education years exactly (5-year multiples) and use annual outcomes.
    """
    rows = []
    edu_years = sorted([int(c) for c in edu_df.columns if c.isdigit()])

    for _, row in edu_df.iterrows():
        country_wdi = row["country_lc"]
        outcome_row = outcome_df[outcome_df["Country"] == country_wdi]
        if outcome_row.empty:
            continue

        for t_edu_m in edu_years:
            t_edu_gm = t_edu_m - (lag_gm - lag_m)  # grandmother's education year
            t_outcome = t_edu_m + lag_m              # outcome year

            if str(t_edu_gm) not in edu_df.columns:
                continue

            gm_edu = row[str(t_edu_gm)]
            m_edu = row[str(t_edu_m)]

            if pd.isna(gm_edu) or pd.isna(m_edu):
                continue

            # Get outcome at t_outcome (annual)
            t_out_str = str(t_outcome)
            if t_out_str not in outcome_row.columns:
                continue
            outcome_val = outcome_row[t_out_str].values[0]
            if pd.isna(outcome_val):
                continue

            rows.append({
                "country": row["country"],
                "t_outcome": t_outcome,
                "grandmother_edu": gm_edu,
                "mother_edu": m_edu,
                outcome_name: outcome_val,
            })

    return pd.DataFrame(rows)


def build_edu_panel(edu_df, lag_gm=50, lag_m=25, lag_child=25):
    """Build three-generation panel for child education outcome.

    grandmother_edu at T-50, mother_edu at T-25, child_edu at T+25.
    Relative to mother's year T: GM at T-25, child at T+25.
    """
    rows = []
    edu_years = sorted([int(c) for c in edu_df.columns if c.isdigit()])

    for _, row in edu_df.iterrows():
        for t_m in edu_years:
            t_gm = t_m - (lag_gm - lag_m)
            t_child = t_m + lag_m  # child edu 25 years after mother

            if str(t_gm) not in edu_df.columns or str(t_child) not in edu_df.columns:
                continue

            gm_edu = row[str(t_gm)]
            m_edu = row[str(t_m)]
            child_edu = row[str(t_child)]

            if pd.isna(gm_edu) or pd.isna(m_edu) or pd.isna(child_edu):
                continue

            rows.append({
                "country": row["country"],
                "t_child": t_child,
                "grandmother_edu": gm_edu,
                "mother_edu": m_edu,
                "child_edu": child_edu,
            })

    return pd.DataFrame(rows)


def run_fe_regression(df, outcome_col, predictors, country_col="country"):
    """Run country fixed effects regression with clustered SEs."""
    df_clean = df.dropna(subset=[outcome_col] + predictors + [country_col])
    if len(df_clean) < 20:
        return None

    # Country dummies
    dummies = pd.get_dummies(df_clean[country_col], drop_first=True, dtype=float)
    X = pd.concat([df_clean[predictors].reset_index(drop=True),
                    dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = df_clean[outcome_col].reset_index(drop=True)

    try:
        # Clustered standard errors
        groups = df_clean[country_col].reset_index(drop=True)
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})

        # Within-R² (from demeaned regression)
        df_dm = df_clean.copy()
        means = df_dm.groupby(country_col)[[outcome_col] + predictors].transform("mean")
        y_dm = df_dm[outcome_col].values - means[outcome_col].values
        X_dm = df_dm[predictors].values - means[predictors].values
        if X_dm.shape[0] > X_dm.shape[1]:
            from numpy.linalg import lstsq
            beta, _, _, _ = lstsq(X_dm, y_dm, rcond=None)
            y_hat = X_dm @ beta
            ss_res = np.sum((y_dm - y_hat) ** 2)
            ss_tot = np.sum(y_dm ** 2)
            within_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        else:
            within_r2 = 0

        result = {
            "n": int(len(df_clean)),
            "n_countries": int(df_clean[country_col].nunique()),
            "within_r2": round(within_r2, 4),
        }
        for p in predictors:
            result[f"beta_{p}"] = round(float(model.params[p]), 4)
            result[f"pval_{p}"] = round(float(model.pvalues[p]), 4)
            result[f"se_{p}"] = round(float(model.bse[p]), 4)
        return result
    except Exception as e:
        print(f"  Regression failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# RUN TESTS
# ══════════════════════════════════════════════════════════════════════
results = {}

print("=" * 70)
print("GRANDMOTHER EFFECT TEST")
print("=" * 70)

# ── Test 1: TFR ──────────────────────────────────────────────────────
print("\n--- TFR (fertility) ---")
panel_tfr = build_panel(edu, tfr, "tfr")
print(f"  Panel: {len(panel_tfr)} obs, {panel_tfr['country'].nunique()} countries")

# Model 1: TFR ~ mother_edu only
m1 = run_fe_regression(panel_tfr, "tfr", ["mother_edu"])
print(f"  Model 1 (mother only):       β_m={m1['beta_mother_edu']:+.4f}  R²={m1['within_r2']:.4f}  n={m1['n']}")

# Model 2: TFR ~ mother_edu + grandmother_edu
m2 = run_fe_regression(panel_tfr, "tfr", ["mother_edu", "grandmother_edu"])
print(f"  Model 2 (mother + GM):       β_m={m2['beta_mother_edu']:+.4f}  β_gm={m2['beta_grandmother_edu']:+.4f}  (p={m2['pval_grandmother_edu']:.4f})  R²={m2['within_r2']:.4f}")

# Model 3: TFR ~ grandmother_edu only (for comparison)
m3 = run_fe_regression(panel_tfr, "tfr", ["grandmother_edu"])
print(f"  Model 3 (GM only):           β_gm={m3['beta_grandmother_edu']:+.4f}  R²={m3['within_r2']:.4f}")

r2_gain_tfr = m2["within_r2"] - m1["within_r2"]
print(f"  R² gain from adding GM:      {r2_gain_tfr:.4f}")
print(f"  GM β / Mother β ratio:       {abs(m2['beta_grandmother_edu'] / m2['beta_mother_edu']):.3f}")

results["tfr"] = {
    "mother_only": m1, "mother_gm": m2, "gm_only": m3,
    "r2_gain": round(r2_gain_tfr, 4),
}

# ── Test 2: Life expectancy ──────────────────────────────────────────
print("\n--- Life expectancy ---")
panel_le = build_panel(edu, le, "le")
print(f"  Panel: {len(panel_le)} obs, {panel_le['country'].nunique()} countries")

m1_le = run_fe_regression(panel_le, "le", ["mother_edu"])
m2_le = run_fe_regression(panel_le, "le", ["mother_edu", "grandmother_edu"])
m3_le = run_fe_regression(panel_le, "le", ["grandmother_edu"])

print(f"  Model 1 (mother only):       β_m={m1_le['beta_mother_edu']:+.4f}  R²={m1_le['within_r2']:.4f}")
print(f"  Model 2 (mother + GM):       β_m={m2_le['beta_mother_edu']:+.4f}  β_gm={m2_le['beta_grandmother_edu']:+.4f}  (p={m2_le['pval_grandmother_edu']:.4f})  R²={m2_le['within_r2']:.4f}")
print(f"  Model 3 (GM only):           β_gm={m3_le['beta_grandmother_edu']:+.4f}  R²={m3_le['within_r2']:.4f}")

r2_gain_le = m2_le["within_r2"] - m1_le["within_r2"]
print(f"  R² gain from adding GM:      {r2_gain_le:.4f}")

results["le"] = {
    "mother_only": m1_le, "mother_gm": m2_le, "gm_only": m3_le,
    "r2_gain": round(r2_gain_le, 4),
}

# ── Test 3: Child education ──────────────────────────────────────────
print("\n--- Child education (grandchild of GM) ---")
panel_edu = build_edu_panel(edu)
print(f"  Panel: {len(panel_edu)} obs, {panel_edu['country'].nunique()} countries")

m1_edu = run_fe_regression(panel_edu, "child_edu", ["mother_edu"])
m2_edu = run_fe_regression(panel_edu, "child_edu", ["mother_edu", "grandmother_edu"])
m3_edu = run_fe_regression(panel_edu, "child_edu", ["grandmother_edu"])

print(f"  Model 1 (mother only):       β_m={m1_edu['beta_mother_edu']:+.4f}  R²={m1_edu['within_r2']:.4f}")
print(f"  Model 2 (mother + GM):       β_m={m2_edu['beta_mother_edu']:+.4f}  β_gm={m2_edu['beta_grandmother_edu']:+.4f}  (p={m2_edu['pval_grandmother_edu']:.4f})  R²={m2_edu['within_r2']:.4f}")
print(f"  Model 3 (GM only):           β_gm={m3_edu['beta_grandmother_edu']:+.4f}  R²={m3_edu['within_r2']:.4f}")

r2_gain_edu = m2_edu["within_r2"] - m1_edu["within_r2"]
print(f"  R² gain from adding GM:      {r2_gain_edu:.4f}")

results["child_edu"] = {
    "mother_only": m1_edu, "mother_gm": m2_edu, "gm_only": m3_edu,
    "r2_gain": round(r2_gain_edu, 4),
}

# ── Test 4: Restrict to low-education countries (where effect should be strongest)
print("\n--- TFR, countries with mother edu < 50% ---")
panel_tfr_low = panel_tfr[panel_tfr["mother_edu"] < 50].copy()
print(f"  Panel: {len(panel_tfr_low)} obs, {panel_tfr_low['country'].nunique()} countries")

if len(panel_tfr_low) > 30 and panel_tfr_low["country"].nunique() > 5:
    m1_low = run_fe_regression(panel_tfr_low, "tfr", ["mother_edu"])
    m2_low = run_fe_regression(panel_tfr_low, "tfr", ["mother_edu", "grandmother_edu"])
    if m1_low and m2_low:
        print(f"  Model 1 (mother only):       β_m={m1_low['beta_mother_edu']:+.4f}  R²={m1_low['within_r2']:.4f}")
        print(f"  Model 2 (mother + GM):       β_m={m2_low['beta_mother_edu']:+.4f}  β_gm={m2_low['beta_grandmother_edu']:+.4f}  (p={m2_low['pval_grandmother_edu']:.4f})  R²={m2_low['within_r2']:.4f}")
        r2_gain_low = m2_low["within_r2"] - m1_low["within_r2"]
        print(f"  R² gain from adding GM:      {r2_gain_low:.4f}")
        results["tfr_low_edu"] = {
            "mother_only": m1_low, "mother_gm": m2_low,
            "r2_gain": round(r2_gain_low, 4),
        }

# ── Test 5: Does GM education predict mother's education? ────────────
# (This is really the parent→child test at the previous generation)
print("\n--- Mother's education ~ GM education (one-gen transmission) ---")
panel_m_gm = panel_edu[["country", "grandmother_edu", "mother_edu"]].drop_duplicates()
m_gm = run_fe_regression(panel_m_gm, "mother_edu", ["grandmother_edu"])
if m_gm:
    print(f"  β_gm={m_gm['beta_grandmother_edu']:+.4f}  R²={m_gm['within_r2']:.4f}  n={m_gm['n']}")
    results["gm_to_mother"] = m_gm

# ── Test 6: Decomposition — how much of GM's effect runs through mother?
print("\n" + "=" * 70)
print("DECOMPOSITION")
print("=" * 70)

# For TFR:
gm_total = m3["beta_grandmother_edu"]  # GM alone
gm_direct = m2["beta_grandmother_edu"]  # GM controlling for mother
gm_indirect = gm_total - gm_direct     # part running through mother
pct_direct = abs(gm_direct / gm_total) * 100 if gm_total != 0 else 0
pct_indirect = abs(gm_indirect / gm_total) * 100 if gm_total != 0 else 0

print(f"\n  TFR:")
print(f"    GM total effect (alone):     {gm_total:+.4f}")
print(f"    GM direct effect (| mother): {gm_direct:+.4f}  ({pct_direct:.1f}%)")
print(f"    GM indirect (via mother):    {gm_indirect:+.4f}  ({pct_indirect:.1f}%)")

results["decomposition_tfr"] = {
    "gm_total": round(gm_total, 4),
    "gm_direct": round(gm_direct, 4),
    "gm_indirect": round(gm_indirect, 4),
    "pct_direct": round(pct_direct, 1),
    "pct_indirect": round(pct_indirect, 1),
}

# For LE:
gm_total_le = m3_le["beta_grandmother_edu"]
gm_direct_le = m2_le["beta_grandmother_edu"]
gm_indirect_le = gm_total_le - gm_direct_le
pct_direct_le = abs(gm_direct_le / gm_total_le) * 100 if gm_total_le != 0 else 0
pct_indirect_le = abs(gm_indirect_le / gm_total_le) * 100 if gm_total_le != 0 else 0

print(f"\n  Life expectancy:")
print(f"    GM total effect (alone):     {gm_total_le:+.4f}")
print(f"    GM direct effect (| mother): {gm_direct_le:+.4f}  ({pct_direct_le:.1f}%)")
print(f"    GM indirect (via mother):    {gm_indirect_le:+.4f}  ({pct_indirect_le:.1f}%)")

results["decomposition_le"] = {
    "gm_total": round(gm_total_le, 4),
    "gm_direct": round(gm_direct_le, 4),
    "gm_indirect": round(gm_indirect_le, 4),
    "pct_direct": round(pct_direct_le, 1),
    "pct_indirect": round(pct_indirect_le, 1),
}

# ── Save checkin ──────────────────────────────────────────────────────
output = {
    "method": ("Grandmother effect test: does GM education (T-50) predict "
               "outcomes controlling for mother education (T-25)? "
               "Country FE, clustered SEs. WCDE v3 + World Bank WDI."),
    "results": results,
}

outpath = os.path.join(CHECKIN, "grandmother_effect.json")
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {outpath}")
