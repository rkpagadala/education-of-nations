"""
co2_placebo.py

CO2 placebo test for Table A2 of:
  "Education of Nations"

If parental education's predictive power (R²=0.455, Table 1) were a
time-trend artefact, any monotone trending variable lagged 25 years
should produce a comparable R². CO2 emissions per capita has the same
global upward trend but no theoretical link to child education.

Design:
  - Outcome: child lower secondary completion at T (WCDE v3, 1975–2015)
  - Predictor: CO2 emissions per capita at T−25, log-transformed
  - Country fixed effects (demeaning), clustered SEs
  - Same panel construction as table_1_main.py

Data sources:
  - Education: wcde/data/processed/lower_sec_both.csv (WCDE v3)
  - CO2: data/co2_emissions_tonnes_per_person.csv (World Bank WDI)

Output: CO2 placebo R² for comparison against Table 1 R²=0.455.
"""

import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROC_DIR = os.path.join(REPO_ROOT, "wcde", "data", "processed")
DATA_DIR = os.path.join(REPO_ROOT, "data")

PARENTAL_LAG = 25
OUTCOME_YEARS = list(range(1975, 2016, 5))

NON_SOVEREIGN = [
    "Africa", "Asia", "Europe", "Latin America and the Caribbean",
    "Northern America", "Oceania", "World",
    "Less developed regions", "More developed regions",
    "Least developed countries",
    "Eastern Africa", "Middle Africa", "Northern Africa",
    "Southern Africa", "Western Africa",
    "Eastern Asia", "South-Central Asia", "South-Eastern Asia", "Western Asia",
    "Eastern Europe", "Northern Europe", "Southern Europe", "Western Europe",
    "Caribbean", "Central America", "South America",
    "Australia and New Zealand", "Melanesia", "Micronesia", "Polynesia",
    "Channel Islands", "Sub-Saharan Africa",
]

# ── Load data ─────────────────────────────────────────────────────
agg = pd.read_csv(os.path.join(PROC_DIR, "lower_sec_both.csv"), index_col="country")

co2_raw = pd.read_csv(os.path.join(DATA_DIR, "co2_emissions_tonnes_per_person.csv"),
                       index_col="Country")
co2_raw.index = co2_raw.index.str.lower()

# ── Build panel ───────────────────────────────────────────────────
rows = []
for country in agg.index:
    if country in NON_SOVEREIGN:
        continue
    for y in OUTCOME_YEARS:
        y_lag = y - PARENTAL_LAG
        sy, sy_lag = str(y), str(y_lag)

        if sy not in agg.columns or sy_lag not in agg.columns:
            continue
        child = agg.loc[country, sy]
        parent = agg.loc[country, sy_lag]
        if np.isnan(child) or np.isnan(parent):
            continue

        log_co2 = np.nan
        c_lower = country.lower()
        if c_lower in co2_raw.index and str(y_lag) in co2_raw.columns:
            try:
                g = float(co2_raw.loc[c_lower, str(y_lag)])
                if g > 0:
                    log_co2 = np.log(g)
            except (ValueError, TypeError):
                pass

        rows.append({
            "country": country,
            "year": y,
            "child": child,
            "parent": parent,
            "log_co2": log_co2,
        })

panel = pd.DataFrame(rows)
co2_panel = panel.dropna(subset=["log_co2"])
print(f"Full panel:  {len(panel)} obs, {panel['country'].nunique()} countries")
print(f"With CO2:    {len(co2_panel)} obs, {co2_panel['country'].nunique()} countries")

# ── Country FE regression via demeaning + clustered SEs ───────────
def fe_regression(df, x_cols, y_col, cluster_col="country"):
    d = df.dropna(subset=x_cols + [y_col]).copy()
    for col in x_cols + [y_col]:
        d[col + "_dm"] = d.groupby(cluster_col)[col].transform(
            lambda x: x - x.mean()
        )
    X = d[[c + "_dm" for c in x_cols]]
    y = d[y_col + "_dm"]
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": d[cluster_col]},
    )
    return model, len(d), d[cluster_col].nunique()

# ── Results ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CO2 PLACEBO TEST (Table A2)")
print("=" * 70)

# Parental education (same as Table 1 Model 1, for comparison)
m_edu, n_edu, nc_edu = fe_regression(panel, ["parent"], "child")
print(f"\n(ref) FE: child ~ parent_edu  [N={n_edu}, {nc_edu} countries]")
print(f"    β = {m_edu.params.iloc[0]:.3f}  (SE={m_edu.bse.iloc[0]:.3f})")
print(f"    R² (within) = {m_edu.rsquared:.3f}")

# CO2 placebo
m_co2, n_co2, nc_co2 = fe_regression(panel, ["log_co2"], "child")
print(f"\n(placebo) FE: child ~ log_CO2(T-25)  [N={n_co2}, {nc_co2} countries]")
print(f"    β = {m_co2.params.iloc[0]:.3f}  (SE={m_co2.bse.iloc[0]:.3f})")
print(f"    R² (within) = {m_co2.rsquared:.3f}")

ratio = m_edu.rsquared / m_co2.rsquared if m_co2.rsquared > 0 else float("inf")
print(f"\nRatio: edu R² / CO2 R² = {m_edu.rsquared:.3f} / {m_co2.rsquared:.3f} = {ratio:.0f}x")

print("\n" + "=" * 70)
print("Numbers for paper:")
print("=" * 70)
print(f"  CO2 placebo R² = {m_co2.rsquared:.3f}")
print(f"  Parental edu R² = {m_edu.rsquared:.3f}")
print(f"  Ratio = {ratio:.0f}x")

# ── Write checkin JSON ───────────────────────────────────────────
CHECKIN_DIR = os.path.join(REPO_ROOT, "checkin")
os.makedirs(CHECKIN_DIR, exist_ok=True)
checkin = {
    "script": "scripts/co2_placebo.py",
    "produced": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "numbers": {
        "co2_placebo_r2": round(m_co2.rsquared, 3),
        "edu_ref_r2": round(m_edu.rsquared, 3),
        "ratio": round(ratio, 0),
        "CO2-R2": round(m_co2.rsquared, 3),
    },
}
checkin_path = os.path.join(CHECKIN_DIR, "co2_placebo.json")
with open(checkin_path, "w") as f:
    json.dump(checkin, f, indent=2)
print(f"\nCheckin written to {checkin_path}")
