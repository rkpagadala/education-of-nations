"""
verify_table_a1_cutoffs.py

Verify Table A1 cutoff-specific rows from "Education of Nations":
  - Row 1 (<30%): beta=0.719, R2=0.153, n=868, 147 countries
  - Row 2 (<20%): beta=1.033, R2=0.150, n=638, 116 countries
  - Row 3 (<10%): beta=1.037, R2=0.070, n=363, 76 countries
  - Row 4 (all):  beta=0.091, R2=0.011, n=1,917, 213 countries
  - t-statistics: t=12.5 at <30%, t=10.6 at <20%

Runs two-way fixed effects (country + year) on the post-1975 panel at
each parental education cutoff, using the same methodology as
scripts/table_a1_two_way_fe.py.

Data sources:
  - Education: wcde/data/processed/lower_sec_both.csv (WCDE v3)

Output: checkin/table_a1_cutoffs.json
"""

import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROC_DIR = os.path.join(REPO_ROOT, "wcde", "data", "processed")
CHECKIN_DIR = os.path.join(REPO_ROOT, "checkin")
os.makedirs(CHECKIN_DIR, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
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

# ── Load education data ─────────────────────────────────────────────────────
agg = pd.read_csv(os.path.join(PROC_DIR, "lower_sec_both.csv"), index_col="country")

# ── Build panel ──────────────────────────────────────────────────────────────
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
        rows.append({
            "country": country,
            "year": y,
            "child": child,
            "parent": parent,
        })

panel = pd.DataFrame(rows)
print(f"Full panel: {len(panel)} obs, {panel['country'].nunique()} countries")


# ── Two-way FE regression ───────────────────────────────────────────────────
def two_way_fe(df, x_col="parent", y_col="child", cluster_col="country"):
    """
    Two-way FE: demean within country, then within year.
    Cluster-robust SEs by country.
    Returns model, n_obs, n_countries.
    """
    d = df.dropna(subset=[x_col, y_col]).copy()

    # Demean within country
    d[x_col + "_dm1"] = d.groupby(cluster_col)[x_col].transform(lambda x: x - x.mean())
    d[y_col + "_dm1"] = d.groupby(cluster_col)[y_col].transform(lambda x: x - x.mean())

    # Demean within year
    d[x_col + "_dm2"] = d.groupby("year")[x_col + "_dm1"].transform(lambda x: x - x.mean())
    d[y_col + "_dm2"] = d.groupby("year")[y_col + "_dm1"].transform(lambda x: x - x.mean())

    X = d[[x_col + "_dm2"]]
    y = d[y_col + "_dm2"]

    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": d[cluster_col]},
    )
    return model, len(d), d[cluster_col].nunique()


# ── Cutoff analysis ─────────────────────────────────────────────────────────
CUTOFFS = [10, 20, 30]

print("\n" + "=" * 70)
print("TABLE A1 CUTOFF-SPECIFIC ROWS (two-way FE, clustered SEs)")
print("=" * 70)

results = {}

for cutoff in CUTOFFS:
    sub = panel[panel["parent"] < cutoff]
    if len(sub) < 10:
        print(f"\n  <{cutoff}%: insufficient data ({len(sub)} obs)")
        continue
    model, n, nc = two_way_fe(sub)
    beta = model.params.iloc[0]
    se = model.bse.iloc[0]
    t = model.tvalues.iloc[0]
    p = model.pvalues.iloc[0]
    r2 = model.rsquared

    label = f"cutoff_{cutoff}"
    results[label] = {
        "beta": round(beta, 3),
        "se": round(se, 3),
        "t": round(t, 1),
        "p": round(p, 4),
        "r2": round(r2, 3),
        "n": n,
        "countries": nc,
    }

    print(f"\n  <{cutoff}%: beta={beta:.3f}  SE={se:.3f}  t={t:.1f}  "
          f"R2={r2:.3f}  n={n}  countries={nc}")

# Full panel (no cutoff)
model_all, n_all, nc_all = two_way_fe(panel)
beta_all = model_all.params.iloc[0]
se_all = model_all.bse.iloc[0]
t_all = model_all.tvalues.iloc[0]
p_all = model_all.pvalues.iloc[0]
r2_all = model_all.rsquared

results["all"] = {
    "beta": round(beta_all, 3),
    "se": round(se_all, 3),
    "t": round(t_all, 1),
    "p": round(p_all, 4),
    "r2": round(r2_all, 3),
    "n": n_all,
    "countries": nc_all,
}

print(f"\n  All:   beta={beta_all:.3f}  SE={se_all:.3f}  t={t_all:.1f}  "
      f"R2={r2_all:.3f}  n={n_all}  countries={nc_all}")

# ── Compare with paper claims ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON WITH PAPER CLAIMS")
print("=" * 70)

paper_claims = {
    "cutoff_30": {"beta": 0.719, "r2": 0.153, "n": 868, "countries": 147, "t": 12.5},
    "cutoff_20": {"beta": 1.033, "r2": 0.150, "n": 638, "countries": 116, "t": 10.6},
    "cutoff_10": {"beta": 1.037, "r2": 0.070, "n": 363, "countries": 76},
    "all":       {"beta": 0.091, "r2": 0.011, "n": 1917, "countries": 213},
}

for label, claim in paper_claims.items():
    actual = results.get(label)
    if actual is None:
        print(f"\n  {label}: no actual result")
        continue
    cutoff_str = label.replace("cutoff_", "<") + "%" if "cutoff" in label else "all"
    print(f"\n  {cutoff_str}:")
    for key in ["beta", "r2", "n", "countries"]:
        if key in claim:
            c = claim[key]
            a = actual[key]
            match = "OK" if (isinstance(c, float) and abs(c - a) < 0.05) or c == a else "DIFF"
            print(f"    {key:>10s}: paper={c}  actual={a}  [{match}]")
    if "t" in claim:
        print(f"    {'t':>10s}: paper={claim['t']}  actual={actual.get('t', 'N/A')}")

# ── Write checkin ────────────────────────────────────────────────────────────
checkin = {
    "script": "scripts/verify_table_a1_cutoffs.py",
    "produced": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "notes": f"Two-way FE (country + year), clustered SEs. Full panel: {n_all} obs, {nc_all} countries.",
    "numbers": results,
    "paper_claims": paper_claims,
}

out_path = os.path.join(CHECKIN_DIR, "table_a1_cutoffs.json")
with open(out_path, "w") as f:
    json.dump(checkin, f, indent=2)
print(f"\nCheckin written: {out_path}")
