"""
edu_vs_gdp_by_level.py
=======================
Entry-cohort + ceiling analysis for three education levels:
  - Primary completion
  - Lower secondary completion
  - Upper secondary completion

For each level, runs the same design as edu_vs_gdp_entry_ceiling.py:
  1. Interpolate education to annual values
  2. For each (entry_threshold, ceiling) pair, find countries that crossed
     entry, include observations while education <= ceiling
  3. Country fixed-effects: education(T) → LE(T+25) vs GDP(T) → LE(T+25)

Entry thresholds: 10% to 90% (1% steps)
Ceilings: 50%, 60%, 70%, 80%, 90%
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROC = os.path.join(REPO_ROOT, "wcde", "data", "processed")
DATA = os.path.join(REPO_ROOT, "data")
CHECKIN = os.path.join(REPO_ROOT, "checkin")

REGIONS = {
    "Africa", "Asia", "Europe", "World", "Oceania", "Caribbean",
    "Central America", "South America",
    "Latin America and the Caribbean",
    "Central Asia", "Eastern Africa", "Eastern Asia", "Eastern Europe",
    "Northern Africa", "Northern America", "Northern Europe",
    "Southern Africa", "Southern Asia", "Southern Europe",
    "Western Africa", "Western Asia", "Western Europe",
    "Middle Africa", "South-Eastern Asia",
    "Melanesia", "Micronesia", "Polynesia",
    "Less developed regions", "More developed regions",
    "Least developed countries",
    "Australia and New Zealand", "Channel Islands", "Sub-Saharan Africa",
}

NAME_MAP = {
    "Viet Nam": "vietnam",
    "Iran (Islamic Republic of)": "iran",
    "Republic of Korea": "south korea",
    "United States of America": "united states",
    "United Kingdom of Great Britain and Northern Ireland": "united kingdom",
    "Russian Federation": "russia",
    "United Republic of Tanzania": "tanzania",
    "Democratic Republic of the Congo": "congo, dem. rep.",
    "Congo": "congo, rep.",
    "Bolivia (Plurinational State of)": "bolivia",
    "Venezuela (Bolivarian Republic of)": "venezuela",
    "Republic of Moldova": "moldova",
    "Syrian Arab Republic": "syria",
    "Taiwan Province of China": "taiwan",
    "Lao People's Democratic Republic": "laos",
    "Türkiye": "turkey",
    "Eswatini": "swaziland",
    "Cabo Verde": "cape verde",
    "Czechia": "czech republic",
    "North Macedonia": "macedonia",
}

# ── Load data ────────────────────────────────────────────────────────

print("Loading data...")

edu_raw = pd.read_csv(os.path.join(PROC, "completion_both_long.csv"))
edu_raw = edu_raw[~edu_raw["country"].isin(REGIONS)].copy()

le_raw = pd.read_csv(os.path.join(DATA, "life_expectancy_years.csv"))
le_raw["Country"] = le_raw["Country"].str.lower()
le_raw = le_raw.set_index("Country")
for c in le_raw.columns:
    le_raw[c] = pd.to_numeric(le_raw[c], errors="coerce")

gdp_raw = pd.read_csv(os.path.join(DATA, "gdppercapita_us_inflation_adjusted.csv"))
gdp_raw["Country"] = gdp_raw["Country"].str.lower()
gdp_raw = gdp_raw.set_index("Country")
for c in gdp_raw.columns:
    gdp_raw[c] = pd.to_numeric(gdp_raw[c], errors="coerce")


def get_wb_val(df, wcde_name, year):
    key = NAME_MAP.get(wcde_name, wcde_name).lower()
    for k in [wcde_name.lower(), key]:
        if k in df.index:
            try:
                v = float(df.loc[k, str(year)])
                return v if not np.isnan(v) else np.nan
            except (KeyError, ValueError):
                pass
    return np.nan


def fe_r2(x_col, y_col, data):
    sub = data.dropna(subset=[x_col, y_col]).copy()
    counts = sub.groupby("country").size()
    keep = counts[counts >= 2].index
    sub = sub[sub["country"].isin(keep)]
    n_countries = sub["country"].nunique()
    if n_countries < 3 or len(sub) < 10:
        return np.nan, 0, 0
    sub[x_col + "_dm"] = sub[x_col] - sub.groupby("country")[x_col].transform("mean")
    sub[y_col + "_dm"] = sub[y_col] - sub.groupby("country")[y_col].transform("mean")
    X = sub[x_col + "_dm"].values.reshape(-1, 1)
    y = sub[y_col + "_dm"].values
    ok = ~np.isnan(X.ravel()) & ~np.isnan(y)
    if ok.sum() < 10:
        return np.nan, 0, 0
    reg = LinearRegression(fit_intercept=False).fit(X[ok], y[ok])
    return reg.score(X[ok], y[ok]), int(ok.sum()), n_countries


def find_entry_year(series, threshold):
    above = series[series >= threshold]
    return above.index[0] if len(above) > 0 else None


# ── Education levels to test ─────────────────────────────────────────

EDU_LEVELS = {
    "primary": "primary",
    "lower_secondary": "lower_sec",
    "upper_secondary": "upper_sec",
}

T_YEARS = list(range(1960, 1995, 5))
LAG = 25
CEILINGS = [50, 60, 70, 80, 90]

all_results = {}

for level_name, col_name in EDU_LEVELS.items():
    print(f"\n{'#' * 90}")
    print(f"# EDUCATION LEVEL: {level_name.upper().replace('_', ' ')}")
    print(f"# Column: {col_name}")
    print(f"{'#' * 90}")

    # Interpolate to annual
    edu_annual = {}
    for c, grp in edu_raw.groupby("country"):
        s = grp.set_index("year")[col_name].sort_index()
        full_idx = range(s.index.min(), s.index.max() + 1)
        edu_annual[c] = s.reindex(full_idx).interpolate(method="linear")

    # Build panel
    rows = []
    for c in sorted(edu_annual.keys()):
        s = edu_annual[c]
        for t in T_YEARS:
            if t not in s.index:
                continue
            edu_val = s[t]
            gdp_t = get_wb_val(gdp_raw, c, t)
            le_tp25 = get_wb_val(le_raw, c, t + LAG)
            if np.isnan(edu_val) or np.isnan(le_tp25):
                continue
            rows.append({
                "country": c, "t": t, "edu_t": edu_val,
                "log_gdp_t": np.log(gdp_t) if not np.isnan(gdp_t) and gdp_t > 0 else np.nan,
                "le_tp25": le_tp25,
            })

    panel = pd.DataFrame(rows)
    print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")

    # Precompute entry years
    entry_years = {}
    for threshold in range(10, 91):
        cohort = {}
        for c, s in edu_annual.items():
            entry = find_entry_year(s, threshold)
            if entry is not None:
                cohort[c] = entry
        entry_years[threshold] = cohort

    # Run ceiling sweep
    level_results = {}

    for ceiling in CEILINGS:
        print(f"\n  Ceiling = {ceiling}%")
        print(f"  {'Entry':<10} {'Edu R²':>8} {'n':>6} {'Ctry':>6}   "
              f"{'GDP R²':>8} {'n':>6} {'Ctry':>6}   {'Ratio':>8}")
        print(f"  {'-' * 80}")

        ceil_results = {}

        for threshold in range(10, 91):
            if threshold > ceiling:
                break

            cohort = entry_years[threshold]
            if len(cohort) < 3:
                continue

            mask = panel.apply(
                lambda r, c=cohort, ceil=ceiling: (
                    r["country"] in c
                    and r["t"] >= c[r["country"]]
                    and r["edu_t"] <= ceil
                ),
                axis=1
            )
            sub = panel[mask]
            if len(sub) < 10:
                continue

            r2_e, n_e, c_e = fe_r2("edu_t", "le_tp25", sub)
            r2_g, n_g, c_g = fe_r2("log_gdp_t", "le_tp25", sub)

            if not np.isnan(r2_g) and r2_g > 0.001:
                ratio_s = f"{r2_e / r2_g:.1f}x"
            elif not np.isnan(r2_e):
                ratio_s = "GDP≈0"
            else:
                ratio_s = "n/a"

            r2_e_s = f"{r2_e:.3f}" if not np.isnan(r2_e) else "n/a"
            r2_g_s = f"{r2_g:.3f}" if not np.isnan(r2_g) else "n/a"

            if threshold % 10 == 0 or threshold == 10:
                print(f"  >= {threshold}%{'':<4} {r2_e_s:>8} {n_e:>6} {c_e:>6}   "
                      f"{r2_g_s:>8} {n_g:>6} {c_g:>6}   {ratio_s:>8}")

            ceil_results[str(threshold)] = {
                "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
                "edu_n": n_e, "edu_countries": c_e,
                "gdp_r2": round(r2_g, 3) if not np.isnan(r2_g) else None,
                "gdp_n": n_g, "gdp_countries": c_g,
            }

        level_results[str(ceiling)] = ceil_results

    all_results[level_name] = level_results

    # Summary for this level
    print(f"\n  SUMMARY ({level_name}): entry=10%, varying ceiling")
    print(f"  {'Ceiling':<10} {'Edu R²':>8} {'GDP R²':>8} {'Ratio':>8} {'n':>6} {'Ctry':>6}")
    print(f"  {'-' * 50}")
    for ceiling in CEILINGS:
        r = level_results[str(ceiling)].get("10", {})
        if r and r.get("edu_r2") is not None:
            ratio = f"{r['edu_r2']/r['gdp_r2']:.1f}x" if r.get("gdp_r2") and r["gdp_r2"] > 0.001 else "GDP≈0"
            print(f"  <= {ceiling}%{'':<4} {r['edu_r2']:>8.3f} {r.get('gdp_r2', 0):>8.3f} "
                  f"{ratio:>8} {r['edu_n']:>6} {r['edu_countries']:>6}")


# ── Cross-level comparison ───────────────────────────────────────────

print(f"\n\n{'=' * 90}")
print("CROSS-LEVEL COMPARISON: entry=10%, ceiling=60%")
print(f"{'=' * 90}")
print(f"{'Level':<20} {'Edu R²':>8} {'GDP R²':>8} {'Ratio':>8} {'n':>6} {'Ctry':>6}")
print("-" * 60)
for level_name in EDU_LEVELS:
    r = all_results[level_name].get("60", {}).get("10", {})
    if r and r.get("edu_r2") is not None:
        ratio = f"{r['edu_r2']/r['gdp_r2']:.1f}x" if r.get("gdp_r2") and r["gdp_r2"] > 0.001 else "GDP≈0"
        print(f"{level_name:<20} {r['edu_r2']:>8.3f} {r.get('gdp_r2', 0):>8.3f} "
              f"{ratio:>8} {r['edu_n']:>6} {r['edu_countries']:>6}")

# ── Checkin ──────────────────────────────────────────────────────────

checkin = {
    "script": "scripts/edu_vs_gdp_by_level.py",
    "produced": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "method": (
        "Country FE (demeaned). Entry-cohort with ceiling, tested at three "
        "education levels: primary, lower secondary, upper secondary. "
        "T = 1960-1990 (5yr), outcome = LE(T+25). "
        "Education interpolated from WCDE 5-year to annual (linear)."
    ),
    "levels": all_results,
}

os.makedirs(CHECKIN, exist_ok=True)
out_path = os.path.join(CHECKIN, "edu_vs_gdp_by_level.json")
with open(out_path, "w") as f:
    json.dump(checkin, f, indent=2)
print(f"\nCheckin written to {out_path}")
