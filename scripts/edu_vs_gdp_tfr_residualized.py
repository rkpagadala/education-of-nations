"""
edu_vs_gdp_tfr_residualized.py
================================
Does education or GDP predict future fertility (TFR)?
And does GDP have any independent effect after controlling for
education's contribution to GDP?

Same design as edu_vs_gdp_residualized.py but with TFR(T+25) as outcome.

METHOD
------
Country fixed effects, entry-cohort with ceiling.
  - Edu R²:     education(T) → TFR(T+25)
  - GDP R²:     log_GDP(T) → TFR(T+25)
  - Resid R²:   GDP_residual(T) → TFR(T+25)
  - Edu→GDP:    how much of GDP is explained by education

Three education levels: primary, lower secondary, upper secondary.
T = 1960-1990 (5yr), lag = 25 years.
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

tfr_raw = pd.read_csv(os.path.join(DATA, "children_per_woman_total_fertility.csv"))
tfr_raw["Country"] = tfr_raw["Country"].str.lower()
tfr_raw = tfr_raw.set_index("Country")
for c in tfr_raw.columns:
    tfr_raw[c] = pd.to_numeric(tfr_raw[c], errors="coerce")
print(f"  TFR: {len(tfr_raw)} countries")

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


def find_entry_year(series, threshold):
    above = series[series >= threshold]
    return above.index[0] if len(above) > 0 else None


def fe_demean(col, data):
    """Demean a column by country (country fixed effects)."""
    return col - data.groupby("country")[col.name].transform("mean")


def fe_r2(x_col, y_col, data):
    """Standard country-FE R²."""
    sub = data.dropna(subset=[x_col, y_col]).copy()
    counts = sub.groupby("country").size()
    sub = sub[sub["country"].isin(counts[counts >= 2].index)]
    n_countries = sub["country"].nunique()
    if n_countries < 3 or len(sub) < 10:
        return np.nan, 0, 0
    xdm = sub[x_col] - sub.groupby("country")[x_col].transform("mean")
    ydm = sub[y_col] - sub.groupby("country")[y_col].transform("mean")
    X = xdm.values.reshape(-1, 1)
    y = ydm.values
    ok = ~np.isnan(X.ravel()) & ~np.isnan(y)
    if ok.sum() < 10:
        return np.nan, 0, 0
    reg = LinearRegression(fit_intercept=False).fit(X[ok], y[ok])
    return reg.score(X[ok], y[ok]), int(ok.sum()), n_countries


def fe_residualize_gdp(data):
    """
    Residualize GDP against education with country FE.
    Returns a copy of data with 'gdp_resid' column = the part of
    log_gdp not explained by education.
    """
    sub = data.dropna(subset=["edu_t", "log_gdp_t"]).copy()
    counts = sub.groupby("country").size()
    sub = sub[sub["country"].isin(counts[counts >= 2].index)]
    if sub["country"].nunique() < 3 or len(sub) < 10:
        return None

    # Demean both by country
    edu_dm = sub["edu_t"] - sub.groupby("country")["edu_t"].transform("mean")
    gdp_dm = sub["log_gdp_t"] - sub.groupby("country")["log_gdp_t"].transform("mean")

    X = edu_dm.values.reshape(-1, 1)
    y = gdp_dm.values
    ok = ~np.isnan(X.ravel()) & ~np.isnan(y)
    if ok.sum() < 10:
        return None

    reg = LinearRegression(fit_intercept=False).fit(X[ok], y[ok])
    predicted = reg.predict(X)
    sub["gdp_resid"] = gdp_dm.values - predicted.ravel()

    # R² of education → GDP (how much of GDP is explained by education)
    edu_gdp_r2 = reg.score(X[ok], y[ok])
    return sub, edu_gdp_r2


# ── Education levels ─────────────────────────────────────────────────

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
    print(f"# {level_name.upper().replace('_', ' ')}")
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
            tfr_tp25 = get_wb_val(tfr_raw, c, t + LAG)
            if np.isnan(edu_val) or np.isnan(tfr_tp25):
                continue
            rows.append({
                "country": c, "t": t, "edu_t": edu_val,
                "log_gdp_t": np.log(gdp_t) if not np.isnan(gdp_t) and gdp_t > 0 else np.nan,
                "tfr_tp25": tfr_tp25,
            })

    panel = pd.DataFrame(rows)

    # Precompute entry years
    entry_years = {}
    for threshold in range(10, 91):
        cohort = {}
        for c, s in edu_annual.items():
            entry = find_entry_year(s, threshold)
            if entry is not None:
                cohort[c] = entry
        entry_years[threshold] = cohort

    level_results = {}

    for ceiling in CEILINGS:
        print(f"\n  Ceiling = {ceiling}%")
        print(f"  {'Entry':<8} {'Edu R²':>7} {'GDP R²':>7} {'Resid R²':>9} {'Edu→GDP':>8} {'n':>5} {'Ctry':>5}")
        print(f"  {'-' * 60}")

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

            # Standard R²s
            r2_e, n_e, c_e = fe_r2("edu_t", "tfr_tp25", sub)
            r2_g, n_g, c_g = fe_r2("log_gdp_t", "tfr_tp25", sub)

            # Residualized GDP
            resid_result = fe_residualize_gdp(sub)
            if resid_result is not None:
                sub_resid, edu_gdp_r2 = resid_result
                r2_resid, n_resid, c_resid = fe_r2("gdp_resid", "tfr_tp25", sub_resid)
            else:
                r2_resid, edu_gdp_r2 = np.nan, np.nan

            if threshold % 10 == 0 or threshold == 10:
                def fmt(v): return f"{v:.3f}" if not np.isnan(v) else "n/a"
                print(f"  >= {threshold}%{'':<2} {fmt(r2_e):>7} {fmt(r2_g):>7} "
                      f"{fmt(r2_resid):>9} {fmt(edu_gdp_r2):>8} {n_e:>5} {c_e:>5}")

            ceil_results[str(threshold)] = {
                "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
                "raw_gdp_r2": round(r2_g, 3) if not np.isnan(r2_g) else None,
                "resid_gdp_r2": round(r2_resid, 3) if not np.isnan(r2_resid) else None,
                "edu_gdp_r2": round(edu_gdp_r2, 3) if not np.isnan(edu_gdp_r2) else None,
                "edu_n": n_e, "edu_countries": c_e,
            }

        level_results[str(ceiling)] = ceil_results

    all_results[level_name] = level_results

    # Summary
    print(f"\n  SUMMARY ({level_name}): entry=10%")
    print(f"  {'Ceiling':<10} {'Edu→TFR':>7} {'GDP→TFR':>7} {'Resid→TFR':>9} {'Edu→GDP':>8}")
    print(f"  {'-' * 45}")
    for ceiling in CEILINGS:
        r = level_results[str(ceiling)].get("10", {})
        if r and r.get("edu_r2") is not None:
            def fmt(v): return f"{v:.3f}" if v is not None else "n/a"
            print(f"  <= {ceiling}%{'':<4} {fmt(r.get('edu_r2')):>7} {fmt(r.get('raw_gdp_r2')):>7} "
                  f"{fmt(r.get('resid_gdp_r2')):>9} {fmt(r.get('edu_gdp_r2')):>8}")


# ── Cross-level comparison ───────────────────────────────────────────

print(f"\n\n{'=' * 90}")
print("CROSS-LEVEL COMPARISON: entry=10%, ceiling=60%")
print("Edu→TFR:    education predicts future fertility")
print("GDP→TFR:    raw GDP predicts future fertility")
print("Resid→TFR:  GDP AFTER removing education's effect on GDP → LE")
print("Edu→GDP:   how much of GDP is explained by education")
print(f"{'=' * 90}")
print(f"{'Level':<20} {'Edu→TFR':>7} {'GDP→TFR':>7} {'Resid→TFR':>9} {'Edu→GDP':>8}")
print("-" * 55)
for level_name in EDU_LEVELS:
    r = all_results[level_name].get("60", {}).get("10", {})
    if r and r.get("edu_r2") is not None:
        def fmt(v): return f"{v:.3f}" if v is not None else "n/a"
        print(f"{level_name:<20} {fmt(r.get('edu_r2')):>7} {fmt(r.get('raw_gdp_r2')):>7} "
              f"{fmt(r.get('resid_gdp_r2')):>9} {fmt(r.get('edu_gdp_r2')):>8}")

print(f"\n{'=' * 90}")
print("CROSS-LEVEL COMPARISON: entry=10%, ceiling=90%")
print(f"{'=' * 90}")
print(f"{'Level':<20} {'Edu→TFR':>7} {'GDP→TFR':>7} {'Resid→TFR':>9} {'Edu→GDP':>8}")
print("-" * 55)
for level_name in EDU_LEVELS:
    r = all_results[level_name].get("90", {}).get("10", {})
    if r and r.get("edu_r2") is not None:
        def fmt(v): return f"{v:.3f}" if v is not None else "n/a"
        print(f"{level_name:<20} {fmt(r.get('edu_r2')):>7} {fmt(r.get('raw_gdp_r2')):>7} "
              f"{fmt(r.get('resid_gdp_r2')):>9} {fmt(r.get('edu_gdp_r2')):>8}")

# ── Checkin ──────────────────────────────────────────────────────────

# Extract Fert-primary-R2: primary edu→TFR R² at entry=10%, ceiling=90%
_fert_primary_r2 = None
try:
    _fert_primary_r2 = all_results["primary"]["90"]["10"]["edu_r2"]
except (KeyError, TypeError):
    pass

checkin = {
    "script": "scripts/edu_vs_gdp_tfr_residualized.py",
    "produced": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "method": (
        "Country FE. Two-step residualization: (1) regress log_GDP on "
        "education with country FE to get GDP residuals (GDP not explained "
        "by education), (2) use residuals to predict LE(T+25). Compares "
        "education R², raw GDP R², and residualized GDP R². "
        "Entry-cohort with ceiling. T=1960-1990, lag=25. "
        "Three education levels: primary, lower secondary, upper secondary."
    ),
    "levels": all_results,
    "numbers": {
        "Fert-primary-R2": _fert_primary_r2,
    },
}

os.makedirs(CHECKIN, exist_ok=True)
out_path = os.path.join(CHECKIN, "edu_vs_gdp_tfr_residualized.json")
with open(out_path, "w") as f:
    json.dump(checkin, f, indent=2)
print(f"\nCheckin written to {out_path}")
