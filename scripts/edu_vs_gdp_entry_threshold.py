"""
edu_vs_gdp_entry_threshold.py
==============================
Does education or GDP predict future life expectancy?

Same question as edu_vs_gdp_predicts_le.py but with a better sample
definition: instead of "observations where education < X%", we define
entry cohorts — countries that first crossed a given education threshold,
tracked from that point onward through their full development trajectory.

METHOD
------
1. Interpolate WCDE 5-year education data to annual values (linear).
2. For each entry threshold (10%, 11%, ..., 90%), find the first year
   each country crosses that threshold.
3. Build a panel: for each country in the cohort, include all T-year
   observations from the entry year onward (T = 1960..1990, 5-year steps).
4. Run country fixed-effects regression:
   - education(T) → life expectancy(T+25)
   - log GDP(T) → life expectancy(T+25)
5. Report within-R² at each threshold.

This gives more observations per country (full trajectory after entry)
and a cleaner sample definition ("countries that started from X% or below").

DATA
----
Same as edu_vs_gdp_predicts_le.py:
- Education: WCDE v3, lower secondary completion, both sexes, age 20-24
- Life expectancy: World Bank WDI (SP.DYN.LE00.IN)
- GDP: World Bank WDI, constant 2017 USD (NY.GDP.PCAP.KD), log-transformed

OUTPUT
------
Table of R² values at each 1% threshold, plus JSON checkin file.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ── Paths ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROC = os.path.join(REPO_ROOT, "wcde", "data", "processed")
DATA = os.path.join(REPO_ROOT, "data")
CHECKIN = os.path.join(REPO_ROOT, "checkin")

# ── Regional aggregates to exclude ───────────────────────────────────

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

# ── Country name mapping (WCDE → World Bank) ────────────────────────

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

# Education: WCDE v3, lower secondary completion, both sexes
edu = pd.read_csv(os.path.join(PROC, "completion_both_long.csv"))
edu = edu[~edu["country"].isin(REGIONS)].copy()
print(f"  Education: {edu['country'].nunique()} countries, "
      f"years {edu['year'].min()}-{edu['year'].max()}")

# Life expectancy: World Bank WDI
le_raw = pd.read_csv(os.path.join(DATA, "life_expectancy_years.csv"))
le_raw["Country"] = le_raw["Country"].str.lower()
le_raw = le_raw.set_index("Country")
for c in le_raw.columns:
    le_raw[c] = pd.to_numeric(le_raw[c], errors="coerce")
print(f"  Life expectancy: {len(le_raw)} countries")

# GDP per capita: World Bank WDI, constant 2017 USD
gdp_raw = pd.read_csv(os.path.join(DATA, "gdppercapita_us_inflation_adjusted.csv"))
gdp_raw["Country"] = gdp_raw["Country"].str.lower()
gdp_raw = gdp_raw.set_index("Country")
for c in gdp_raw.columns:
    gdp_raw[c] = pd.to_numeric(gdp_raw[c], errors="coerce")
print(f"  GDP: {len(gdp_raw)} countries")


def get_wb_val(df, wcde_name, year):
    """Look up a World Bank value for a WCDE country name."""
    key = NAME_MAP.get(wcde_name, wcde_name).lower()
    for k in [wcde_name.lower(), key]:
        if k in df.index:
            try:
                v = float(df.loc[k, str(year)])
                return v if not np.isnan(v) else np.nan
            except (KeyError, ValueError):
                pass
    return np.nan


# ── Interpolate education to annual values ───────────────────────────
# WCDE is at 5-year intervals. Interpolate linearly to get annual
# values so we can find the precise year each country crosses a threshold.

print("\nInterpolating education to annual values...")

edu_annual = {}  # {country: pd.Series indexed by year}
for c, grp in edu.groupby("country"):
    s = grp.set_index("year")["lower_sec"].sort_index()
    # Reindex to annual, interpolate linearly
    full_idx = range(s.index.min(), s.index.max() + 1)
    s_annual = s.reindex(full_idx).interpolate(method="linear")
    edu_annual[c] = s_annual

print(f"  Interpolated {len(edu_annual)} countries")


# ── Find entry year for each country at each threshold ───────────────

def find_entry_year(series, threshold):
    """Find the first year where education >= threshold."""
    above = series[series >= threshold]
    if len(above) == 0:
        return None
    return above.index[0]


# ── Build the full panel (all T-years, all countries) ────────────────
# We build once, then filter per threshold.

T_YEARS = list(range(1960, 1995, 5))  # 1960, 1965, 1970, 1975, 1980, 1985, 1990
LAG = 25

print(f"\nBuilding full panel (T={T_YEARS[0]}-{T_YEARS[-1]}, lag={LAG})...")

rows = []
for c in sorted(edu_annual.keys()):
    s = edu_annual[c]
    for t in T_YEARS:
        if t not in s.index:
            continue
        tp25 = t + LAG
        edu_val = s[t]
        gdp_t = get_wb_val(gdp_raw, c, t)
        le_tp25 = get_wb_val(le_raw, c, tp25)
        if np.isnan(edu_val) or np.isnan(le_tp25):
            continue
        rows.append({
            "country": c,
            "t": t,
            "edu_t": edu_val,
            "log_gdp_t": np.log(gdp_t) if not np.isnan(gdp_t) and gdp_t > 0 else np.nan,
            "le_tp25": le_tp25,
        })

panel = pd.DataFrame(rows)
print(f"  Full panel: {len(panel)} obs, {panel['country'].nunique()} countries")


# ── Country fixed effects regression ─────────────────────────────────

def fe_r2(x_col, y_col, data):
    """
    Country-FE regression. Returns (r2, n_obs, n_countries).
    Demean by country, regress demeaned y on demeaned x (no intercept).
    """
    sub = data.dropna(subset=[x_col, y_col]).copy()
    # Need at least 2 observations per country for FE to make sense,
    # and at least 3 countries
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
    r2 = reg.score(X[ok], y[ok])
    return r2, int(ok.sum()), n_countries


# ── Run at every 1% threshold from 10% to 90% ───────────────────────

print("\n" + "=" * 90)
print("Entry-Cohort Analysis: Education(T) vs GDP(T) → Life Expectancy(T+25)")
print("Country fixed effects | T = 1960-1990 (5yr) | Lag = 25 years")
print("Entry = first year country crosses threshold; all observations from then on")
print("=" * 90)
print(f"{'Threshold':<12} {'Edu R²':>8} {'n':>6} {'Ctry':>6}   "
      f"{'GDP R²':>8} {'n':>6} {'Ctry':>6}   {'Ratio':>8}")
print("-" * 90)

results = {}

for threshold in range(10, 91):
    # Find which countries ever crossed this threshold, and when
    cohort = {}  # {country: entry_year}
    for c, s in edu_annual.items():
        entry = find_entry_year(s, threshold)
        if entry is not None:
            cohort[c] = entry

    # Filter panel: only countries in cohort, only T >= entry_year
    if len(cohort) < 3:
        continue

    mask = panel.apply(
        lambda r: r["country"] in cohort and r["t"] >= cohort[r["country"]],
        axis=1
    )
    sub = panel[mask]

    if len(sub) < 10:
        continue

    r2_e, n_e, c_e = fe_r2("edu_t", "le_tp25", sub)
    r2_g, n_g, c_g = fe_r2("log_gdp_t", "le_tp25", sub)

    if not np.isnan(r2_g) and r2_g > 0.001:
        ratio = f"{r2_e / r2_g:.1f}x"
    elif not np.isnan(r2_e):
        ratio = "GDP≈0"
    else:
        ratio = "n/a"

    r2_e_s = f"{r2_e:.3f}" if not np.isnan(r2_e) else "n/a"
    r2_g_s = f"{r2_g:.3f}" if not np.isnan(r2_g) else "n/a"

    # Print every 5% to keep output readable, but store all
    if threshold % 5 == 0 or threshold == 10:
        print(f"  >= {threshold}%{'':<6} {r2_e_s:>8} {n_e:>6} {c_e:>6}   "
              f"{r2_g_s:>8} {n_g:>6} {c_g:>6}   {ratio:>8}")

    results[str(threshold)] = {
        "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
        "edu_n": n_e,
        "edu_countries": c_e,
        "gdp_r2": round(r2_g, 3) if not np.isnan(r2_g) else None,
        "gdp_n": n_g,
        "gdp_countries": c_g,
    }

# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

# Find the range of edu R² and GDP R²
edu_r2s = [v["edu_r2"] for v in results.values() if v["edu_r2"] is not None]
gdp_r2s = [v["gdp_r2"] for v in results.values() if v["gdp_r2"] is not None]

if edu_r2s and gdp_r2s:
    print(f"Education R² range: {min(edu_r2s):.3f} - {max(edu_r2s):.3f}")
    print(f"GDP R² range:       {min(gdp_r2s):.3f} - {max(gdp_r2s):.3f}")
    print(f"Education consistently explains {min(edu_r2s)*100:.0f}-{max(edu_r2s)*100:.0f}% "
          f"of within-country variation in future life expectancy.")
    print(f"GDP consistently explains {min(gdp_r2s)*100:.0f}-{max(gdp_r2s)*100:.0f}%.")

# ── Write checkin file ───────────────────────────────────────────────

checkin = {
    "script": "scripts/edu_vs_gdp_entry_threshold.py",
    "produced": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "method": (
        "Country FE (demeaned). Entry-cohort design: for each threshold "
        "(10-90%), find first year each country crosses it, then include "
        "all T-year observations from entry onward. "
        "T = 1960/1965/.../1990, outcome = LE(T+25). "
        "Education interpolated from WCDE 5-year to annual (linear)."
    ),
    "education_variable": "lower secondary completion, both sexes, age 20-24 (WCDE v3)",
    "gdp_variable": "log GDP per capita, constant 2017 USD (World Bank WDI)",
    "le_variable": "life expectancy at birth (World Bank WDI)",
    "thresholds": results,
}

os.makedirs(CHECKIN, exist_ok=True)
out_path = os.path.join(CHECKIN, "edu_vs_gdp_entry_threshold.json")
with open(out_path, "w") as f:
    json.dump(checkin, f, indent=2)
print(f"\nCheckin written to {out_path}")
