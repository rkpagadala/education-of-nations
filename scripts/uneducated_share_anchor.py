"""
Test whether the uneducated share of the adult population converges at
the development threshold crossing.

Hypothesis: development arrives when the uneducated share of the adult
population (ages 20-64) drops below a consistent threshold — i.e., when
enough uneducated cohorts have died off and been replaced by educated ones.

Method:
  1. From WCDE v3 raw data (pop_both.csv), extract population by age, year,
     education level for each country.
  2. For adults 20-64, compute the share WITHOUT lower secondary completion
     (= No Education + Incomplete Primary + Primary only).
  3. At each country's development crossing year, report the uneducated
     adult share and compare its CV across countries.

Data source:
  - wcde/data/raw/pop_both.csv (WCDE v3, population by age × education)
  - wcde/data/processed/lower_sec_both.csv (20-24 cohort completion, for comparison)

Development crossing dates (from paper):
  Taiwan 1972, South Korea 1987, Cuba 1974, Bangladesh 2014,
  Sri Lanka 1993, China 1994.
"""

import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(REPO_ROOT, "wcde", "data", "raw")
PROC_DIR = os.path.join(REPO_ROOT, "wcde", "data", "processed")

# ── Constants ─────────────────────────────────────────────────────
COUNTRIES = {
    "Taiwan Province of China":            {"label": "Taiwan",      "cross": 1972},
    "Republic of Korea":                   {"label": "South Korea", "cross": 1987},
    "Cuba":                                {"label": "Cuba",        "cross": 1974},
    "Bangladesh":                          {"label": "Bangladesh",  "cross": 2014},
    "Sri Lanka":                           {"label": "Sri Lanka",   "cross": 1993},
    "China":                               {"label": "China",       "cross": 1994},
}

# WCDE age bands for working-age adults (20-64)
ADULT_AGES = [
    "20--24", "25--29", "30--34", "35--39", "40--44",
    "45--49", "50--54", "55--59", "60--64",
]

# Education levels that count as "below lower secondary completion"
UNEDUCATED_LEVELS = ["No Education", "Incomplete Primary", "Primary"]

# All education levels for adults (excludes "Under 15")
ALL_ADULT_LEVELS = [
    "No Education", "Incomplete Primary", "Primary",
    "Lower Secondary", "Upper Secondary", "Post Secondary",
]

# ── Load raw WCDE population data ─────────────────────────────────
# Only load the countries we need to keep memory reasonable
print("Loading WCDE v3 raw population data (filtered)...")
target_names = set(COUNTRIES.keys())

chunks = []
for chunk in pd.read_csv(
    os.path.join(RAW_DIR, "pop_both.csv"),
    chunksize=200_000,
    dtype={"year": int, "pop": float},
):
    filtered = chunk[chunk["name"].isin(target_names)]
    if len(filtered) > 0:
        chunks.append(filtered)

raw = pd.concat(chunks, ignore_index=True)
print(f"  Loaded {len(raw):,} rows for {raw['name'].nunique()} countries")

# Strip quotes from string columns
for col in ["name", "age", "education"]:
    raw[col] = raw[col].str.strip('"').str.strip()

# ── Compute uneducated adult share by country and year ────────────
# Filter to adult ages only
adults = raw[raw["age"].isin(ADULT_AGES)].copy()
adults = adults[adults["education"].isin(ALL_ADULT_LEVELS)]

print(f"  Adult rows (20-64, education levels): {len(adults):,}")

# Total adult population by country × year
total_pop = adults.groupby(["name", "year"])["pop"].sum().rename("total_pop")

# Uneducated adult population by country × year
uneducated_pop = (
    adults[adults["education"].isin(UNEDUCATED_LEVELS)]
    .groupby(["name", "year"])["pop"]
    .sum()
    .rename("uneducated_pop")
)

# Merge and compute share
shares = pd.merge(total_pop, uneducated_pop, left_index=True, right_index=True, how="left")
shares["uneducated_pop"] = shares["uneducated_pop"].fillna(0)
shares["uneducated_pct"] = 100.0 * shares["uneducated_pop"] / shares["total_pop"]
shares["educated_pct"] = 100.0 - shares["uneducated_pct"]
shares = shares.reset_index()

# ── Load 20-24 cohort completion for comparison ───────────────────
cohort = pd.read_csv(os.path.join(PROC_DIR, "lower_sec_both.csv"), index_col="country")

# ── Interpolate to crossing years ─────────────────────────────────
# WCDE data is at 5-year intervals; crossing years may fall between them.
# Linear interpolation between bracketing WCDE years.

def interpolate_at_year(df, country, year, value_col):
    """Linearly interpolate a value at a non-WCDE year."""
    cdata = df[df["name"] == country].sort_values("year")
    years = cdata["year"].values
    vals = cdata[value_col].values

    if year in years:
        return vals[years == year][0]

    # Find bracketing years
    before = years[years <= year]
    after = years[years >= year]
    if len(before) == 0 or len(after) == 0:
        return np.nan

    y0, y1 = before[-1], after[0]
    v0 = vals[years == y0][0]
    v1 = vals[years == y1][0]

    # Linear interpolation
    frac = (year - y0) / (y1 - y0) if y1 != y0 else 0
    return v0 + frac * (v1 - v0)


def interpolate_cohort(cohort_df, country, year):
    """Interpolate 20-24 cohort completion from the wide-format CSV."""
    if country not in cohort_df.index:
        return np.nan
    cols = [c for c in cohort_df.columns if c.isdigit()]
    years_avail = np.array([int(c) for c in cols])
    vals = cohort_df.loc[country, cols].values.astype(float)

    if year in years_avail:
        idx = np.where(years_avail == year)[0][0]
        return vals[idx]

    before = years_avail[years_avail <= year]
    after = years_avail[years_avail >= year]
    if len(before) == 0 or len(after) == 0:
        return np.nan

    y0, y1 = before[-1], after[0]
    v0 = vals[years_avail == y0][0]
    v1 = vals[years_avail == y1][0]
    frac = (year - y0) / (y1 - y0) if y1 != y0 else 0
    return v0 + frac * (v1 - v0)


# ── Build results table ───────────────────────────────────────────
print("\n" + "=" * 78)
print("UNEDUCATED ADULT SHARE AT DEVELOPMENT THRESHOLD CROSSING")
print("=" * 78)
print(f"{'Country':<15} {'Cross':>5}  {'Cohort 20-24':>12}  {'Adult Educ%':>11}"
      f"  {'Uneducated%':>11}  {'Adult Pop':>10}")
print("-" * 78)

results = []
for wcde_name, info in COUNTRIES.items():
    label = info["label"]
    cross = info["cross"]

    uneducated_pct = interpolate_at_year(shares, wcde_name, cross, "uneducated_pct")
    educated_pct = interpolate_at_year(shares, wcde_name, cross, "educated_pct")
    total = interpolate_at_year(shares, wcde_name, cross, "total_pop")
    cohort_pct = interpolate_cohort(cohort, wcde_name, cross)

    results.append({
        "country": label,
        "crossing": cross,
        "cohort_20_24": cohort_pct,
        "adult_educated_pct": educated_pct,
        "uneducated_pct": uneducated_pct,
        "total_adult_pop": total,
    })

    print(f"{label:<15} {cross:>5}  {cohort_pct:>11.1f}%  {educated_pct:>10.1f}%"
          f"  {uneducated_pct:>10.1f}%  {total:>10,.0f}k")

# ── Summary statistics ────────────────────────────────────────────
df = pd.DataFrame(results)

uneducated_vals = df["uneducated_pct"].values
cohort_vals = df["cohort_20_24"].values

mean_uneducated = np.mean(uneducated_vals)
std_uneducated = np.std(uneducated_vals, ddof=1)
cv_uneducated = std_uneducated / mean_uneducated

mean_cohort = np.mean(cohort_vals)
std_cohort = np.std(cohort_vals, ddof=1)
cv_cohort = std_cohort / mean_cohort

print("\n" + "=" * 78)
print("SUMMARY STATISTICS")
print("=" * 78)

print(f"\nUneducated adult share at crossing (ages 20-64):")
print(f"  Mean:  {mean_uneducated:.1f}%")
print(f"  Std:   {std_uneducated:.1f}pp")
print(f"  CV:    {cv_uneducated:.2f}")
print(f"  Range: {np.min(uneducated_vals):.1f}% – {np.max(uneducated_vals):.1f}%")

print(f"\n20-24 cohort completion at crossing (for comparison):")
print(f"  Mean:  {mean_cohort:.1f}%")
print(f"  Std:   {std_cohort:.1f}pp")
print(f"  CV:    {cv_cohort:.2f}")
print(f"  Range: {np.min(cohort_vals):.1f}% – {np.max(cohort_vals):.1f}%")

print(f"\nComparison of CVs:")
print(f"  Uneducated adult share:  CV = {cv_uneducated:.2f}")
print(f"  20-24 cohort completion: CV = {cv_cohort:.2f}")
print(f"  Education integral:      CV = 0.34  (from prior analysis)")

if cv_uneducated < cv_cohort:
    print(f"\n  → Uneducated share is TIGHTER (lower CV) than cohort completion.")
    print(f"    Consistent with: development arrives when the uneducated adult")
    print(f"    population drops below a threshold.")
else:
    print(f"\n  → Uneducated share is LOOSER (higher CV) than cohort completion.")
    print(f"    The cohort-leading-edge signal is stronger than the stock depletion signal.")

# ── Time series for each country ──────────────────────────────────
print("\n" + "=" * 78)
print("TIME SERIES: Uneducated adult share (%) by country")
print("=" * 78)

wcde_years = sorted(shares["year"].unique())
print(f"\n{'Country':<15}", end="")
for y in wcde_years:
    if 1950 <= y <= 2020:
        print(f" {y:>6}", end="")
print()
print("-" * 78)

for wcde_name, info in COUNTRIES.items():
    label = info["label"]
    cdata = shares[shares["name"] == wcde_name].set_index("year")
    print(f"{label:<15}", end="")
    for y in wcde_years:
        if 1950 <= y <= 2020:
            if y in cdata.index:
                print(f" {cdata.loc[y, 'uneducated_pct']:>5.1f}%", end="")
            else:
                print(f"    --", end="")
    print(f"   ← crosses at {info['cross']}")
