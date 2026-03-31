"""
le_floor_test.py — Test whether 50% lower secondary completion is a necessary
condition (floor) for life expectancy > 69.8.

For every country in the dataset:
  1. Find the year it first crossed LE > 69.8
  2. Find lower secondary completion (age 20-24, both sexes) at that year
  3. Check whether any country crossed LE > 69.8 with completion below 50%

Data sources:
  - Education: wcde/data/processed/lower_sec_both.csv (187 countries, 5-year intervals 1950-2025)
  - Life expectancy: data/life_expectancy_years.csv (annual 1960-2022)
"""

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
EDU_PATH = "wcde/data/processed/lower_sec_both.csv"
LE_PATH = "data/life_expectancy_years.csv"

LE_THRESHOLD = 69.8

OIL_STATES = ["Qatar", "United Arab Emirates", "Kuwait", "Saudi Arabia", "Oman", "Bahrain"]

# ── Load education data ────────────────────────────────────────────────────
edu_raw = pd.read_csv(EDU_PATH)
edu_raw = edu_raw.rename(columns={edu_raw.columns[0]: "country"})

# Melt to long format: country, year, completion
edu_years = [c for c in edu_raw.columns if c != "country"]
edu = edu_raw.melt(id_vars="country", value_vars=edu_years, var_name="year", value_name="completion")
edu["year"] = edu["year"].astype(int)
edu = edu.dropna(subset=["completion"])
edu = edu.sort_values(["country", "year"])

# ── Load life expectancy data ──────────────────────────────────────────────
le_raw = pd.read_csv(LE_PATH)
le_raw = le_raw.rename(columns={le_raw.columns[0]: "country"})
# Clean country names (some have quotes from CSV)
le_raw["country"] = le_raw["country"].str.strip().str.strip('"')

le_years = [c for c in le_raw.columns if c != "country"]
le = le_raw.melt(id_vars="country", value_vars=le_years, var_name="year", value_name="le")
le["year"] = le["year"].astype(int)
le["le"] = pd.to_numeric(le["le"], errors="coerce")
le = le.dropna(subset=["le"])
le = le.sort_values(["country", "year"])

# ── Country name mapping (education -> LE file) ───────────────────────────
# Build mapping for names that differ between files
NAME_MAP_EDU_TO_LE = {
    "Bolivia (Plurinational State of)": "Bolivia",
    "Brunei Darussalam": "Brunei Darussalam",
    "Cape Verde": "Cabo Verde",
    "China": "China",
    "Congo": "Congo, Rep.",  # try both
    "Cote d'Ivoire": "Cote d'Ivoire",
    "Czech Republic": "Czechia",
    "Democratic People's Republic of Korea": "Korea, Dem. People's Rep.",
    "Democratic Republic of the Congo": "Congo, Dem. Rep.",
    "Hong Kong Special Administrative Region of China": "Hong Kong SAR, China",
    "Iran (Islamic Republic of)": "Iran, Islamic Rep.",
    "Lao People's Democratic Republic": "Lao PDR",
    "Libyan Arab Jamahiriya": "Libya",
    "Macao Special Administrative Region of China": "Macao SAR, China",
    "Micronesia (Federated States of)": "Micronesia, Fed. Sts.",
    "Occupied Palestinian Territory": "West Bank and Gaza",
    "Republic of Korea": "Korea, Rep.",
    "Republic of Moldova": "Moldova",
    "Russian Federation": "Russian Federation",
    "Saint Lucia": "St. Lucia",
    "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
    "Swaziland": "Eswatini",
    "Syrian Arab Republic": "Syrian Arab Republic",
    "Taiwan Province of China": "Taiwan Province of China",  # may not be in WB
    "The former Yugoslav Republic of Macedonia": "North Macedonia",
    "Turkey": "Turkiye",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "United Republic of Tanzania": "Tanzania",
    "United States of America": "United States",
    "United States Virgin Islands": "Virgin Islands (U.S.)",
    "Venezuela (Bolivarian Republic of)": "Venezuela, RB",
    "Viet Nam": "Viet Nam",
    "Yemen": "Yemen, Rep.",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Slovakia": "Slovak Republic",
    "Curaçao": "Curacao",
    "Sao Tome and Principe": "Sao Tome and Principe",
    "Gambia": "Gambia, The",
    "Egypt": "Egypt, Arab Rep.",
    "Somalia": "Somalia",
    "Bahamas": "Bahamas, The",
    "Reunion": "Reunion",
    "French Guiana": "French Guiana",
    "Martinique": "Martinique",
    "Guadeloupe": "Guadeloupe",
    "Mayotte": "Mayotte",
    "Western Sahara": "Western Sahara",
    "South Sudan": "South Sudan",
    "Puerto Rico": "Puerto Rico (US)",
}

# Get unique country names in both datasets
edu_countries = set(edu["country"].unique())
le_countries = set(le["country"].unique())

# Filter out region aggregates from education data
REGION_KEYWORDS = [
    "Africa", "Asia", "Europe", "America", "Caribbean", "Melanesia",
    "Micronesia", "Polynesia", "World", "Eastern", "Western", "Northern",
    "Southern", "Central", "South-Eastern", "Latin America",
    "Australia and New Zealand",
]

def is_region(name):
    """Check if a country name is actually a region aggregate."""
    # Specific country exceptions that contain region keywords
    exceptions = {
        "South Africa", "Central African Republic", "South Sudan",
        "Equatorial Guinea", "Western Sahara", "American Samoa",
    }
    if name in exceptions:
        return False
    for kw in REGION_KEYWORDS:
        if name == kw or (kw in name and name not in exceptions):
            # Be more careful: only flag exact matches or clear aggregates
            if name in ["Africa", "Asia", "Europe", "World", "Caribbean",
                        "Melanesia", "Polynesia", "Oceania",
                        "Central America", "Central Asia",
                        "Eastern Africa", "Eastern Asia", "Eastern Europe",
                        "Middle Africa", "Northern Africa", "Northern America",
                        "Northern Europe", "South America",
                        "South-Eastern Asia", "Southern Africa",
                        "Southern Asia", "Southern Europe",
                        "Western Africa", "Western Asia", "Western Europe",
                        "Latin America and the Caribbean",
                        "Australia and New Zealand"]:
                return True
    return False

edu_country_list = [c for c in edu["country"].unique() if not is_region(c)]

# ── Match countries ────────────────────────────────────────────────────────
def find_le_name(edu_name):
    """Map education country name to LE country name."""
    if edu_name in NAME_MAP_EDU_TO_LE:
        mapped = NAME_MAP_EDU_TO_LE[edu_name]
        if mapped in le_countries:
            return mapped
    if edu_name in le_countries:
        return edu_name
    return None

matched = {}
unmatched = []
for c in edu_country_list:
    le_name = find_le_name(c)
    if le_name:
        matched[c] = le_name
    else:
        unmatched.append(c)

print(f"Matched: {len(matched)} countries")
print(f"Unmatched: {len(unmatched)} countries")
if unmatched:
    print(f"  Unmatched: {unmatched}")

# ── Interpolate education completion for any year ──────────────────────────
def get_completion_at_year(edu_df, country, year):
    """Interpolate lower secondary completion for a country at a given year."""
    cdata = edu_df[edu_df["country"] == country].sort_values("year")
    if cdata.empty:
        return np.nan

    years = cdata["year"].values
    vals = cdata["completion"].values

    if year <= years[0]:
        return vals[0]
    if year >= years[-1]:
        return vals[-1]

    # Linear interpolation between bracketing 5-year points
    for i in range(len(years) - 1):
        if years[i] <= year <= years[i + 1]:
            frac = (year - years[i]) / (years[i + 1] - years[i])
            return vals[i] + frac * (vals[i + 1] - vals[i])
    return np.nan

# ── Find first year LE crossed threshold for each country ──────────────────
results = []
not_crossed = []

for edu_name, le_name in sorted(matched.items()):
    cle = le[le["country"] == le_name].sort_values("year")
    if cle.empty:
        continue

    # Find first year LE > threshold
    crossed = cle[cle["le"] > LE_THRESHOLD]
    if crossed.empty:
        # Record as not-crossed
        latest_year = cle["year"].max()
        latest_le = cle[cle["year"] == latest_year]["le"].values[0]
        latest_comp = get_completion_at_year(edu, edu_name, min(latest_year, 2020))
        not_crossed.append({
            "country": edu_name,
            "latest_year": latest_year,
            "latest_le": latest_le,
            "completion": latest_comp,
        })
        continue

    first_year = crossed["year"].min()
    first_le = crossed[crossed["year"] == first_year]["le"].values[0]
    comp = get_completion_at_year(edu, edu_name, first_year)

    is_oil = edu_name in OIL_STATES or le_name in OIL_STATES
    results.append({
        "country": edu_name,
        "le_name": le_name,
        "first_year_above": first_year,
        "le_at_crossing": first_le,
        "completion_at_crossing": comp,
        "oil_state": is_oil,
    })

df = pd.DataFrame(results).sort_values("completion_at_crossing")
df_not = pd.DataFrame(not_crossed).sort_values("completion")

# ── Output ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print(f"TEST: Is 50% lower secondary completion a necessary floor for LE > {LE_THRESHOLD}?")
print("=" * 90)

print(f"\n{len(df)} countries crossed LE > {LE_THRESHOLD}")
print(f"{len(df_not)} matched countries have NOT crossed LE > {LE_THRESHOLD}\n")

# Full table
print("-" * 90)
print(f"{'Country':<45} {'Year':>6} {'LE':>7} {'Completion%':>12} {'Oil?':>5}")
print("-" * 90)
for _, r in df.iterrows():
    oil_flag = " OIL" if r["oil_state"] else ""
    print(f"{r['country']:<45} {r['first_year_above']:>6.0f} {r['le_at_crossing']:>7.1f} "
          f"{r['completion_at_crossing']:>11.1f}%{oil_flag:>5}")

# Bottom 10
print("\n" + "=" * 90)
print("10 LOWEST COMPLETION AT LE CROSSING (testing the floor)")
print("=" * 90)
bottom10 = df.head(10)
for _, r in bottom10.iterrows():
    oil_flag = " ** OIL STATE **" if r["oil_state"] else ""
    print(f"  {r['country']:<42} {r['first_year_above']:>6.0f}  LE={r['le_at_crossing']:.1f}  "
          f"Completion={r['completion_at_crossing']:.1f}%{oil_flag}")

# Counts below thresholds
print("\n" + "=" * 90)
print("FLOOR TEST COUNTS")
print("=" * 90)
for thresh in [50, 40, 30, 20, 10]:
    n = (df["completion_at_crossing"] < thresh).sum()
    countries_below = df[df["completion_at_crossing"] < thresh]["country"].tolist()
    print(f"  Crossed LE > {LE_THRESHOLD} with completion < {thresh}%: {n} countries")
    if countries_below and n <= 10:
        for c in countries_below:
            row = df[df["country"] == c].iloc[0]
            print(f"    - {c}: {row['completion_at_crossing']:.1f}% (year {row['first_year_above']:.0f})")

# Countries that haven't crossed
print("\n" + "=" * 90)
print(f"COUNTRIES THAT HAVE NOT CROSSED LE > {LE_THRESHOLD} (as of latest data)")
print("=" * 90)
print(f"{'Country':<45} {'Latest Year':>11} {'LE':>7} {'Completion%':>12}")
print("-" * 90)
for _, r in df_not.iterrows():
    print(f"{r['country']:<45} {r['latest_year']:>11.0f} {r['latest_le']:>7.1f} "
          f"{r['completion']:>11.1f}%")

above_50_not_crossed = df_not[df_not["completion"] >= 50]
below_50_not_crossed = df_not[df_not["completion"] < 50]
print(f"\n  Not crossed + completion >= 50%: {len(above_50_not_crossed)} countries")
if not above_50_not_crossed.empty:
    for _, r in above_50_not_crossed.iterrows():
        print(f"    - {r['country']}: {r['completion']:.1f}% (LE={r['latest_le']:.1f})")
print(f"  Not crossed + completion < 50%: {len(below_50_not_crossed)} countries")

# Oil state analysis
print("\n" + "=" * 90)
print("OIL STATE ANALYSIS")
print("=" * 90)
oil_results = df[df["oil_state"]]
if oil_results.empty:
    print("  No oil states found in matched data.")
else:
    for _, r in oil_results.iterrows():
        print(f"  {r['country']:<30} Year={r['first_year_above']:.0f}  LE={r['le_at_crossing']:.1f}  "
              f"Completion={r['completion_at_crossing']:.1f}%")
    below_50_oil = oil_results[oil_results["completion_at_crossing"] < 50]
    print(f"\n  Oil states that crossed LE > {LE_THRESHOLD} with completion < 50%: {len(below_50_oil)}")
    if not below_50_oil.empty:
        for _, r in below_50_oil.iterrows():
            print(f"    ** COUNTEREXAMPLE: {r['country']} — {r['completion_at_crossing']:.1f}% completion at LE crossing")

# Summary verdict
print("\n" + "=" * 90)
print("VERDICT")
print("=" * 90)
below_50_count = (df["completion_at_crossing"] < 50).sum()
min_comp = df["completion_at_crossing"].min()
min_country = df.loc[df["completion_at_crossing"].idxmin(), "country"]
median_comp = df["completion_at_crossing"].median()
print(f"  Countries that crossed LE > {LE_THRESHOLD}: {len(df)}")
print(f"  Crossed with completion < 50%: {below_50_count}")
print(f"  Lowest completion at crossing: {min_comp:.1f}% ({min_country})")
print(f"  Median completion at crossing: {median_comp:.1f}%")
if below_50_count == 0:
    print(f"\n  ==> 50% lower secondary completion is a NECESSARY CONDITION for LE > {LE_THRESHOLD}")
    print(f"      Zero countries in the dataset crossed LE > {LE_THRESHOLD} without it.")
else:
    print(f"\n  ==> 50% is NOT a strict necessary condition: {below_50_count} countries crossed below 50%.")
    print(f"      Examine whether these are oil states or data artefacts.")
