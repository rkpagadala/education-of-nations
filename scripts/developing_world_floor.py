"""
developing_world_floor.py
─────────────────────────
Test the 50% lower-secondary floor for LE > 69.8 crossing,
restricted to developing-world countries OUTSIDE the Western/European
knowledge network.

Exclusions:
  - All of Europe (Western, Eastern, Southern, Northern)
  - European settler states (USA, Canada, Australia, NZ, Argentina, Uruguay, Chile)
  - Oil states (Qatar, UAE, Kuwait, Saudi Arabia, Oman, Bahrain)
  - Small island states (<1M population)
  - Israel, Japan

Kept:
  - All of Asia (except Japan, Israel, oil states)
  - All of Africa
  - Central America, Caribbean, South America (except Arg, Uru, Chile)
  - Large island states (Cuba, Jamaica, Trinidad, etc.)
"""

import pandas as pd
import numpy as np

# ── paths ────────────────────────────────────────────────────────────
EDU_PATH = "wcde/data/processed/lower_sec_both.csv"
LE_PATH  = "data/life_expectancy_years.csv"

LE_THRESHOLD = 69.8

# ── exclusion lists ──────────────────────────────────────────────────

EUROPE = {
    # Western
    "France", "Germany", "Netherlands", "Belgium", "Luxembourg", "Austria",
    "Switzerland", "Liechtenstein", "Monaco",
    # Northern
    "United Kingdom", "Ireland", "Denmark", "Sweden", "Norway", "Finland",
    "Iceland",
    # Southern
    "Italy", "Spain", "Portugal", "Greece", "Malta", "Cyprus", "Andorra",
    "San Marino", "Vatican City", "Holy See",
    # Eastern
    "Poland", "Czechia", "Czech Republic", "Slovakia", "Hungary", "Romania",
    "Bulgaria", "Serbia", "Croatia", "Slovenia", "Bosnia and Herzegovina",
    "North Macedonia", "Montenegro", "Albania", "Kosovo",
    "Moldova", "Ukraine", "Belarus", "Russia", "Russian Federation",
    "Lithuania", "Latvia", "Estonia", "Georgia", "Armenia",
    # Transcontinental but European-network
    "Turkey",
}

SETTLER_STATES = {
    "United States", "United States of America", "USA",
    "Canada", "Australia", "New Zealand",
    "Argentina", "Uruguay", "Chile",
}

OIL_STATES = {
    "Qatar", "United Arab Emirates", "Kuwait",
    "Saudi Arabia", "Oman", "Bahrain",
}

SMALL_ISLANDS = {
    "Malta", "Maldives", "Cape Verde", "Cabo Verde",
    "Saint Lucia", "St. Lucia", "St Lucia",
    "Saint Vincent and the Grenadines", "St. Vincent and the Grenadines",
    "Belize", "Vanuatu", "Samoa", "Tonga", "Kiribati",
    "Micronesia", "Micronesia (Fed. States of)", "Federated States of Micronesia",
    "Marshall Islands", "Palau", "Nauru", "Tuvalu",
    "Sao Tome and Principe", "São Tomé and Príncipe",
    "Comoros", "Seychelles", "Antigua and Barbuda",
    "Dominica", "Grenada", "Saint Kitts and Nevis",
    "Barbados", "Bahamas", "Brunei", "Brunei Darussalam",
    "Suriname", "Guyana", "Iceland",
    "Solomon Islands", "Fiji", "Timor-Leste", "East Timor",
    "Mauritius",  # ~1.3M but often classed small-island; keep if you prefer
    "Equatorial Guinea",
    "Luxembourg", "Djibouti", "Montenegro",
}

OTHER_EXCLUDE = {"Israel", "Japan"}

# Aggregates / regions that are not countries
AGGREGATES = {
    "World", "Africa", "Asia", "Europe", "Latin America and the Caribbean",
    "Northern America", "Oceania", "Africa Eastern and Southern",
    "Africa Western and Central", "Arab World", "Caribbean small states",
    "Central Europe and the Baltics", "Channel Islands",
    "Early-demographic dividend", "East Asia & Pacific",
    "East Asia & Pacific (excluding high income)",
    "Euro area", "European Union", "Fragile and conflict affected situations",
    "Heavily indebted poor countries (HIPC)",
    "High income", "IDA & IBRD total", "IDA blend", "IDA only", "IDA total",
    "Late-demographic dividend", "Latin America & Caribbean",
    "Latin America & Caribbean (excluding high income)",
    "Least developed countries: UN classification",
    "Low & middle income", "Low income", "Lower middle income",
    "Middle East & North Africa",
    "Middle East & North Africa (excluding high income)",
    "Middle income", "North America",
    "Not classified", "OECD members",
    "Other small states", "Pacific island small states",
    "Post-demographic dividend", "Pre-demographic dividend",
    "Small states", "South Asia", "Sub-Saharan Africa",
    "Sub-Saharan Africa (excluding high income)",
    "Upper middle income",
    "More developed regions", "Less developed regions",
    "Least developed countries",
}

ALL_EXCLUDE = EUROPE | SETTLER_STATES | OIL_STATES | SMALL_ISLANDS | OTHER_EXCLUDE | AGGREGATES

# ── load data ────────────────────────────────────────────────────────

edu = pd.read_csv(EDU_PATH)
le  = pd.read_csv(LE_PATH)

# Normalise country column name
edu = edu.rename(columns={edu.columns[0]: "country"})
le  = le.rename(columns={le.columns[0]: "country"})

# Strip whitespace
edu["country"] = edu["country"].str.strip()
le["country"]  = le["country"].str.strip()

# ── filter to developing world ───────────────────────────────────────

def is_kept(c):
    return c not in ALL_EXCLUDE

edu_countries = set(edu["country"])
le_countries  = set(le["country"])
common = edu_countries & le_countries

kept = sorted([c for c in common if is_kept(c)])
print(f"Countries in both datasets: {len(common)}")
print(f"After exclusions (developing world): {len(kept)}")
print()

# ── melt to long form ───────────────────────────────────────────────

edu_long = edu.melt(id_vars="country", var_name="year", value_name="lower_sec_pct")
edu_long["year"] = edu_long["year"].astype(int)

le_long = le.melt(id_vars="country", var_name="year", value_name="le")
le_long["year"] = le_long["year"].astype(int)

# ── interpolate education to annual (linear between 5-yr points) ─────

edu_annual_rows = []
for c in kept:
    sub = edu_long[edu_long["country"] == c].sort_values("year")
    if sub.empty:
        continue
    # Create annual index and interpolate
    yrs = range(sub["year"].min(), sub["year"].max() + 1)
    annual = pd.DataFrame({"year": list(yrs)})
    annual = annual.merge(sub[["year", "lower_sec_pct"]], on="year", how="left")
    annual["lower_sec_pct"] = annual["lower_sec_pct"].interpolate(method="linear")
    annual["country"] = c
    edu_annual_rows.append(annual)

edu_annual = pd.concat(edu_annual_rows, ignore_index=True)

# ── merge education + LE ─────────────────────────────────────────────

merged = le_long[le_long["country"].isin(kept)].merge(
    edu_annual, on=["country", "year"], how="inner"
)

# ── find crossing year for each country ──────────────────────────────

crossers = []
non_crossers = []

for c in kept:
    sub = merged[merged["country"] == c].sort_values("year")
    if sub.empty:
        continue
    above = sub[sub["le"] > LE_THRESHOLD]
    if above.empty:
        # non-crosser: latest data
        latest = sub.iloc[-1]
        non_crossers.append({
            "country": c,
            "latest_year": int(latest["year"]),
            "latest_le": round(latest["le"], 1),
            "latest_lower_sec": round(latest["lower_sec_pct"], 1),
        })
    else:
        row = above.iloc[0]  # first year above threshold
        crossers.append({
            "country": c,
            "crossing_year": int(row["year"]),
            "le_at_crossing": round(row["le"], 1),
            "lower_sec_pct": round(row["lower_sec_pct"], 1),
        })

crossers_df = pd.DataFrame(crossers).sort_values("lower_sec_pct")
non_crossers_df = pd.DataFrame(non_crossers).sort_values("latest_lower_sec", ascending=False)

# ── report ───────────────────────────────────────────────────────────

print("=" * 72)
print(f"DEVELOPING-WORLD LE > {LE_THRESHOLD} FLOOR TEST")
print(f"Lower-secondary completion (age 20–24, both sexes) at crossing year")
print("=" * 72)

print(f"\nTotal crossers: {len(crossers_df)}")
print(f"Total non-crossers: {len(non_crossers_df)}")

below_50 = crossers_df[crossers_df["lower_sec_pct"] < 50]
below_40 = crossers_df[crossers_df["lower_sec_pct"] < 40]

print(f"\nCrossed below 50% completion: {len(below_50)}")
if len(below_50) > 0:
    print(below_50.to_string(index=False))
else:
    print("  NONE — 50% floor holds.")

print(f"\nCrossed below 40% completion: {len(below_40)}")
if len(below_40) > 0:
    print(below_40.to_string(index=False))
else:
    print("  NONE — 40% floor holds.")

print("\n" + "─" * 72)
print("FULL LIST OF CROSSERS (sorted by completion, lowest first)")
print("─" * 72)
print(crossers_df.to_string(index=False))

print("\n" + "─" * 72)
print("10 LOWEST-COMPLETION CROSSERS")
print("─" * 72)
print(crossers_df.head(10).to_string(index=False))

print("\n" + "─" * 72)
print("SUMMARY STATISTICS — completion at crossing")
print("─" * 72)
vals = crossers_df["lower_sec_pct"]
print(f"  Min:    {vals.min():.1f}%")
print(f"  Mean:   {vals.mean():.1f}%")
print(f"  Median: {vals.median():.1f}%")
print(f"  Max:    {vals.max():.1f}%")
print(f"  SD:     {vals.std():.1f}")
print(f"  CV:     {vals.std() / vals.mean():.3f}")

print("\n" + "─" * 72)
print("NON-CROSSERS (latest data, sorted by completion desc)")
print("Are any above 50% and still haven't crossed?")
print("─" * 72)
above_50_non = non_crossers_df[non_crossers_df["latest_lower_sec"] >= 50]
print(f"\nNon-crossers with >= 50% completion: {len(above_50_non)}")
if len(above_50_non) > 0:
    print(above_50_non.to_string(index=False))

print(f"\nFull non-crosser list ({len(non_crossers_df)} countries):")
print(non_crossers_df.to_string(index=False))
