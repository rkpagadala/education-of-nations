"""
primary_completion_check.py — Check PRIMARY completion levels for countries
that crossed LE > 69.8 with low lower-secondary completion.

Hypothesis: countries like Greece, Portugal, Costa Rica that crossed LE > 69.8
with lower-secondary below 50% had HIGH primary completion at the time —
meaning education diffusion happened at the primary level through language,
religion, and cultural integration, even without formal secondary schooling.

Data sources:
  - Primary completion: wcde/data/processed/primary_both.csv (age 20-24, both sexes)
  - Lower-secondary:   wcde/data/processed/lower_sec_both.csv
  - Life expectancy:    data/life_expectancy_years.csv (WDI, annual 1960-2022)
  - WCDE LE:           wcde/data/processed/e0.csv (5-year, 1950-2025)
"""

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIMARY_PATH = f"{BASE}/wcde/data/processed/primary_both.csv"
LOWER_SEC_PATH = f"{BASE}/wcde/data/processed/lower_sec_both.csv"
LE_WDI_PATH = f"{BASE}/data/life_expectancy_years.csv"
LE_WCDE_PATH = f"{BASE}/wcde/data/processed/e0.csv"

LE_THRESHOLD = 69.8

OIL_STATES = ["Qatar", "United Arab Emirates", "Kuwait", "Saudi Arabia",
              "Oman", "Bahrain"]

# Regions/aggregates to exclude
AGGREGATES = [
    "Africa", "Asia", "Europe", "Latin America and the Caribbean",
    "Northern America", "Oceania", "World", "More developed regions",
    "Less developed regions", "Least developed countries",
    "Less developed regions, excluding least developed countries",
    "Less developed regions, excluding China",
    "Northern Africa", "Sub-Saharan Africa", "Eastern Africa",
    "Middle Africa", "Southern Africa", "Western Africa",
    "Eastern Asia", "South-Central Asia", "South-Eastern Asia",
    "Western Asia", "Eastern Europe", "Northern Europe",
    "Southern Europe", "Western Europe", "Caribbean",
    "Central America", "South America", "Australia/New Zealand",
    "Melanesia", "Micronesia", "Polynesia",
]

# ── Country name mapping: WCDE → WDI ─────────────────────────────────────
NAME_MAP_EDU_TO_LE = {
    "Bolivia (Plurinational State of)": "Bolivia",
    "Cape Verde": "Cabo Verde",
    "Congo": "Congo, Rep.",
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
    "Saint Lucia": "St. Lucia",
    "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
    "Swaziland": "Eswatini",
    "Syrian Arab Republic": "Syrian Arab Republic",
    "Taiwan Province of China": "Taiwan Province of China",
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
    "Gambia": "Gambia, The",
    "Egypt": "Egypt, Arab Rep.",
    "Bahamas": "Bahamas, The",
}


def load_education(path):
    """Load WCDE education CSV → long format (country, year, value)."""
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "country"})
    years = [c for c in df.columns if c != "country"]
    long = df.melt(id_vars="country", value_vars=years,
                   var_name="year", value_name="value")
    long["year"] = long["year"].astype(int)
    long = long.dropna(subset=["value"])
    return long.sort_values(["country", "year"])


def load_wdi_le():
    """Load WDI life expectancy (annual 1960-2022)."""
    df = pd.read_csv(LE_WDI_PATH)
    df = df.rename(columns={df.columns[0]: "country"})
    df["country"] = df["country"].str.strip().str.strip('"')
    years = [c for c in df.columns if c != "country"]
    long = df.melt(id_vars="country", value_vars=years,
                   var_name="year", value_name="le")
    long["year"] = long["year"].astype(int)
    long["le"] = pd.to_numeric(long["le"], errors="coerce")
    long = long.dropna(subset=["le"])
    return long.sort_values(["country", "year"])


def load_wcde_le():
    """Load WCDE LE (5-year intervals 1950-2025)."""
    df = pd.read_csv(LE_WCDE_PATH)
    df = df.rename(columns={df.columns[0]: "country"})
    years = [c for c in df.columns if c != "country"]
    long = df.melt(id_vars="country", value_vars=years,
                   var_name="year", value_name="le")
    long["year"] = long["year"].astype(int)
    long["le"] = pd.to_numeric(long["le"], errors="coerce")
    long = long.dropna(subset=["le"])
    return long.sort_values(["country", "year"])


def find_crossing_year(le_df, country, threshold):
    """Find first year country crosses threshold. Returns year or None."""
    sub = le_df[le_df["country"] == country].sort_values("year")
    crossed = sub[sub["le"] >= threshold]
    if len(crossed) > 0:
        return crossed.iloc[0]["year"]
    return None


def get_education_at_year(edu_df, country, year):
    """Get education value at or nearest to given year (5-year intervals)."""
    sub = edu_df[edu_df["country"] == country].sort_values("year")
    if len(sub) == 0:
        return np.nan
    # Find closest year <= target, then interpolate if needed
    before = sub[sub["year"] <= year]
    after = sub[sub["year"] >= year]
    if len(before) > 0 and len(after) > 0:
        b = before.iloc[-1]
        a = after.iloc[0]
        if b["year"] == a["year"]:
            return b["value"]
        # Linear interpolation
        frac = (year - b["year"]) / (a["year"] - b["year"])
        return b["value"] + frac * (a["value"] - b["value"])
    elif len(before) > 0:
        return before.iloc[-1]["value"]
    elif len(after) > 0:
        return after.iloc[0]["value"]
    return np.nan


# ── Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
primary = load_education(PRIMARY_PATH)
lower_sec = load_education(LOWER_SEC_PATH)
le_wdi = load_wdi_le()
le_wcde = load_wcde_le()

# Get all WCDE country names (from primary)
all_edu_countries = sorted(primary["country"].unique())
all_edu_countries = [c for c in all_edu_countries if c not in AGGREGATES]

# ── For each WCDE country, find LE crossing year ─────────────────────────
print(f"\nFinding LE > {LE_THRESHOLD} crossing years...")

results = []
for edu_country in all_edu_countries:
    # Map to WDI name
    le_country = NAME_MAP_EDU_TO_LE.get(edu_country, edu_country)

    # Try WDI first (annual resolution), then WCDE (5-year)
    cross_year = find_crossing_year(le_wdi, le_country, LE_THRESHOLD)
    le_source = "WDI"
    if cross_year is None:
        cross_year = find_crossing_year(le_wcde, edu_country, LE_THRESHOLD)
        le_source = "WCDE"

    if cross_year is None:
        # Check max LE to see if country is close
        sub_wdi = le_wdi[le_wdi["country"] == le_country]
        sub_wcde = le_wcde[le_wcde["country"] == edu_country]
        max_le = max(
            sub_wdi["le"].max() if len(sub_wdi) > 0 else 0,
            sub_wcde["le"].max() if len(sub_wcde) > 0 else 0
        )
        results.append({
            "country": edu_country,
            "crossed": False,
            "cross_year": None,
            "le_source": None,
            "max_le": max_le,
            "primary_at_cross": np.nan,
            "lower_sec_at_cross": np.nan,
        })
        continue

    prim_val = get_education_at_year(primary, edu_country, cross_year)
    lsec_val = get_education_at_year(lower_sec, edu_country, cross_year)

    results.append({
        "country": edu_country,
        "crossed": True,
        "cross_year": cross_year,
        "le_source": le_source,
        "max_le": np.nan,
        "primary_at_cross": prim_val,
        "lower_sec_at_cross": lsec_val,
    })

df = pd.DataFrame(results)

# Exclude oil states
df["is_oil"] = df["country"].isin(OIL_STATES)

# ── ANALYSIS 1: Countries that crossed with LOW lower-secondary (<50%) ───
print("\n" + "=" * 80)
print("ANALYSIS 1: Countries that crossed LE > 69.8 with lower-secondary < 50%")
print("=" * 80)

crossed = df[(df["crossed"]) & (~df["is_oil"])].copy()
low_lsec = crossed[crossed["lower_sec_at_cross"] < 50].sort_values("lower_sec_at_cross")

print(f"\n{'Country':<45} {'Year':>5} {'Primary%':>9} {'LowerSec%':>10}")
print("-" * 72)
for _, r in low_lsec.iterrows():
    print(f"{r['country']:<45} {int(r['cross_year']):>5} "
          f"{r['primary_at_cross']:>8.1f}% {r['lower_sec_at_cross']:>9.1f}%")

print(f"\n  Count: {len(low_lsec)} countries crossed LE > 69.8 with lower-sec < 50%")
if len(low_lsec) > 0:
    print(f"  Primary completion range: {low_lsec['primary_at_cross'].min():.1f}% – "
          f"{low_lsec['primary_at_cross'].max():.1f}%")
    print(f"  Primary completion mean:  {low_lsec['primary_at_cross'].mean():.1f}%")
    print(f"  Primary completion median: {low_lsec['primary_at_cross'].median():.1f}%")

# ── ANALYSIS 2: All crossers — primary and lower-sec at crossing ─────────
print("\n" + "=" * 80)
print("ANALYSIS 2: ALL countries that crossed LE > 69.8 (sorted by lower-sec)")
print("=" * 80)

all_crossers = crossed.sort_values("lower_sec_at_cross")
print(f"\n{'Country':<45} {'Year':>5} {'Primary%':>9} {'LowerSec%':>10}")
print("-" * 72)
for _, r in all_crossers.iterrows():
    marker = " ***" if r["lower_sec_at_cross"] < 50 else ""
    print(f"{r['country']:<45} {int(r['cross_year']):>5} "
          f"{r['primary_at_cross']:>8.1f}% {r['lower_sec_at_cross']:>9.1f}%{marker}")

print(f"\n  *** = lower-secondary < 50%")

# ── ANALYSIS 3: Table 4 countries specifically ───────────────────────────
print("\n" + "=" * 80)
print("ANALYSIS 3: Table 4 countries — primary completion at LE crossing")
print("=" * 80)

TABLE4_COUNTRIES = {
    "Taiwan Province of China": "Taiwan",
    "Republic of Korea": "S. Korea",
    "Cuba": "Cuba",
    "Bangladesh": "Bangladesh",
    "Sri Lanka": "Sri Lanka",
    "China": "China",
}

print(f"\n{'Country':<45} {'Year':>5} {'Primary%':>9} {'LowerSec%':>10}")
print("-" * 72)
for edu_name, display in TABLE4_COUNTRIES.items():
    row = df[df["country"] == edu_name]
    if len(row) > 0:
        r = row.iloc[0]
        if r["crossed"]:
            print(f"{display:<45} {int(r['cross_year']):>5} "
                  f"{r['primary_at_cross']:>8.1f}% {r['lower_sec_at_cross']:>9.1f}%")
        else:
            print(f"{display:<45}  NOT CROSSED (max LE: {r['max_le']:.1f})")
    else:
        print(f"{display:<45}  NOT FOUND in data")

# ── ANALYSIS 4: Countries that HAVEN'T crossed — primary & lower-sec ────
print("\n" + "=" * 80)
print("ANALYSIS 4: Countries that HAVEN'T crossed LE > 69.8 — latest education levels")
print("=" * 80)

not_crossed = df[(~df["crossed"]) & (~df["is_oil"])].copy()
not_crossed = not_crossed[not_crossed["max_le"] > 0]  # exclude missing

# Get latest primary and lower-sec for these countries
latest_year = primary["year"].max()
for idx in not_crossed.index:
    c = not_crossed.loc[idx, "country"]
    p_sub = primary[(primary["country"] == c) & (primary["year"] == latest_year)]
    l_sub = lower_sec[(lower_sec["country"] == c) & (lower_sec["year"] == latest_year)]
    not_crossed.loc[idx, "primary_latest"] = p_sub["value"].values[0] if len(p_sub) > 0 else np.nan
    not_crossed.loc[idx, "lower_sec_latest"] = l_sub["value"].values[0] if len(l_sub) > 0 else np.nan

not_crossed = not_crossed.sort_values("max_le", ascending=False)

print(f"\n{'Country':<45} {'MaxLE':>6} {'Primary%(latest)':>16} {'LowerSec%(latest)':>17}")
print("-" * 87)
for _, r in not_crossed.head(30).iterrows():
    print(f"{r['country']:<45} {r['max_le']:>5.1f} "
          f"{r.get('primary_latest', np.nan):>15.1f}% "
          f"{r.get('lower_sec_latest', np.nan):>16.1f}%")

# ── ANALYSIS 5: Key hypothesis test — summary statistics ─────────────────
print("\n" + "=" * 80)
print("ANALYSIS 5: HYPOTHESIS TEST SUMMARY")
print("=" * 80)

# Group crossers by lower-sec level
high_lsec_crossers = crossed[crossed["lower_sec_at_cross"] >= 50]
low_lsec_crossers = crossed[crossed["lower_sec_at_cross"] < 50]

print(f"\nCountries that crossed LE > 69.8 WITH lower-sec >= 50%:")
print(f"  N = {len(high_lsec_crossers)}")
print(f"  Primary completion: mean={high_lsec_crossers['primary_at_cross'].mean():.1f}%, "
      f"min={high_lsec_crossers['primary_at_cross'].min():.1f}%, "
      f"median={high_lsec_crossers['primary_at_cross'].median():.1f}%")

print(f"\nCountries that crossed LE > 69.8 WITH lower-sec < 50%:")
print(f"  N = {len(low_lsec_crossers)}")
if len(low_lsec_crossers) > 0:
    print(f"  Primary completion: mean={low_lsec_crossers['primary_at_cross'].mean():.1f}%, "
          f"min={low_lsec_crossers['primary_at_cross'].min():.1f}%, "
          f"median={low_lsec_crossers['primary_at_cross'].median():.1f}%")

# Non-crossers close to threshold (LE > 60)
close_non_crossers = not_crossed[not_crossed["max_le"] > 60].copy()
print(f"\nCountries with max LE > 60 that HAVEN'T crossed 69.8:")
print(f"  N = {len(close_non_crossers)}")
if len(close_non_crossers) > 0 and "primary_latest" in close_non_crossers.columns:
    valid = close_non_crossers.dropna(subset=["primary_latest"])
    print(f"  Primary (latest): mean={valid['primary_latest'].mean():.1f}%, "
          f"min={valid['primary_latest'].min():.1f}%, "
          f"median={valid['primary_latest'].median():.1f}%")
    valid2 = close_non_crossers.dropna(subset=["lower_sec_latest"])
    print(f"  Lower-sec (latest): mean={valid2['lower_sec_latest'].mean():.1f}%, "
          f"min={valid2['lower_sec_latest'].min():.1f}%, "
          f"median={valid2['lower_sec_latest'].median():.1f}%")

# ── ANALYSIS 6: Specific focus countries ──────────────────────────────────
print("\n" + "=" * 80)
print("ANALYSIS 6: FOCUS COUNTRIES — detailed primary trajectory")
print("=" * 80)

FOCUS = ["Greece", "Portugal", "Costa Rica", "Syrian Arab Republic",
         "Honduras", "Nicaragua", "Morocco"]

for c in FOCUS:
    row = df[df["country"] == c]
    if len(row) == 0:
        print(f"\n{c}: NOT FOUND in data")
        continue
    r = row.iloc[0]

    print(f"\n{c}:")
    if r["crossed"]:
        print(f"  LE crossing year: {int(r['cross_year'])} (source: {r.get('le_source', '?')})")
        print(f"  At crossing: Primary = {r['primary_at_cross']:.1f}%, "
              f"Lower-sec = {r['lower_sec_at_cross']:.1f}%")
    else:
        print(f"  Has NOT crossed LE > 69.8 (max LE: {r['max_le']:.1f})")

    # Show trajectory
    p_traj = primary[primary["country"] == c].sort_values("year")
    l_traj = lower_sec[lower_sec["country"] == c].sort_values("year")

    print(f"  {'Year':>6} {'Primary%':>9} {'LowerSec%':>10}")
    for _, pt in p_traj.iterrows():
        yr = pt["year"]
        lt = l_traj[l_traj["year"] == yr]
        lv = lt["value"].values[0] if len(lt) > 0 else np.nan
        marker = " <-- LE crossing" if r["crossed"] and yr == int(r["cross_year"]) else ""
        # Also check closest crossing year for 5-year data
        if r["crossed"] and abs(yr - int(r["cross_year"])) <= 2 and marker == "":
            marker = " <-- ~LE crossing"
        print(f"  {int(yr):>6} {pt['value']:>8.1f}% {lv:>9.1f}%{marker}")

# ── ANALYSIS 7: Is primary >80% a necessary condition? ───────────────────
print("\n" + "=" * 80)
print("ANALYSIS 7: Is primary > 80% a NECESSARY condition for LE > 69.8?")
print("=" * 80)

low_primary_crossers = crossed[crossed["primary_at_cross"] < 80]
print(f"\nCountries that crossed LE > 69.8 with PRIMARY < 80%:")
print(f"  N = {len(low_primary_crossers)}")
if len(low_primary_crossers) > 0:
    print(f"\n{'Country':<45} {'Year':>5} {'Primary%':>9} {'LowerSec%':>10}")
    print("-" * 72)
    for _, r in low_primary_crossers.sort_values("primary_at_cross").iterrows():
        print(f"{r['country']:<45} {int(r['cross_year']):>5} "
              f"{r['primary_at_cross']:>8.1f}% {r['lower_sec_at_cross']:>9.1f}%")

high_primary_crossers = crossed[crossed["primary_at_cross"] >= 80]
print(f"\nCountries that crossed LE > 69.8 with PRIMARY >= 80%:")
print(f"  N = {len(high_primary_crossers)}")
print(f"  (of {len(crossed)} total non-oil crossers = "
      f"{100*len(high_primary_crossers)/len(crossed):.0f}%)")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
