"""
02b_cohort_reconstruction.py
Reconstruct historical education by cohort (going back to ~1875-1890).

WCDE provides education attainment by age group at each observation year.
By looking at OLDER age groups at EARLIER observation years, we can
reconstruct the education levels of past cohorts when they were young adults.

Method:
  For each (country, obs_year, age_group, education_level), compute:
    cohort_year = obs_year - (midpoint_age - 22)
  This is the year the cohort was 20-24 (their "young adult year").

  For each (country, cohort_year), take the EARLIEST available observation
  (youngest age group measured), which minimises survivorship bias.

Coverage:
  - Directly from WCDE: obs_year 1950-2015, age groups 20-24 through 100+
  - Earliest cohort_year: 1950 - (97-22) = 1875 (age 95-99 in 1950)
  - Most reliable (before heavy survivorship bias): cohort_years 1890-1960

Caveats:
  - Pre-1960 data is reliable for countries with good historical records
    (Japan, USA, UK, Western Europe, Australia, Canada, Argentina, Chile).
  - For colonised countries, pre-1960 reflects colonial investment decisions.
  - Sri Lanka is an exception: British colonial education investment was active.
  - Survivorship bias: two opposing effects.
    (a) Education → longevity: educated people survive to old age at higher rates.
        This OVERESTIMATES education for cohorts measured at 70+ (pre-1910 cohorts).
    (b) Women live longer: historically women had lower education + higher longevity.
        This partially OFFSETS (a) by pulling the surviving pool toward lower-educated
        women. Effect strongest where pre-1940 gender gaps were large.
    Net: modest upward bias for pre-1920 cohorts; minimal for 1930+ where youngest
    age groups (35-50) are used.
  - Implication for T-25 regression β: parent cohort is observed older than child
    cohort → parent education inflated more → β is biased downward (conservative).
  - Use with caution before 1900.

Output (per sex in {both, female, male}):
  wcde/data/processed/cohort_completion_{sex}_long.csv
    columns: country, cohort_year, primary, lower_sec, upper_sec, college,
             obs_year, age_group (diagnostic columns)

  wcde/data/processed/cohort_lower_sec_{sex}.csv  (wide: country × cohort_year)
  wcde/data/processed/cohort_primary_{sex}.csv
"""

import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(SCRIPT_DIR, "../data/raw")
PROC = os.path.join(SCRIPT_DIR, "../data/processed")
os.makedirs(PROC, exist_ok=True)

# WCDE 9-level → 4 completion rates (cumulative from top)
LEVELS_FOR = {
    "primary":   ["Primary","Lower Secondary","Upper Secondary",
                  "Short Post Secondary","Post Secondary","Bachelor","Master and higher"],
    "lower_sec": ["Lower Secondary","Upper Secondary",
                  "Short Post Secondary","Post Secondary","Bachelor","Master and higher"],
    "upper_sec": ["Upper Secondary",
                  "Short Post Secondary","Post Secondary","Bachelor","Master and higher"],
    "college":   ["Short Post Secondary","Post Secondary","Bachelor","Master and higher"],
}

# Age group → midpoint age → offset from 22 (when cohort was ~20-24)
AGE_MIDPOINTS = {
    "20--24": 22,
    "25--29": 27,
    "30--34": 32,
    "35--39": 37,
    "40--44": 42,
    "45--49": 47,
    "50--54": 52,
    "55--59": 57,
    "60--64": 62,
    "65--69": 67,
    "70--74": 72,
    "75--79": 77,
    "80--84": 82,
    "85--89": 87,
    "90--94": 92,
    "95--99": 97,
    "100+":   102,
}

# Limit obs years to historical (not projections)
OBS_YEARS_KEEP = list(range(1950, 2020, 5))

def reconstruct_cohorts(raw_path, sex_label, sex_filter):
    """
    Load raw WCDE prop file, filter to sex_filter, and reconstruct cohort
    completion rates back to ~1875. Saves long and wide CSVs.
    """
    print(f"\n=== Cohort reconstruction: {sex_label} (sex={sex_filter}) ===")
    print(f"Loading {os.path.basename(raw_path)}...")
    df = pd.read_csv(raw_path)
    print(f"  Rows: {len(df):,}")

    df["country"] = df["name"].str.strip()

    # Filter to historical obs years, relevant age groups, and correct sex
    df = df[df["year"].isin(OBS_YEARS_KEEP)].copy()
    df = df[df["age"].isin(AGE_MIDPOINTS.keys())].copy()
    df = df[df["sex"] == sex_filter].copy()
    print(f"  After filtering: {len(df):,} rows")

    # Compute cohort_year for each observation
    df["midpoint_age"] = df["age"].map(AGE_MIDPOINTS)
    df["cohort_year"]  = df["year"] - (df["midpoint_age"] - 22)

    # Drop cohorts before 1870 (too sparse/unreliable)
    df = df[df["cohort_year"] >= 1870].copy()

    # For each (country, cohort_year, education): keep EARLIEST obs_year
    # (youngest age group = least survivorship bias)
    df_best = (df.sort_values(["country","cohort_year","year"])
                 .groupby(["country","cohort_year","education"], as_index=False)
                 .first())

    print(f"  Unique (country, cohort_year) pairs: {df_best.groupby(['country','cohort_year']).ngroups:,}")

    # Compute completion rates
    print("  Computing completion rates by cohort...")
    results = []
    for (country, cohort_yr), grp in df_best.groupby(["country","cohort_year"]):
        best_obs_year  = grp["year"].min()
        best_age_group = grp.loc[grp["year"] == best_obs_year, "age"].iloc[0] if len(grp) > 0 else "?"
        edu_dict = dict(zip(grp["education"], grp["prop"]))
        row = {"country": country, "cohort_year": cohort_yr,
               "obs_year": best_obs_year, "age_group": best_age_group}
        for level, cats in LEVELS_FOR.items():
            vals = [edu_dict.get(c, np.nan) for c in cats]
            valid = [v for v in vals if not np.isnan(v)]
            row[level] = min(sum(valid), 100.0) if valid else np.nan
        results.append(row)

    long = pd.DataFrame(results).sort_values(["country","cohort_year"]).reset_index(drop=True)
    print(f"  Long table: {len(long)} rows, {long['country'].nunique()} countries")
    print(f"  Cohort years: {long['cohort_year'].min()} – {long['cohort_year'].max()}")

    # Save long format
    long_path = os.path.join(PROC, f"cohort_completion_{sex_label}_long.csv")
    long.to_csv(long_path, index=False, float_format="%.2f")
    print(f"  Saved: {long_path}")

    # Save wide format for lower_sec and primary
    for level in ["lower_sec", "primary"]:
        wide = long.pivot(index="country", columns="cohort_year", values=level)
        wide.columns = [str(int(c)) for c in wide.columns]
        wide_path = os.path.join(PROC, f"cohort_{level}_{sex_label}.csv")
        wide.to_csv(wide_path, float_format="%.2f")
        print(f"  Saved: {wide_path} — {wide.shape}")

    return long


# Run all three sexes
# prop_both.csv and prop_female.csv both contain rows for Both/Female/Male
long_both   = reconstruct_cohorts(os.path.join(RAW, "prop_both.csv"),   "both",   "Both")
long_female = reconstruct_cohorts(os.path.join(RAW, "prop_female.csv"), "female", "Female")
long_male   = reconstruct_cohorts(os.path.join(RAW, "prop_female.csv"), "male",   "Male")

# Spot-check (both sexes)
print("\n=== Spot-check (both) ===")
for c in ["Japan", "United States of America",
          "United Kingdom of Great Britain and Northern Ireland",
          "Sri Lanka", "Republic of Korea", "India"]:
    sub = long_both[long_both["country"] == c][
        ["country","cohort_year","primary","lower_sec","upper_sec","college","obs_year","age_group"]]
    sub = sub[sub["cohort_year"].isin(
        [1875,1890,1900,1910,1920,1930,1940,1950,1960,1970,1980,2000,2015])].sort_values("cohort_year")
    if len(sub):
        print(f"\n{c}:")
        print(sub.to_string(index=False))

print("\n=== Cohort reconstruction complete (both + female + male) ===")
print("Note: pre-1900 estimates subject to survivorship bias.")
print("Female/male cohort series: additional caveat — women's higher longevity")
print("means female pre-1930 cohorts are observed at older ages and carry")
print("stronger survivorship bias than the combined series.")
