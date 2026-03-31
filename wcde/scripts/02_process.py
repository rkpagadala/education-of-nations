"""
02_process.py
Process raw WCDE v3 downloads into analysis-ready completion-rate CSVs.

Input:  wcde/data/raw/*.csv
Output: wcde/data/processed/
  completion_both.csv    — wide: country × year, columns = pri/lowsec/uppsec/college (both sexes)
  completion_female.csv  — same for female only
  completion_male.csv    — same for male only
  tfr.csv                — country × year TFR
  e0.csv                 — country × year life expectancy (both)
  country_list.csv       — country codes + names (all 228 entities)

Education level mapping (WCDE 9 categories → 4 completion rates):
  Primary completion    = Primary + Lower Sec + Upper Sec + Short Post Sec + Post Sec + Bachelor + Master
  Lower sec completion  = Lower Sec + Upper Sec + Short Post Sec + Post Sec + Bachelor + Master
  Upper sec completion  = Upper Sec + Short Post Sec + Post Sec + Bachelor + Master
  College completion    = Short Post Sec + Post Sec + Bachelor + Master

Age group used: 20--24 (completed cohort, not currently enrolled)
Years kept: 1960-2025 (historical reconstruction + SSP2 projections)
"""

import os, sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(SCRIPT_DIR, "../data/raw")
PROC = os.path.join(SCRIPT_DIR, "../data/processed")
os.makedirs(PROC, exist_ok=True)

YEARS_KEEP = list(range(1950, 2030, 5))  # 1950, 1955, 1960, ... 2025 (pre-1960 valid for non-colonial countries)

# Education levels that count toward each completion tier (cumulative from top)
LEVELS_FOR = {
    "primary":  ["Primary","Lower Secondary","Upper Secondary",
                 "Short Post Secondary","Post Secondary","Bachelor","Master and higher"],
    "lower_sec":["Lower Secondary","Upper Secondary",
                 "Short Post Secondary","Post Secondary","Bachelor","Master and higher"],
    "upper_sec":["Upper Secondary",
                 "Short Post Secondary","Post Secondary","Bachelor","Master and higher"],
    "college":  ["Short Post Secondary","Post Secondary","Bachelor","Master and higher"],
}

def process_prop(path, label, sex_filter):
    print(f"Processing {os.path.basename(path)} -> {label} (sex={sex_filter})...")
    df = pd.read_csv(path)

    # Keep only 20-24 age group, years of interest, and correct sex
    df = df[df["age"] == "20--24"].copy()
    df = df[df["year"].isin(YEARS_KEEP)].copy()
    df = df[df["sex"] == sex_filter].copy()

    # Normalise country name
    df["country"] = df["name"].str.strip()

    results = []
    for (country, year), grp in df.groupby(["country","year"]):
        edu_dict = dict(zip(grp["education"], grp["prop"]))
        row = {"country": country, "year": year}
        for level, cats in LEVELS_FOR.items():
            vals = [edu_dict.get(c, np.nan) for c in cats]
            valid = [v for v in vals if not np.isnan(v)]
            row[level] = sum(valid) if valid else np.nan
        results.append(row)

    long = pd.DataFrame(results)

    # Save long format (more flexible for analysis)
    long_path = os.path.join(PROC, f"completion_{label}_long.csv")
    long.to_csv(long_path, index=False, float_format="%.4f")
    print(f"  Long: {long_path} — {len(long)} rows, {long['country'].nunique()} countries")

    # Save wide format per level (country × year) — matches existing dataset format
    for level in LEVELS_FOR:
        wide = long.pivot(index="country", columns="year", values=level)
        wide.columns = [str(c) for c in wide.columns]
        wide_path = os.path.join(PROC, f"{level}_{label}.csv")
        wide.to_csv(wide_path, float_format="%.2f")
    print(f"  Wide CSVs saved for {list(LEVELS_FOR.keys())}")
    return long

# Process both, female, and male completion rates
# Note: both raw files contain rows for sex=Both/Male/Female; sex_filter selects the correct rows.
prop_both   = process_prop(os.path.join(RAW, "prop_both.csv"),   "both",   sex_filter="Both")
prop_female = process_prop(os.path.join(RAW, "prop_female.csv"), "female", sex_filter="Female")
prop_male   = process_prop(os.path.join(RAW, "prop_female.csv"), "male",   sex_filter="Male")

# Save country list
countries = prop_both[["country"]].drop_duplicates().sort_values("country")
countries.to_csv(os.path.join(PROC, "country_list.csv"), index=False)
print(f"\nCountry list: {len(countries)} entities")

# Process TFR
print("\nProcessing TFR...")
tfr = pd.read_csv(os.path.join(RAW, "tfr.csv"))
tfr["year"] = tfr["period"].str.split("-").str[0].astype(int)
tfr = tfr[tfr["year"].isin(YEARS_KEEP)].copy()
tfr["country"] = tfr["name"].str.strip()
tfr_wide = tfr.pivot(index="country", columns="year", values="tfr")
tfr_wide.columns = [str(c) for c in tfr_wide.columns]
tfr_wide.to_csv(os.path.join(PROC, "tfr.csv"), float_format="%.3f")
print(f"  TFR: {tfr_wide.shape} — {tfr['country'].nunique()} countries")

# Process life expectancy (both sexes)
print("\nProcessing life expectancy...")
e0 = pd.read_csv(os.path.join(RAW, "e0_both.csv"))
e0["year"] = e0["period"].str.split("-").str[0].astype(int)
e0 = e0[e0["year"].isin(YEARS_KEEP)].copy()
e0["country"] = e0["name"].str.strip()
e0 = e0.groupby(["country","year"])["e0"].mean().reset_index()  # average male+female
e0_wide = e0.pivot(index="country", columns="year", values="e0")
e0_wide.columns = [str(c) for c in e0_wide.columns]
e0_wide.to_csv(os.path.join(PROC, "e0.csv"), float_format="%.2f")
print(f"  E0: {e0_wide.shape} — {e0['country'].nunique()} countries")

# Summary
print("\n=== Processing complete ===")
print(f"Output: {PROC}")
print(f"Countries: {prop_both['country'].nunique()}")
print(f"Years: {sorted(prop_both['year'].unique())}")

# Spot-check Korea and Taiwan
for c in ["Republic of Korea","Taiwan Province of China","India","Niger"]:
    sub = prop_both[prop_both["country"] == c][["country","year","primary","lower_sec","upper_sec","college"]]
    sub = sub[sub["year"].isin([1960,1980,2000,2015,2025])].sort_values("year")
    if len(sub):
        print(f"\n{c}:")
        print(sub.to_string(index=False))
