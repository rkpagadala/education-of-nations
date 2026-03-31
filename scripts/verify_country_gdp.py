# =============================================================================
# PAPER REFERENCE
# Script:  scripts/verify_country_gdp.py
# Paper:   "Education of Nations"
#
# Produces:
#   Verification of all country-specific GDP per capita figures cited in
#   the paper. Loads World Bank constant-2017-USD GDP data, checks each
#   claim, flags mismatches, and writes checkin/country_gdp.json.
#
# Inputs:
#   data/gdppercapita_us_inflation_adjusted.csv
#
# Cross-check for all sections that cite GDP per capita.
# =============================================================================
"""
verify_country_gdp.py

Load World Bank GDP per capita data (constant 2017 USD) and verify every
country-specific GDP figure cited in the paper.
"""

import os
import json
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..")
DATA = os.path.join(ROOT, "data")
CHECKIN = os.path.join(ROOT, "checkin")
os.makedirs(CHECKIN, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
gdp_raw = pd.read_csv(os.path.join(DATA, "gdppercapita_us_inflation_adjusted.csv"))

# The first column is Country; remaining columns are year strings.
# Country names may be quoted (e.g. "Korea, Rep.").
gdp_raw.rename(columns={gdp_raw.columns[0]: "Country"}, inplace=True)
gdp_raw["Country"] = gdp_raw["Country"].str.strip().str.strip('"')

# Melt to long format
year_cols = [c for c in gdp_raw.columns if c != "Country"]
gdp = gdp_raw.melt(id_vars="Country", var_name="year", value_name="gdp")
gdp["year"] = pd.to_numeric(gdp["year"], errors="coerce")
gdp["gdp"] = pd.to_numeric(gdp["gdp"], errors="coerce")

# Lowercase country for matching
gdp["country_lc"] = gdp["Country"].str.lower()

def get_gdp(country_lc, year):
    """Get GDP for a country/year using lowercase matching."""
    row = gdp[(gdp["country_lc"] == country_lc) & (gdp["year"] == year)]
    if row.empty or pd.isna(row["gdp"].iloc[0]):
        return None
    return round(float(row["gdp"].iloc[0]), 0)

# ── Define claims ────────────────────────────────────────────────────────────
# (label, wb_country_lowercase, year, cited_value, tolerance_pct)
# tolerance_pct = fractional tolerance (e.g. 0.15 = 15%)
CLAIMS = [
    ("Korea 1960 GDP", "korea, rep.", 1960, 1038, 0.15),
    ("Costa Rica 1960 GDP", "costa rica", 1960, 3609, 0.15),
    ("Costa Rica 1990 GDP", "costa rica", 1990, 6037, 0.15),
    ("Korea 1990 GDP", "korea, rep.", 1990, 9673, 0.15),
    ("Bangladesh 2014 GDP", "bangladesh", 2014, 1159, 0.15),
    ("Nepal ~1990 GDP", "nepal", 1990, 423, 0.20),
    ("Myanmar 2015 GDP (<$1200)", "myanmar", 2015, 1200, 0.25),
    ("Qatar 2015 GDP (~$69k)", "qatar", 2015, 69000, 0.20),
]

# ── Run checks ───────────────────────────────────────────────────────────────
results = []
print("=" * 80)
print("COUNTRY GDP VERIFICATION (constant 2017 USD)")
print("=" * 80)
print(f"{'Claim':<30} {'Cited':>10} {'Actual':>10} {'Diff%':>8} {'OK?':>5}")
print("-" * 80)

for label, country_lc, year, cited, tol in CLAIMS:
    actual = get_gdp(country_lc, year)
    if actual is None:
        diff_pct = None
        ok = False
        actual_str = "N/A"
        diff_str = "N/A"
    else:
        diff_pct = (actual - cited) / cited
        ok = bool(abs(diff_pct) <= tol)
        actual_str = f"${actual:,.0f}"
        diff_str = f"{diff_pct:+.1%}"

    # Special case: Myanmar check is "below $1,200"
    if "Myanmar" in label and actual is not None:
        ok = bool(actual < 1200)
        diff_str = f"{'<' if actual < 1200 else '>='} $1200"

    status = "PASS" if ok else "FAIL"
    cited_str = f"${cited:,}"
    print(f"{label:<30} {cited_str:>10} {actual_str:>10} {diff_str:>8} {status:>5}")
    results.append({
        "claim": label,
        "cited": cited,
        "actual": actual,
        "diff_pct": round(diff_pct, 4) if diff_pct is not None else None,
        "pass": ok,
    })

# ── Also print nearby years for Nepal ────────────────────────────────────────
print("\nNepal GDP around 1990s:")
for yr in range(1988, 2000):
    v = get_gdp("nepal", yr)
    if v:
        print(f"  {yr}: ${v:,.0f}")

# ── Summary ──────────────────────────────────────────────────────────────────
n_pass = sum(1 for r in results if r["pass"])
n_total = len(results)
print(f"\n{'=' * 80}")
print(f"SUMMARY: {n_pass}/{n_total} GDP claims passed")
print(f"{'=' * 80}")

# ── Write checkin ────────────────────────────────────────────────────────────
checkin = {
    "script": "scripts/verify_country_gdp.py",
    "source": "data/gdppercapita_us_inflation_adjusted.csv",
    "claims": results,
    "summary": f"{n_pass}/{n_total} GDP claims passed",
}
out_path = os.path.join(CHECKIN, "country_gdp.json")
with open(out_path, "w") as f:
    json.dump(checkin, f, indent=2)
print(f"\nCheckin written to {out_path}")
