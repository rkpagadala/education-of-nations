# =============================================================================
# PAPER REFERENCE
# Script:  scripts/residualization/female_edu_all_outcomes.py
# Paper:   "Education of Nations" (supplementary nugget)
#
# Produces:
#   Ranks the four outcomes (LE, TFR, U5MR, ChildEdu) by how much
#   female-only lower-secondary completion (age 20-24 cohort) predicts
#   each, versus both-sexes completion. Tests the Lutz/Kebede prediction
#   that maternal education has a disproportionate effect on child
#   survival (U-5 mortality).
#
# Inputs:
#   wcde/data/processed/cohort_completion_both_long.csv
#   wcde/data/processed/cohort_completion_female_long.csv
#   data/gdppercapita_us_inflation_adjusted.csv
#   data/life_expectancy_years.csv
#   data/children_per_woman_total_fertility.csv
#   data/child_mortality_u5.csv
#
# Key parameters:
#   LAG = 25                      (matches main-paper forward-lag)
#   T_YEARS = 1960, 1965, ..., 1990
#   COL_NAME = "lower_sec"        (lower-secondary completion)
#   CEILINGS = [60, 90]           (entry-cohort filter)
#
# Output:
#   checkin/female_edu_all_outcomes.json
# =============================================================================
"""
residualization/female_edu_all_outcomes.py

For each of 4 outcomes (LE, TFR, U5MR, ChildEdu):
  1. Regress outcome at T+25 on both-sexes education at T (country FE).
  2. Regress outcome at T+25 on female-only education at T (country FE).
  3. Report R² for each, the ratio female/both, and the delta (female - both).

Expectation (Lutz & Kebede 2011): U-5 mortality should show the largest
"female advantage" because maternal literacy drives sanitation / feeding
/ health-seeking behaviour. LE (adult longevity) depends more on own
habits, so the gap should be smaller.

Uses the main-paper specification: lower-secondary completion, age 20-24
cohort (cohort_completion_*_long.csv), within-country FE, 25-year lag.
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(1, os.path.dirname(SCRIPT_DIR))

from _shared import (
    load_education, load_wb, interpolate_to_annual,
    precompute_entry_years, build_panel, build_child_edu_panel,
    filter_panel, fe_r2, write_checkin, fmt_r2,
)

# ── Parameters ─────────────────────────────────────────────────────────────
T_YEARS = list(range(1960, 1995, 5))
LAG = 25
CEILINGS = [60, 90]
COL_NAME = "lower_sec"

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading cohort (age 20-24) education data...")
edu_both = load_education("cohort_completion_both_long.csv")
edu_female = load_education("cohort_completion_female_long.csv")

# cohort CSVs use cohort_year, not year — rename so interpolate_to_annual works.
edu_both = edu_both.rename(columns={"cohort_year": "year"})
edu_female = edu_female.rename(columns={"cohort_year": "year"})

# Within each country/year, WCDE cohort data may have multiple rows (from
# different obs_years / age_groups). Take the earliest obs_year per
# (country, year) as 02b_cohort_reconstruction.py documents minimises
# survivorship bias.
edu_both = (edu_both.sort_values(["country", "year", "obs_year"])
                    .groupby(["country", "year"], as_index=False).first())
edu_female = (edu_female.sort_values(["country", "year", "obs_year"])
                        .groupby(["country", "year"], as_index=False).first())

gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df = load_wb("life_expectancy_years.csv")
tfr_df = load_wb("children_per_woman_total_fertility.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

# ── Run analysis ──────────────────────────────────────────────────────────
all_results = {}

for sex_label, edu_raw in [("both", edu_both), ("female", edu_female)]:
    print(f"\n{'*' * 78}")
    print(f"* SEX: {sex_label.upper()} (cohort, age 20-24)")
    print(f"{'*' * 78}")

    edu_annual = interpolate_to_annual(edu_raw, COL_NAME)
    entry_years = precompute_entry_years(edu_annual)
    sex_results = {}

    # World Bank outcomes: LE, TFR, U5MR
    for outcome_label, outcome_col, outcome_df in [
        ("LE", "le", le_df),
        ("TFR", "tfr", tfr_df),
        ("U5MR", "u5mr", u5mr_df),
    ]:
        panel = build_panel(edu_annual, outcome_df, gdp_df, T_YEARS, LAG, outcome_col)
        outcome_results = {}
        for ceiling in CEILINGS:
            cohort = entry_years.get(10, {})
            sub = filter_panel(panel, cohort, ceiling)
            if len(sub) < 10:
                continue
            r2_e, n_e, c_e = fe_r2("edu_t", outcome_col, sub)
            outcome_results[str(ceiling)] = {
                "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
                "n": n_e, "countries": c_e,
            }
        sex_results[outcome_label] = outcome_results
        r90 = outcome_results.get("90", {})
        print(f"  {outcome_label}: R² (ceil 90%) = {fmt_r2(r90.get('edu_r2'))}  "
              f"(n={r90.get('n')}, ctry={r90.get('countries')})")

    # Child-education transmission: parent edu (T) → child edu (T+25)
    panel_ce = build_child_edu_panel(edu_annual, gdp_df, T_YEARS, LAG)
    ce_results = {}
    for ceiling in CEILINGS:
        cohort = entry_years.get(10, {})
        sub = filter_panel(panel_ce, cohort, ceiling)
        if len(sub) < 10:
            continue
        r2_e, n_e, c_e = fe_r2("edu_t", "child_edu", sub)
        ce_results[str(ceiling)] = {
            "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
            "n": n_e, "countries": c_e,
        }
    sex_results["ChildEdu"] = ce_results
    r90 = ce_results.get("90", {})
    print(f"  ChildEdu: R² (ceil 90%) = {fmt_r2(r90.get('edu_r2'))}  "
          f"(n={r90.get('n')}, ctry={r90.get('countries')})")

    all_results[sex_label] = sex_results

# ── Comparison table ──────────────────────────────────────────────────────
print(f"\n\n{'=' * 78}")
print(f"FEMALE vs BOTH-SEXES: cohort lower-sec (age 20-24), entry=10%, ceil=90%")
print(f"{'=' * 78}")
print(f"{'Outcome':<10} {'Both R²':>9} {'Female R²':>11} {'Ratio F/B':>11} {'ΔR² (F-B)':>12}")
print("-" * 56)

comparison = {}
for outcome in ["LE", "TFR", "U5MR", "ChildEdu"]:
    rb = all_results["both"].get(outcome, {}).get("90", {}).get("edu_r2")
    rf = all_results["female"].get(outcome, {}).get("90", {}).get("edu_r2")
    if rb is None or rf is None or rb == 0:
        continue
    ratio = rf / rb
    delta = rf - rb
    comparison[outcome] = {
        "both_r2": rb, "female_r2": rf,
        "ratio_f_over_b": round(ratio, 3),
        "delta_f_minus_b": round(delta, 3),
    }
    print(f"{outcome:<10} {rb:>9.3f} {rf:>11.3f} {ratio:>11.3f} {delta:>+12.3f}")

# Rank by delta (largest "female advantage")
ranked = sorted(comparison.items(), key=lambda kv: -kv[1]["delta_f_minus_b"])
print("\nRanked by ΔR² (female advantage, largest → smallest):")
for i, (outcome, vals) in enumerate(ranked, 1):
    print(f"  {i}. {outcome:<10} ΔR² = {vals['delta_f_minus_b']:+.3f}   "
          f"ratio = {vals['ratio_f_over_b']:.3f}")

# ── Write checkin JSON ────────────────────────────────────────────────────
write_checkin("female_edu_all_outcomes.json", {
    "method": ("Country FE within-R². Female-only vs both-sexes cohort "
               "lower-secondary completion (age 20-24). Entry=10%, "
               "ceilings 60/90. T=1960-1990, lag=25. Ranks outcomes by "
               "female-vs-both ΔR²."),
    "results": all_results,
    "comparison_ceil90": comparison,
    "ranked_by_delta_ceil90": [
        {"outcome": k, **v} for k, v in ranked
    ],
}, script_path="scripts/residualization/female_edu_all_outcomes.py")

print("\nWrote checkin/female_edu_all_outcomes.json")
