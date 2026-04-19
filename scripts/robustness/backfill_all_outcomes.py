"""
backfill_all_outcomes.py

Generalises le_r2_by_lag_backfilled.py from life expectancy to all four
outcomes in the paper: LE, TFR, U-5 mortality (log), and child education
(lower-secondary completion as outcome).

Motivation:
  Reviewers object that GDP's weak long-lag performance is an artifact of
  thin pre-1960 coverage rather than a real lack of signal. The LE-only
  backfill test already refutes this for life expectancy. This script
  extends the same test to the other three outcomes so the defense
  generalises.

Method (identical to le_r2_by_lag_backfilled.py):
  For each lag L in {0, 5, ..., 100} and each outcome Y in {LE, TFR,
  U5log, ChildEdu}:
    - Outcome at year T (WDI 1960-2015, or WCDE for ChildEdu)
    - Predictor A: lower-secondary completion at T-L (WCDE, 1875-2015)
    - Predictor B: log GDP/cap at T-L, backfilled to $500/cap for all
                   country-years before 1960 (constant 2017 USD), no lag cap.
    - Country fixed effects: demean each series by country mean.
    - Record within-country R² for each predictor.

  The claim: backfilled-GDP R² collapses to (near) zero once the predictor
  window sits entirely in the backfilled region because a constant has no
  cross-country variance to predict anything. Education, drawn from the
  full WCDE panel, remains informative across the full 100-year horizon.

Outputs:
  - Console summary table per outcome.
  - checkin/backfill_all_outcomes.json
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import PROC as WCDE_PROC, DATA as WB_DIR, write_checkin

# ── parameters (match le_r2_by_lag_backfilled.py) ─────────────────────────────
LAG_MIN     = 0
LAG_MAX     = 100
LAG_STEP    = 5
PANEL_START = 1960   # outcome panel start (WDI coverage)
PANEL_END   = 2015   # outcome panel end

SUBSISTENCE_GDP = 500    # constant 2017 USD (approximate pre-industrial income)
BACKFILL_START  = 1870   # earliest year to backfill GDP
BACKFILL_END    = 1959   # last year before WDI GDP data begins

MIN_OBS       = 500   # minimum observations for a valid R² estimate
MIN_OBS_PER_C = 3     # drop countries with < this many obs
GDP_LAG_MAX   = 25    # cap for original (non-backfilled) GDP, per fig_a1

# threshold for "GDP R² hits zero" in summary reporting
ZERO_R2_THRESHOLD = 0.01


# ── load data ─────────────────────────────────────────────────────────────────
def load_wide(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.str.lower().str.strip()
    df.columns = df.columns.astype(int)
    return df.clip(lower=0)


# WCDE v3 cohort lower-secondary completion — interpolated to annual
_edu_raw = pd.read_csv(os.path.join(WCDE_PROC, "cohort_lower_sec_both.csv"),
                       index_col="country")
_edu_raw.columns = [int(c) for c in _edu_raw.columns]
_all_yrs = list(range(min(_edu_raw.columns), max(_edu_raw.columns) + 1))
_edu_raw = _edu_raw.reindex(columns=_all_yrs).interpolate(axis=1) \
                    .bfill(axis=1).ffill(axis=1)
_edu_raw.index = _edu_raw.index.str.lower().str.strip()
edu = _edu_raw.clip(lower=0)

gdp_orig = load_wide(os.path.join(WB_DIR, "gdppercapita_us_inflation_adjusted.csv"))
le_df    = load_wide(os.path.join(WB_DIR, "life_expectancy_years.csv"))
tfr_df   = load_wide(os.path.join(WB_DIR, "children_per_woman_total_fertility.csv"))
u5_df    = load_wide(os.path.join(WB_DIR, "child_mortality_u5.csv"))


# ── build backfilled GDP (identical construction to LE script) ────────────────
gdp_back = gdp_orig.copy()
backfill_years = list(range(BACKFILL_START, BACKFILL_END + 1))
for yr in backfill_years:
    if yr not in gdp_back.columns:
        gdp_back[yr] = SUBSISTENCE_GDP
    else:
        gdp_back[yr] = gdp_back[yr].fillna(SUBSISTENCE_GDP)

for yr in gdp_back.columns:
    if yr >= 1960:
        gdp_back[yr] = gdp_back[yr].fillna(SUBSISTENCE_GDP)

gdp_back = gdp_back[sorted(gdp_back.columns)]

# common countries across all relevant datasets (edu always; outcome varies
# but we take the intersection for a fair single-panel comparison across Ys)
countries = sorted(
    set(edu.index)
    & set(gdp_orig.index)
    & set(le_df.index)
    & set(tfr_df.index)
    & set(u5_df.index)
)

print(f"N countries (intersection of edu/GDP/LE/TFR/U5): {len(countries)}")
print(f"GDP original columns:   {min(gdp_orig.columns)}–{max(gdp_orig.columns)}")
print(f"GDP backfilled columns: {min(gdp_back.columns)}–{max(gdp_back.columns)}")
print(f"Education columns:      {min(edu.columns)}–{max(edu.columns)}")
print(f"Subsistence backfill value: ${SUBSISTENCE_GDP}/capita (constant 2017 USD)\n")


# ── within-country R² (identical to LE script) ────────────────────────────────
def within_r2(predictor_col, outcome_col, rows):
    if len(rows) < MIN_OBS:
        return np.nan, 0, 0
    p = pd.DataFrame(rows)
    counts = p.groupby("country")[outcome_col].transform("count")
    p = p[counts >= MIN_OBS_PER_C].copy()
    if len(p) < MIN_OBS:
        return np.nan, 0, 0
    n_obs = len(p)
    n_ctry = p["country"].nunique()
    for col in [predictor_col, outcome_col]:
        p[col] = p[col] - p.groupby("country")[col].transform("mean")
    model = sm.OLS(p[outcome_col], p[[predictor_col]]).fit()
    return model.rsquared, n_obs, n_ctry


# ── outcome spec ──────────────────────────────────────────────────────────────
# (key, label, dataframe-or-None, log?, use_edu_as_outcome?)
OUTCOMES = [
    ("le",    "Life expectancy",          le_df,  False, False),
    ("tfr",   "Total fertility rate",     tfr_df, False, False),
    ("u5log", "Under-5 mortality (log)",  u5_df,  True,  False),
    ("cedu",  "Child education",          None,   False, True),
]


def get_outcome_value(key, outcome_df, log_outcome, use_edu_as_outcome, country, yr):
    """Return outcome value for (country, yr), honouring log + edu-as-outcome flags."""
    if use_edu_as_outcome:
        if yr not in edu.columns:
            return np.nan
        v = edu.loc[country, yr]
    else:
        if yr not in outcome_df.columns:
            return np.nan
        v = outcome_df.loc[country, yr]
    if pd.isna(v):
        return np.nan
    if log_outcome:
        if v <= 0:
            return np.nan
        return float(np.log(v))
    return float(v)


# ── main sweep: for each outcome × lag, compute all 3 R² ──────────────────────
lags = list(range(LAG_MIN, LAG_MAX + 1, LAG_STEP))

# per-outcome result dicts
results = {key: {"edu": [], "gdp_orig": [], "gdp_back": []} for key, *_ in OUTCOMES}

print(f"Sweeping {len(lags)} lags × {len(OUTCOMES)} outcomes "
      f"× 3 predictors (edu, GDP-orig, GDP-backfill)...\n")

for key, label, outcome_df, logit, use_edu in OUTCOMES:
    print(f"\n### OUTCOME: {label}  (key='{key}')")
    for lag in lags:
        edu_rows, gdp_orig_rows, gdp_back_rows = [], [], []

        for country in countries:
            for yr in range(PANEL_START, PANEL_END + 1):
                out_val = get_outcome_value(key, outcome_df, logit, use_edu,
                                             country, yr)
                if np.isnan(out_val):
                    continue
                yr_pred = yr - lag

                # education predictor (WCDE, 1875–2015)
                if yr_pred in edu.columns:
                    ev = edu.loc[country, yr_pred]
                    if not np.isnan(ev):
                        edu_rows.append({"country": country, "edu": ev,
                                         "y": out_val})

                # original GDP (WDI, 1960+), capped at GDP_LAG_MAX
                if lag <= GDP_LAG_MAX and yr_pred in gdp_orig.columns:
                    gv = gdp_orig.loc[country, yr_pred]
                    if not np.isnan(gv) and gv > 0:
                        gdp_orig_rows.append({"country": country,
                                              "log_gdp": np.log(gv),
                                              "y": out_val})

                # backfilled GDP (1870+), no lag cap
                if yr_pred in gdp_back.columns:
                    gv = gdp_back.loc[country, yr_pred]
                    if not np.isnan(gv) and gv > 0:
                        gdp_back_rows.append({"country": country,
                                              "log_gdp": np.log(gv),
                                              "y": out_val})

        re, ne, nce = within_r2("edu",     "y", edu_rows)
        ro, no, nco = within_r2("log_gdp", "y", gdp_orig_rows)
        rb, nb, ncb = within_r2("log_gdp", "y", gdp_back_rows)

        results[key]["edu"].append(re)
        results[key]["gdp_orig"].append(ro)
        results[key]["gdp_back"].append(rb)

        ro_s = "   nan" if np.isnan(ro) else f"{ro:>5.3f}"
        print(f"  lag={lag:3d}  edu R²={re:>6.3f} (n={ne:5d}, {nce:3d} ctry)  "
              f"gdp_orig R²={ro_s} (n={no:5d}, {nco:3d} ctry)  "
              f"gdp_back R²={rb:>5.3f} (n={nb:5d}, {ncb:3d} ctry)")


# ── summary tables per outcome ────────────────────────────────────────────────
print("\n" + "=" * 90)
print("PER-OUTCOME SUMMARY: Edu R² vs GDP-orig vs GDP-backfill across lags")
print("=" * 90)

for key, label, *_ in OUTCOMES:
    print(f"\n--- {label} (key='{key}') ---")
    print(f"{'Lag':>5}  {'Edu R²':>8}  {'GDP orig':>10}  {'GDP backfill':>12}  {'Delta':>8}")
    print("-" * 55)
    for lag, re, ro, rb in zip(lags,
                                results[key]["edu"],
                                results[key]["gdp_orig"],
                                results[key]["gdp_back"]):
        marker = {25: " <- 1 gen", 50: " <- 2 gen",
                  75: " <- 3 gen", 100: " <- 4 gen"}.get(lag, "")
        re_s = f"{re:>8.3f}" if not np.isnan(re) else "     nan"
        ro_s = f"{ro:>10.3f}" if not np.isnan(ro) else "       nan"
        rb_s = f"{rb:>12.3f}" if not np.isnan(rb) else "         nan"
        if not np.isnan(ro) and not np.isnan(rb):
            d_s = f"{(rb - ro):>+8.3f}"
        elif np.isnan(ro) and not np.isnan(rb):
            d_s = "   (new)"
        else:
            d_s = "        "
        print(f"{lag:>5}  {re_s}  {ro_s}  {rb_s}  {d_s}{marker}")


# ── collapse-to-zero detection ────────────────────────────────────────────────
def first_lag_below(threshold, lags, curve):
    """First lag at which `curve` falls strictly below `threshold`.

    Returns None if the curve never falls below the threshold within the
    swept lag range.
    """
    for lag, v in zip(lags, curve):
        if not np.isnan(v) and v < threshold:
            return lag
    return None


print("\n" + "=" * 90)
print(f"COLLAPSE POINT: first lag at which backfilled-GDP R² drops below {ZERO_R2_THRESHOLD}")
print("=" * 90)
print(f"{'Outcome':>25}  {'Edu R² @ lag100':>16}  {'GDP-bf R² @ lag100':>20}  "
      f"{'GDP-bf collapse lag':>22}")
print("-" * 90)

collapse_summary = {}
for key, label, *_ in OUTCOMES:
    edu_100 = results[key]["edu"][-1]
    bf_100  = results[key]["gdp_back"][-1]
    cl      = first_lag_below(ZERO_R2_THRESHOLD, lags, results[key]["gdp_back"])
    collapse_summary[key] = {
        "edu_r2_lag100": None if np.isnan(edu_100) else round(edu_100, 3),
        "gdp_back_r2_lag100": None if np.isnan(bf_100) else round(bf_100, 3),
        "gdp_back_collapse_lag": cl,
    }
    cl_s = f"{cl}" if cl is not None else "never"
    print(f"{label:>25}  {edu_100:>16.3f}  {bf_100:>20.3f}  {cl_s:>22}")

print("\nInterpretation:")
print("  If every outcome shows backfilled-GDP R² collapsing to < 0.01 while")
print("  education stays informative at lag 100, the LE-only backfill defense")
print("  generalises: GDP's long-lag weakness is not a missing-data artifact.")


# ── checkin JSON ──────────────────────────────────────────────────────────────
numbers = {
    "n_countries": len(countries),
    "lags": lags,
    "subsistence_gdp_usd_2017": SUBSISTENCE_GDP,
    "backfill_start_year": BACKFILL_START,
    "backfill_end_year": BACKFILL_END,
    "gdp_orig_lag_cap": GDP_LAG_MAX,
    "zero_r2_threshold": ZERO_R2_THRESHOLD,
}

for key, label, *_ in OUTCOMES:
    for i, lag in enumerate(lags):
        for pred_name in ("edu", "gdp_orig", "gdp_back"):
            v = results[key][pred_name][i]
            numbers[f"{key}_{pred_name}_r2_lag{lag}"] = (
                None if np.isnan(v) else round(v, 3)
            )

numbers["collapse_summary"] = collapse_summary

write_checkin(
    "backfill_all_outcomes.json",
    {
        "notes": (
            f"{len(countries)} countries. For each of 4 outcomes (LE, TFR, "
            f"U-5 log, child education), sweep lag 0-100 step 5 with three "
            f"predictors: education (WCDE 1875-2015), GDP (WDI, capped at "
            f"lag {GDP_LAG_MAX}), and backfilled GDP (WDI 1960+ merged with "
            f"${SUBSISTENCE_GDP}/cap for {BACKFILL_START}-{BACKFILL_END}, "
            f"no lag cap). Within-country fixed effects; outcome panel "
            f"{PANEL_START}-{PANEL_END}."
        ),
        "numbers": numbers,
    },
    script_path="scripts/robustness/backfill_all_outcomes.py",
)

print("\nDone.")
