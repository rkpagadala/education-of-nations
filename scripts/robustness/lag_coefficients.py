"""
lag_coefficients.py

Companion to backfill_all_outcomes.py: at each lag L in {0, 5, ..., 100}
reports the standardized within-country regression coefficient (beta) of
lower-secondary completion on each of four development outcomes, along
with its standard error and t-statistic.

Motivation:
  R^2 conflates effect size with predictor variance and outcome noise and
  is not directly comparable across outcomes. The standardized beta -- the
  expected outcome change (in outcome SDs) for a one-SD change in the
  predictor -- is the causal quantity and is comparable across outcomes.
  The t-statistic is the clean test of whether a signal is present.

Method (mirrors backfill_all_outcomes.py for data alignment):
  - Education: WCDE v3 lower-secondary completion, 1875-2015 (cohort, both
    sexes), interpolated to annual.
  - Outcomes: life expectancy (WDI), total fertility (WDI), under-5
    mortality logged (WDI), child education (WCDE, autoregression of the
    predictor).
  - Country fixed effects: demean every series by country mean.
  - Standardize both predictor and outcome by pooled within-country SD
    (SD of the demeaned series), then regress outcome on predictor with
    no intercept (FE absorbed by demean; standardized so intercept = 0).
  - Panel: 142 common countries (intersection of edu / GDP / LE / TFR / U-5),
    outcome years 1960-2015. Same as backfill_all_outcomes.py.

Output:
  checkin/lag_coefficients.json
    beta_<outcome>_lag<L>        standardized coefficient
    se_<outcome>_lag<L>          standard error (classical OLS)
    t_<outcome>_lag<L>           t-statistic
    n_<outcome>_lag<L>           observations
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import PROC as WCDE_PROC, DATA as WB_DIR, write_checkin

LAG_MIN     = 0
LAG_MAX     = 100
LAG_STEP    = 5
PANEL_START = 1960
PANEL_END   = 2015

MIN_OBS       = 500
MIN_OBS_PER_C = 3


def load_wide(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.str.lower().str.strip()
    df.columns = df.columns.astype(int)
    return df.clip(lower=0)


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

countries = sorted(
    set(edu.index)
    & set(gdp_orig.index)
    & set(le_df.index)
    & set(tfr_df.index)
    & set(u5_df.index)
)
print(f"N countries: {len(countries)}")


def _mat(df, cols):
    return df.reindex(index=countries, columns=cols).to_numpy(dtype=float, na_value=np.nan)


def _log_nonpos_to_nan(mat):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mat > 0, np.log(mat), np.nan)


def standardized_beta(pred, out, country_labels):
    valid = ~np.isnan(pred) & ~np.isnan(out)
    if valid.sum() < MIN_OBS:
        return np.nan, np.nan, np.nan, 0
    df = pd.DataFrame({"country": country_labels[valid],
                       "p": pred[valid], "o": out[valid]})
    cnt = df.groupby("country")["o"].transform("count")
    df = df[cnt >= MIN_OBS_PER_C]
    if len(df) < MIN_OBS:
        return np.nan, np.nan, np.nan, 0
    df["p"] = df["p"] - df.groupby("country")["p"].transform("mean")
    df["o"] = df["o"] - df.groupby("country")["o"].transform("mean")
    sd_p = df["p"].std(ddof=0)
    sd_o = df["o"].std(ddof=0)
    if sd_p == 0 or sd_o == 0:
        return np.nan, np.nan, np.nan, int(len(df))
    df["p"] = df["p"] / sd_p
    df["o"] = df["o"] / sd_o
    model = sm.OLS(df["o"], df[["p"]]).fit()
    beta = float(model.params["p"])
    se   = float(model.bse["p"])
    t    = float(model.tvalues["p"])
    return beta, se, t, int(len(df))


OUTCOMES = [
    ("le",    "Life expectancy",          le_df,  False, False),
    ("tfr",   "Total fertility rate",     tfr_df, False, False),
    ("u5log", "Under-5 mortality (log)",  u5_df,  True,  False),
    ("cedu",  "Child education",          None,   False, True),
]

lags = list(range(LAG_MIN, LAG_MAX + 1, LAG_STEP))

_outcome_yrs = list(range(PANEL_START, PANEL_END + 1))
_country_labels_flat = np.repeat(np.array(countries, dtype=object), len(_outcome_yrs))

results = {key: {"beta": [], "se": [], "t": [], "n": []} for key, *_ in OUTCOMES}

print(f"\nSweeping {len(lags)} lags x {len(OUTCOMES)} outcomes...\n")

for key, label, outcome_df, logit, use_edu in OUTCOMES:
    print(f"\n### {label}")
    out_src = edu if use_edu else outcome_df
    out_mat = _mat(out_src, _outcome_yrs)
    if logit:
        out_mat = _log_nonpos_to_nan(out_mat)
    out_flat = out_mat.ravel()

    for lag in lags:
        pred_cols = [y - lag for y in _outcome_yrs]
        edu_pred = _mat(edu, pred_cols).ravel()
        b, s, t, n = standardized_beta(edu_pred, out_flat, _country_labels_flat)
        results[key]["beta"].append(b)
        results[key]["se"].append(s)
        results[key]["t"].append(t)
        results[key]["n"].append(n)
        b_s = f"{b:>+6.3f}" if not np.isnan(b) else "   nan"
        t_s = f"{t:>+7.2f}" if not np.isnan(t) else "    nan"
        print(f"  lag={lag:3d}  beta={b_s}  se={s:>5.3f}  t={t_s}  n={n}")


print("\n" + "=" * 90)
print("SUMMARY: standardized beta by outcome, at generation anchors")
print("=" * 90)
print(f"{'Lag':>5}  {'Gen':>4}  "
      f"{'LE beta (t)':>15}  {'TFR beta (t)':>15}  "
      f"{'U-5 beta (t)':>15}  {'CE beta (t)':>15}")
print("-" * 90)
for i, lag in enumerate(lags):
    if lag not in (0, 25, 50, 75, 100):
        continue
    gen = {0: "-", 25: "1", 50: "2", 75: "3", 100: "4"}[lag]
    parts = [f"{lag:>5}", f"{gen:>4}"]
    for key, *_ in OUTCOMES:
        b = results[key]["beta"][i]
        t = results[key]["t"][i]
        parts.append(f"{b:>+6.3f} ({t:>+6.2f})")
    print("  ".join(parts))


numbers = {
    "n_countries": len(countries),
    "lags": lags,
    "panel_start": PANEL_START,
    "panel_end": PANEL_END,
}
for key, *_ in OUTCOMES:
    for i, lag in enumerate(lags):
        for tag in ("beta", "se", "t", "n"):
            v = results[key][tag][i]
            if isinstance(v, (int, np.integer)):
                numbers[f"{key}_{tag}_lag{lag}"] = int(v)
            else:
                numbers[f"{key}_{tag}_lag{lag}"] = (
                    None if v is None or (isinstance(v, float) and np.isnan(v))
                    else round(float(v), 4)
                )

write_checkin(
    "lag_coefficients.json",
    {
        "notes": (
            f"{len(countries)} countries, outcome years {PANEL_START}-{PANEL_END}. "
            f"Standardized within-country OLS of development outcome on "
            f"lower-secondary completion at T-lag. Each series demeaned by "
            f"country mean and standardized by pooled within-country SD, then "
            f"regressed with no intercept. Panel matches backfill_all_outcomes.py."
        ),
        "numbers": numbers,
    },
    script_path="scripts/robustness/lag_coefficients.py",
)

print("\nDone.")
