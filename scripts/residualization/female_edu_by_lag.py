"""
residualization/female_edu_by_lag.py

Sweep lags 0-30 years to test whether the female-only vs both-sexes
within-R² ratio peaks at a different lag for each outcome.

Motivation: the main paper uses age 20-24 cohort education at T,
predicting outcomes at T+25. For under-5 mortality the actual mothers
are aged ~25-35 at the time the child is born — so at lag 25 the
measured cohort (who were 20-24 at T) are aged 45-49 at T+25, too old
to be the mothers. Shorter lags (5-15 years) match education measured
on women who are of child-bearing age at the outcome year.

For each lag L in {0, 5, 10, 15, 20, 25, 30}:
  - predictor: lower-secondary completion at T-L (both vs female)
  - outcomes: life expectancy, TFR, log U-5 mortality, child education
  - within-country FE (demeaned OLS)
  - report R² for both-sexes and female-only, plus ratio and delta

Output:
  paper/female_edu_by_lag.png
  checkin/female_edu_by_lag.json
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import PROC, DATA, REPO_ROOT, write_checkin

WCDE_PROC = PROC
WB_DIR    = DATA
OUT_FIG   = os.path.join(REPO_ROOT, "paper", "figures", "female_edu_by_lag.png")

LAG_MIN     = 0
LAG_MAX     = 30
LAG_STEP    = 5
PANEL_START = 1960
PANEL_END   = 2015
MIN_OBS       = 200
MIN_OBS_PER_C = 3


def load_wide(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.str.lower().str.strip()
    df.columns = df.columns.astype(int)
    return df.clip(lower=0)


def load_edu(filename):
    df = pd.read_csv(os.path.join(WCDE_PROC, filename), index_col="country")
    df.columns = [int(c) for c in df.columns]
    all_yrs = list(range(min(df.columns), max(df.columns) + 1))
    df = df.reindex(columns=all_yrs).interpolate(axis=1).bfill(axis=1).ffill(axis=1)
    df.index = df.index.str.lower().str.strip()
    return df.clip(lower=0)


edu_both   = load_edu("cohort_lower_sec_both.csv")
edu_female = load_edu("cohort_lower_sec_female.csv")

le   = load_wide(os.path.join(WB_DIR, "life_expectancy_years.csv"))
tfr  = load_wide(os.path.join(WB_DIR, "children_per_woman_total_fertility.csv"))
u5   = load_wide(os.path.join(WB_DIR, "child_mortality_u5.csv"))


def within_r2(rows, predictor_col="edu", outcome_col="out"):
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
    reg = sm.OLS(p[outcome_col], p[[predictor_col]]).fit()
    return reg.rsquared, n_obs, n_ctry


def collect_rows(edu_df, outcome_df, lag, countries,
                 log_outcome=False, use_edu_as_outcome=False):
    rows = []
    for country in countries:
        for yr in range(PANEL_START, PANEL_END + 1):
            if use_edu_as_outcome:
                out_val = edu_both.loc[country, yr] if yr in edu_both.columns else np.nan
            else:
                out_val = outcome_df.loc[country, yr] if yr in outcome_df.columns else np.nan
            if pd.isna(out_val):
                continue
            if log_outcome:
                if out_val <= 0:
                    continue
                out_val = np.log(out_val)
            yr_pred = yr - lag
            if yr_pred not in edu_df.columns:
                continue
            edu_val = edu_df.loc[country, yr_pred]
            if pd.isna(edu_val):
                continue
            rows.append({"country": country, "edu": edu_val, "out": out_val})
    return rows


OUTCOMES = [
    ("Life expectancy",         "le",    le,  False, False),
    ("Total fertility rate",    "tfr",   tfr, False, False),
    ("Under-5 mortality (log)", "u5log", u5,  True,  False),
    ("Child education",         "cedu",  None, False, True),
]

countries = sorted(set(edu_both.index) & set(edu_female.index)
                   & set(le.index) & set(tfr.index) & set(u5.index))
print(f"N countries (intersection): {len(countries)}")

lags = list(range(LAG_MIN, LAG_MAX + 1, LAG_STEP))

both_results   = {key: [] for (_, key, _, _, _) in OUTCOMES}
female_results = {key: [] for (_, key, _, _, _) in OUTCOMES}
both_n         = {key: [] for (_, key, _, _, _) in OUTCOMES}
female_n       = {key: [] for (_, key, _, _, _) in OUTCOMES}

print(f"\nLag sweep: {lags}\n")
print(f"{'Lag':>4}  {'Outcome':<24}  {'Both R²':>8}  {'Fem R²':>8}  {'Ratio':>6}  {'ΔR²':>7}  {'n':>5}")
print("-" * 80)

for lag in lags:
    for (label, key, odf, logit, use_edu) in OUTCOMES:
        rows_b = collect_rows(edu_both,   odf, lag, countries, log_outcome=logit, use_edu_as_outcome=use_edu)
        rows_f = collect_rows(edu_female, odf, lag, countries, log_outcome=logit, use_edu_as_outcome=use_edu)
        r2_b, n_b, _ = within_r2(rows_b)
        r2_f, n_f, _ = within_r2(rows_f)
        both_results[key].append(r2_b)
        female_results[key].append(r2_f)
        both_n[key].append(n_b)
        female_n[key].append(n_f)
        if not np.isnan(r2_b) and not np.isnan(r2_f) and r2_b > 0:
            ratio = r2_f / r2_b
            delta = r2_f - r2_b
            print(f"{lag:>4}  {label:<24}  {r2_b:>8.3f}  {r2_f:>8.3f}  {ratio:>6.3f}  {delta:>+7.3f}  {n_b:>5}")
        else:
            print(f"{lag:>4}  {label:<24}  {'nan':>8}  {'nan':>8}  {'---':>6}  {'---':>7}  {n_b:>5}")


PLOT_ORDER = [
    ("Under-5 mortality (log)",  "u5log", "#6a3d9a"),
    ("Child education",          "cedu",  "#e67e22"),
    ("Total fertility rate",     "tfr",   "#2ca25f"),
    ("Life expectancy",          "le",    "#1f6feb"),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True)

ax = axes[0]
for label, key, color in PLOT_ORDER:
    y_both = np.array(both_results[key], dtype=float)
    y_fem  = np.array(female_results[key], dtype=float)
    ax.plot(lags, y_both, linestyle="--", linewidth=1.6, marker="s", markersize=4,
            color=color, alpha=0.75, label=f"{label} — both")
    ax.plot(lags, y_fem,  linestyle="-",  linewidth=2.0, marker="o", markersize=4,
            color=color, label=f"{label} — female")
ax.set_xlabel("Lag (years) between predictor and outcome")
ax.set_ylabel("Within-country $R^2$")
ax.set_title("Education → outcome $R^2$ by lag\nSolid: female-only 20-24; Dashed: both-sexes 20-24",
             fontsize=10)
ax.grid(True, linestyle=":", linewidth=0.6, color="0.8", zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=7)

ax = axes[1]
for label, key, color in PLOT_ORDER:
    y_both = np.array(both_results[key], dtype=float)
    y_fem  = np.array(female_results[key], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(y_both > 0, y_fem / y_both, np.nan)
    ax.plot(lags, ratio, linestyle="-", linewidth=2.0, marker="o", markersize=4.5,
            color=color, label=label)
ax.axhline(1.0, color="0.5", linestyle=":", linewidth=0.9, zorder=0)
ax.set_xlabel("Lag (years) between predictor and outcome")
ax.set_ylabel("Ratio: female-only $R^2$ / both-sexes $R^2$")
ax.set_title("Female advantage by lag\n(>1 means female-only predicts better)",
             fontsize=10)
ax.grid(True, linestyle=":", linewidth=0.6, color="0.8", zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=8)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT_FIG}")


print("\n" + "=" * 90)
print("SUMMARY: female/both R² ratio at each lag")
print("=" * 90)
print(f"{'Lag':>4}  " + "  ".join(f"{key:>10}" for (_, key, _) in PLOT_ORDER))
print("-" * 64)
for i, lag in enumerate(lags):
    cells = []
    for _, key, _ in PLOT_ORDER:
        b = both_results[key][i]
        f = female_results[key][i]
        if b is None or np.isnan(b) or b == 0 or np.isnan(f):
            cells.append(f"{'nan':>10}")
        else:
            cells.append(f"{f/b:>10.3f}")
    print(f"{lag:>4}  " + "  ".join(cells))


peak_lag = {}
for _, key, _ in PLOT_ORDER:
    ratios = []
    for i in range(len(lags)):
        b = both_results[key][i]
        f = female_results[key][i]
        if b and not np.isnan(b) and not np.isnan(f) and b > 0:
            ratios.append((lags[i], f / b))
    if ratios:
        best_lag, best_ratio = max(ratios, key=lambda x: x[1])
        peak_lag[key] = {"lag": best_lag, "ratio": round(best_ratio, 3)}

print("\nPeak female/both ratio by outcome:")
for key, v in peak_lag.items():
    print(f"  {key:<8}  peak ratio = {v['ratio']:.3f} at lag {v['lag']}")


numbers = {
    "n_countries": len(countries),
    "lags": lags,
    "lag_min": LAG_MIN, "lag_max": LAG_MAX, "lag_step": LAG_STEP,
    "panel_start": PANEL_START, "panel_end": PANEL_END,
    "peak_by_outcome": peak_lag,
}
for (_, key, _, _, _) in OUTCOMES:
    for i, lag in enumerate(lags):
        b = both_results[key][i]
        f = female_results[key][i]
        numbers[f"{key}_both_r2_lag{lag}"]   = round(b, 3) if not np.isnan(b) else None
        numbers[f"{key}_female_r2_lag{lag}"] = round(f, 3) if not np.isnan(f) else None
        if b and not np.isnan(b) and not np.isnan(f) and b > 0:
            numbers[f"{key}_ratio_lag{lag}"] = round(f / b, 3)
            numbers[f"{key}_delta_lag{lag}"] = round(f - b, 3)
        else:
            numbers[f"{key}_ratio_lag{lag}"] = None
            numbers[f"{key}_delta_lag{lag}"] = None

write_checkin("female_edu_by_lag.json", {
    "notes": (f"{len(countries)} countries. For each lag L in [{LAG_MIN}, "
              f"{LAG_MAX}] step {LAG_STEP}, within-country FE R² regressing "
              f"each outcome on both-sexes vs female-only lower-secondary "
              f"completion (age 20-24) at T-L. Addresses the age-cohort "
              f"mismatch for U-5: mothers of U-5 deaths are ~25-35, so the "
              f"20-24 cohort at T-L matches mother-age when L is 5-15, not 25. "
              f"Panel 1960-2015."),
    "numbers": numbers,
}, script_path="scripts/residualization/female_edu_by_lag.py")

print("\nDone.")
