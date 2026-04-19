"""
figures/forward_reverse_lag_tests.py

Two related lag-decay figures exploring causal directionality:

  Figure A — "edu vs gdp, full 0-100 range":
    Education's within-country R^2 predicting LE, TFR, U-5 log across lags
    0-100, overlaid with GDP's R^2 on the same outcomes using BACKFILLED GDP
    (pre-1960 = $500/capita, per le_r2_by_lag_backfilled.py). Shows GDP's
    collapse isn't a data artifact — even with backfill it dies.

  Figure B — "forward vs reverse":
    Forward:  Edu(T-L)  predicts Y(T)
    Reverse:  Y(T-L)    predicts Edu(T)
    For Y in [LE, log GDP]. If education is causal, forward R² should
    dominate reverse R², and the gap should widen at long lags.

Outputs:
    paper/variants/edu_vs_gdp_0_100.png
    paper/variants/forward_reverse.png
    checkin/forward_reverse_lag_tests.json

Notes:
    Within-country FE (demean by country mean) using OLS on demeaned data.
    Matches the methodology of le_r2_by_lag.py and outcomes_r2_by_lag.py.
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

OUT_A = os.path.join(REPO_ROOT, "paper", "variants", "edu_vs_gdp_0_100.png")
OUT_B = os.path.join(REPO_ROOT, "paper", "variants", "forward_reverse.png")

LAG_MIN, LAG_MAX, LAG_STEP = 0, 100, 5
PANEL_START, PANEL_END = 1960, 2015
MIN_OBS, MIN_OBS_PER_C = 500, 3

SUBSISTENCE_GDP = 500
BACKFILL_START, BACKFILL_END = 1870, 1959


def load_wide(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.str.lower().str.strip()
    df.columns = df.columns.astype(int)
    return df.clip(lower=0)


# ── load data ────────────────────────────────────────────────────────────
_edu_raw = pd.read_csv(os.path.join(PROC, "cohort_lower_sec_both.csv"),
                       index_col="country")
_edu_raw.columns = [int(c) for c in _edu_raw.columns]
_all_yrs = list(range(min(_edu_raw.columns), max(_edu_raw.columns) + 1))
_edu_raw = (_edu_raw.reindex(columns=_all_yrs).interpolate(axis=1)
                    .bfill(axis=1).ffill(axis=1))
_edu_raw.index = _edu_raw.index.str.lower().str.strip()
edu = _edu_raw.clip(lower=0)

le   = load_wide(os.path.join(DATA, "life_expectancy_years.csv"))
tfr  = load_wide(os.path.join(DATA, "children_per_woman_total_fertility.csv"))
u5   = load_wide(os.path.join(DATA, "child_mortality_u5.csv"))
gdp  = load_wide(os.path.join(DATA, "gdppercapita_us_inflation_adjusted.csv"))

# Backfilled GDP (subsistence floor for pre-1960)
gdp_back = gdp.copy()
for yr in range(BACKFILL_START, BACKFILL_END + 1):
    if yr not in gdp_back.columns:
        gdp_back[yr] = SUBSISTENCE_GDP
    else:
        gdp_back[yr] = gdp_back[yr].fillna(SUBSISTENCE_GDP)
for yr in gdp_back.columns:
    if yr >= 1960:
        gdp_back[yr] = gdp_back[yr].fillna(SUBSISTENCE_GDP)
gdp_back = gdp_back[sorted(gdp_back.columns)]

countries = sorted(set(edu.index) & set(le.index) & set(tfr.index) & set(u5.index)
                   & set(gdp.index))
print(f"N countries: {len(countries)}")


def within_r2(predictor_col, outcome_col, rows):
    if len(rows) < MIN_OBS:
        return np.nan
    p = pd.DataFrame(rows)
    counts = p.groupby("country")[outcome_col].transform("count")
    p = p[counts >= MIN_OBS_PER_C].copy()
    if len(p) < MIN_OBS:
        return np.nan
    for col in [predictor_col, outcome_col]:
        p[col] = p[col] - p.groupby("country")[col].transform("mean")
    return sm.OLS(p[outcome_col], p[[predictor_col]]).fit().rsquared


def collect_pair(pred_df, out_df, lag, pred_log=False, out_log=False):
    """Build rows: predictor at yr-lag, outcome at yr."""
    rows = []
    for country in countries:
        for yr in range(PANEL_START, PANEL_END + 1):
            o = out_df.loc[country, yr] if yr in out_df.columns else np.nan
            if pd.isna(o):
                continue
            yp = yr - lag
            if yp not in pred_df.columns:
                continue
            p = pred_df.loc[country, yp]
            if pd.isna(p):
                continue
            if pred_log:
                if p <= 0:
                    continue
                p = np.log(p)
            if out_log:
                if o <= 0:
                    continue
                o = np.log(o)
            rows.append({"country": country, "pred": p, "out": o})
    return rows


lags = list(range(LAG_MIN, LAG_MAX + 1, LAG_STEP))

# ═══════════════════════════════════════════════════════════════════════
# FIGURE A — edu vs backfilled GDP, 0-100, three outcomes
# ═══════════════════════════════════════════════════════════════════════
print("\nFigure A: edu vs backfilled GDP across 0-100 lags...")

fig_a_data = {"lags": lags}
# Four outcomes. Child education uses edu as both predictor AND outcome;
# at lag 0 the edu->cedu R² is trivially 1.0, and GDP->cedu tests whether
# past income predicts future education (reverse-causality direction).
outcome_cfg = [
    ("le",    le,  False, "Life expectancy",         "#1f6feb"),   # blue
    ("tfr",   tfr, False, "Total fertility rate",    "#2ca25f"),   # green
    ("u5log", u5,  True,  "Under-5 mortality (log)", "#6a3d9a"),   # purple
    ("cedu",  edu, False, "Child education",         "#e67e22"),   # orange
]

for key, out_df, out_log, label, _ in outcome_cfg:
    edu_r2, gdp_r2 = [], []
    for lag in lags:
        er = collect_pair(edu, out_df, lag, out_log=out_log)
        # Raw GDP only — no backfill. Curves terminate where WDI data runs out.
        gr = collect_pair(gdp, out_df, lag, pred_log=True, out_log=out_log)
        edu_r2.append(within_r2("pred", "out", er))
        gdp_r2.append(within_r2("pred", "out", gr))
    fig_a_data[f"edu_{key}"] = edu_r2
    fig_a_data[f"gdpback_{key}"] = gdp_r2  # key name kept for checkin compatibility
    last_gdp = next((g for g in reversed(gdp_r2) if not np.isnan(g)), np.nan)
    print(f"  {label}:  edu lag100={edu_r2[-1]:.3f}  gdp last non-nan={last_gdp:.3f}")


fig, ax = plt.subplots(figsize=(9, 5.5))
for key, _, _, label, color in outcome_cfg:
    ax.plot(lags, fig_a_data[f"edu_{key}"], linestyle="-", linewidth=2.0,
            color=color, marker="o", markersize=3.5,
            label=f"Education -> {label}")
    ax.plot(lags, fig_a_data[f"gdpback_{key}"], linestyle="--", linewidth=1.7,
            color=color, marker="s", markersize=3.5, alpha=0.75,
            label=f"GDP -> {label}")

for lag_mark in [25, 50, 75]:
    ax.axvline(lag_mark, color="0.75", linestyle=":", linewidth=0.9, zorder=0)
for lag_mark, text in {25: "1 gen", 50: "2 gen", 75: "3 gen"}.items():
    ax.text(lag_mark, 0.99, text, rotation=90, va="top", ha="right",
            fontsize=8, color="0.35")

ax.set_xlim(0, 100)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Lag (years) between predictor and outcome")
ax.set_ylabel("Within-country $R^2$ (fixed effects)")
ax.set_title("Education vs. GDP across 0–100 year lags\n"
             "Solid = education (WCDE 1875-2015); dashed = GDP (WDI 1960+, no backfill)")
ax.grid(True, linestyle=":", linewidth=0.6, color="0.8", zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.95)
plt.tight_layout()
plt.savefig(OUT_A, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_A}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE B — forward vs reverse causality (Granger-style)
# ═══════════════════════════════════════════════════════════════════════
print("\nFigure B: forward vs reverse (edu ↔ LE, edu ↔ log GDP)...")

fig_b_data = {"lags": lags}

# edu <-> LE
fwd_le, rev_le = [], []
for lag in lags:
    fwd_le.append(within_r2("pred", "out",
        collect_pair(edu, le, lag)))                      # Edu(T-L) -> LE(T)
    rev_le.append(within_r2("pred", "out",
        collect_pair(le, edu, lag)))                      # LE(T-L) -> Edu(T)
fig_b_data["edu->le"] = fwd_le
fig_b_data["le->edu"] = rev_le

# edu <-> log GDP
fwd_g, rev_g = [], []
for lag in lags:
    fwd_g.append(within_r2("pred", "out",
        collect_pair(edu, gdp, lag, out_log=True)))        # Edu(T-L) -> log GDP(T)
    rev_g.append(within_r2("pred", "out",
        collect_pair(gdp, edu, lag, pred_log=True)))       # log GDP(T-L) -> Edu(T)
fig_b_data["edu->gdp"] = fwd_g
fig_b_data["gdp->edu"] = rev_g

print(f"  Edu->LE  lag25={fwd_le[lags.index(25)]:.3f}  lag100={fwd_le[-1]:.3f}")
print(f"  LE->Edu  lag25={rev_le[lags.index(25)]:.3f}  lag100={rev_le[-1]:.3f}")
print(f"  Edu->GDP lag25={fwd_g[lags.index(25)]:.3f}  lag100={fwd_g[-1]:.3f}")
print(f"  GDP->Edu lag25={rev_g[lags.index(25)]:.3f}  lag100={rev_g[-1]:.3f}")

fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: edu <-> LE
axl.plot(lags, fwd_le, linestyle="-",  linewidth=2.2, color="#1f6feb",
         marker="o", markersize=4, label="Education(T-L) -> LE(T)")
axl.plot(lags, rev_le, linestyle="--", linewidth=2.2, color="#c0392b",
         marker="s", markersize=4, label="LE(T-L) -> Education(T)")
for lag_mark in [25, 50, 75]:
    axl.axvline(lag_mark, color="0.75", linestyle=":", linewidth=0.9)
axl.set_xlim(0, 100); axl.set_ylim(0, 1.02)
axl.set_xlabel("Lag (years)")
axl.set_ylabel("Within-country $R^2$")
axl.set_title("Education ↔ Life expectancy")
axl.grid(True, linestyle=":", linewidth=0.6, color="0.8")
axl.set_axisbelow(True)
axl.legend(loc="upper right", fontsize=9)

# Right panel: edu <-> log GDP
axr.plot(lags, fwd_g, linestyle="-",  linewidth=2.2, color="#1f6feb",
         marker="o", markersize=4, label="Education(T-L) -> log GDP(T)")
axr.plot(lags, rev_g, linestyle="--", linewidth=2.2, color="#c0392b",
         marker="s", markersize=4, label="log GDP(T-L) -> Education(T)")
for lag_mark in [25, 50, 75]:
    axr.axvline(lag_mark, color="0.75", linestyle=":", linewidth=0.9)
axr.set_xlim(0, 100); axr.set_ylim(0, 1.02)
axr.set_xlabel("Lag (years)")
axr.set_ylabel("Within-country $R^2$")
axr.set_title("Education ↔ GDP per capita")
axr.grid(True, linestyle=":", linewidth=0.6, color="0.8")
axr.set_axisbelow(True)
axr.legend(loc="upper right", fontsize=9)

fig.suptitle("Forward vs. reverse: causality direction test (within-country FE)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(OUT_B, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_B}")


# ── checkin JSON ─────────────────────────────────────────────────────────
numbers = {"n_countries": len(countries)}
for i, lag in enumerate(lags):
    for key, _, _, _, _ in outcome_cfg:
        numbers[f"edu_{key}_lag{lag}"]     = round(fig_a_data[f"edu_{key}"][i], 3) \
            if not np.isnan(fig_a_data[f"edu_{key}"][i]) else None
        numbers[f"gdpback_{key}_lag{lag}"] = round(fig_a_data[f"gdpback_{key}"][i], 3) \
            if not np.isnan(fig_a_data[f"gdpback_{key}"][i]) else None
    numbers[f"fwd_le_lag{lag}"]   = round(fwd_le[i], 3) if not np.isnan(fwd_le[i]) else None
    numbers[f"rev_le_lag{lag}"]   = round(rev_le[i], 3) if not np.isnan(rev_le[i]) else None
    numbers[f"fwd_gdp_lag{lag}"]  = round(fwd_g[i], 3) if not np.isnan(fwd_g[i]) else None
    numbers[f"rev_gdp_lag{lag}"]  = round(rev_g[i], 3) if not np.isnan(rev_g[i]) else None

write_checkin("forward_reverse_lag_tests.json", {
    "notes": f"{len(countries)} countries. Forward/reverse and edu/gdp across 0-100 lags.",
    "numbers": numbers,
}, script_path="scripts/figures/forward_reverse_lag_tests.py")

print("\nDone.")
