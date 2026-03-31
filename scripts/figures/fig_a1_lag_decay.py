"""
figures/fig_a1_lag_decay.py

Generates Figure A1 for:
  "Education of Nations"

Output:
  paper/fig_a1_lag_decay.png

What it does:
  For each lag L from 0 to 100 years (step 5):
    - Outcome: life expectancy at year T (WDI, 1960–2015)
    - Predictor A: lower secondary completion at year T-L  (WCDE v3, 1875–2015)
    - Predictor B: log GDP per capita at year T-L          (WDI, 1960–2015)
    - Country fixed effects: demean each variable by country mean
    - Each predictor uses its own panel (max available data)
    - Record within-country R² for each predictor separately

  Plots both R² curves against lag length.
  Marks generational horizons at lag 25, 50, 75 on the x-axis.

Methodology note:
  Education and GDP use separate panels because education (WCDE) has data back to
  1875 while GDP (WDI) starts in 1960. Requiring both simultaneously would discard
  early observations where education data is available but GDP data is not.
  Within-country R² = reg.score() on country-demeaned data.

Verified output values (within-country R², country FE, separate panels):
  Education (169 countries, WCDE 1875-2015, all lags):
    lag 0: 0.528,  lag 25: 0.330,  lag 50: 0.158,  lag 75: 0.096,  lag 100: 0.050
  GDP (WDI 1960-2015, plotted only through lag 25; beyond that sample shrinks
  to high-income countries, causing selection bias):
    lag 0: 0.355,  lag 5: 0.287,  lag 10: 0.210,  lag 15: 0.140,  lag 20: 0.101,
    lag 25: 0.108

Key parameters:
  LAG_MIN    = 0    # years
  LAG_MAX    = 100  # years
  LAG_STEP   = 5    # years
  PANEL_START = 1960 # first outcome year (WDI LE coverage)
  PANEL_END  = 2015 # last outcome year
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import PROC, DATA, CHECKIN, REGIONS, REPO_ROOT, write_checkin

WCDE_PROC   = PROC
WB_DIR      = DATA
OUT         = os.path.join(REPO_ROOT, "paper", "fig_a1_lag_decay.png")

# ── parameters ────────────────────────────────────────────────────────────────
LAG_MIN     = 0
LAG_MAX     = 100
LAG_STEP    = 5
PANEL_START = 1960   # WDI life expectancy starts here
PANEL_END   = 2015   # last outcome year

# ── load data ─────────────────────────────────────────────────────────────────
def load_wide(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.str.lower().str.strip()
    df.columns = df.columns.astype(int)
    return df.clip(lower=0)

# WCDE v3 cohort completion (5-year intervals), interpolated to annual
_edu_raw = pd.read_csv(os.path.join(WCDE_PROC, "cohort_lower_sec_both.csv"), index_col="country")
_edu_raw.columns = [int(c) for c in _edu_raw.columns]
_all_yrs = list(range(min(_edu_raw.columns), max(_edu_raw.columns) + 1))
_edu_raw = _edu_raw.reindex(columns=_all_yrs).interpolate(axis=1).bfill(axis=1).ffill(axis=1)
_edu_raw.index = _edu_raw.index.str.lower().str.strip()
edu = _edu_raw.clip(lower=0)

gdp = load_wide(os.path.join(WB_DIR, "gdppercapita_us_inflation_adjusted.csv"))  # WDI
le  = load_wide(os.path.join(WB_DIR, "life_expectancy_years.csv"))               # WDI

# common countries present in all three datasets
countries = sorted(set(edu.index) & set(gdp.index) & set(le.index))
print(f"N countries: {len(countries)}")


MIN_OBS        = 500   # minimum observations for a valid R² estimate
MIN_OBS_PER_C  = 3     # drop countries with < this many obs (avoids FE overfitting)
GDP_LAG_MAX    = 25    # WDI data limit: beyond this lag the GDP sample shrinks and
                       # shifts toward high-income countries (sample-selection bias)


def within_r2(predictor_col, outcome_col, rows):
    """Within-country (FE-demeaned) R² using reg.score().

    rows: list of dicts with keys 'country', predictor_col, outcome_col
    Countries with fewer than MIN_OBS_PER_C observations are excluded to
    avoid overfitting when country means are estimated from very few points.
    """
    if len(rows) < MIN_OBS:
        return np.nan
    p = pd.DataFrame(rows)
    # drop countries with too few observations (avoids FE overfitting)
    counts = p.groupby("country")[outcome_col].transform("count")
    p = p[counts >= MIN_OBS_PER_C].copy()
    if len(p) < MIN_OBS:
        return np.nan
    for col in [predictor_col, outcome_col]:
        p[col] = p[col] - p.groupby("country")[col].transform("mean")
    reg = sm.OLS(p[outcome_col], p[[predictor_col]]).fit()
    return reg.rsquared


# ── compute R² at each lag ────────────────────────────────────────────────────
lags   = list(range(LAG_MIN, LAG_MAX + 1, LAG_STEP))
r2_edu = []
r2_gdp = []

print(f"Computing within-country R² at {len(lags)} lag values...")

for lag in lags:
    edu_rows = []
    gdp_rows = []

    for country in countries:
        for yr in range(PANEL_START, PANEL_END + 1):
            le_val = le.loc[country, yr] if yr in le.columns else np.nan
            if np.isnan(le_val):
                continue
            yr_pred = yr - lag

            # education predictor (WCDE, available 1875–2015)
            if yr_pred in edu.columns:
                edu_val = edu.loc[country, yr_pred]
                if not np.isnan(edu_val):
                    edu_rows.append({"country": country, "edu": edu_val, "le": le_val})

            # GDP predictor (WDI, available 1960–2015)
            # Skip beyond GDP_LAG_MAX to avoid sample-selection bias
            if lag <= GDP_LAG_MAX and yr_pred in gdp.columns:
                gdp_val = gdp.loc[country, yr_pred]
                if not np.isnan(gdp_val) and gdp_val > 0:
                    gdp_rows.append({"country": country, "log_gdp": np.log(gdp_val),
                                     "le": le_val})

    re = within_r2("edu",     "le", edu_rows)
    rg = within_r2("log_gdp", "le", gdp_rows)
    r2_edu.append(re)
    r2_gdp.append(rg)

    nc_e = len({r["country"] for r in edu_rows})
    nc_g = len({r["country"] for r in gdp_rows})
    print(f"  lag={lag:3d}  edu R²={re:.3f}  gdp R²={rg:.3f}  "
          f"n_edu={len(edu_rows)} ({nc_e} countries)  "
          f"n_gdp={len(gdp_rows)} ({nc_g} countries)")


# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(lags, r2_edu, color="#2166ac", linewidth=2.5,
        label="Lower secondary completion (education)")

# GDP plotted only where data is comparable; beyond lag 25 WDI coverage shrinks
r2_gdp_plot = [rg if (not np.isnan(rg)) else np.nan for rg in r2_gdp]
ax.plot(lags, r2_gdp_plot, color="#d6604d", linewidth=2.5, linestyle="--",
        label=f"Log GDP per capita (income; data to lag {GDP_LAG_MAX})")

# position generational markers after axes are set
plt.tight_layout()
ax.set_xlabel("Lag (years)", fontsize=12)
ax.set_ylabel("Within-country R² (country fixed effects)", fontsize=12)
ax.set_title(
    "Figure 4. Education vs. Income Predictive Power Across Lag Lengths\n"
    "Outcome: life expectancy at birth",
    fontsize=12,
)
ax.legend(fontsize=10)
ax.set_xlim(LAG_MIN, LAG_MAX)
ax.set_ylim(0, None)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)

# generational markers
ymax = ax.get_ylim()[1]
for lag_mark, label in [(25, "1 generation\n(lag 25)"), (50, "2 generations\n(lag 50)"), (75, "3 generations\n(lag 75)")]:
    ax.axvline(lag_mark, color="grey", linewidth=0.8, linestyle=":")
    ax.text(lag_mark + 1, ymax * 0.02, label, fontsize=8, color="grey", va="bottom")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT}")

# ── summary table ─────────────────────────────────────────────────────────────
print("\nKey values:")
print(f"{'Lag':>5}  {'Edu R²':>8}  {'GDP R²':>8}")
for lag, re, rg in zip(lags, r2_edu, r2_gdp):
    marker = (" ← " + {25: "1 gen", 50: "2 gen", 75: "3 gen"}.get(lag, "")
              if lag in (25, 50, 75) else "")
    re_str = f"{re:>8.3f}" if not np.isnan(re) else "     nan"
    rg_str = f"{rg:>8.3f}" if not np.isnan(rg) else "     nan"
    print(f"{lag:>5}  {re_str}  {rg_str}{marker}")

# ── Write checkin JSON ───────────────────────────────────────────────────────

# Build numbers dict with all lag values
lag_numbers = {}
for lag_val, re_val, rg_val in zip(lags, r2_edu, r2_gdp):
    lag_numbers[f"edu_r2_lag{lag_val}"] = round(re_val, 3) if not np.isnan(re_val) else None
    if not np.isnan(rg_val):
        lag_numbers[f"gdp_r2_lag{lag_val}"] = round(rg_val, 3)
lag_numbers["n_countries"] = len(countries)

# Add verify-script keys
lag_numbers["FA1-lag0"] = round(r2_edu[lags.index(0)], 3)
lag_numbers["FA1-lag25"] = round(r2_edu[lags.index(25)], 3)
lag_numbers["FA1-lag50"] = round(r2_edu[lags.index(50)], 3)
lag_numbers["FA1-lag75"] = round(r2_edu[lags.index(75)], 3)

write_checkin("fig_a1_lag_decay.json", {
    "notes": f"{len(countries)} countries. Education uses WCDE 1875-2015; GDP uses WDI 1960-2015. GDP only plotted through lag {GDP_LAG_MAX}.",
    "numbers": lag_numbers,
}, script_path="scripts/figures/fig_a1_lag_decay.py")
