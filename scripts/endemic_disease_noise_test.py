"""
Test: Is the educational rupture rate an order of magnitude larger than
endemic disease attendance effects, even for slow-rupture countries?

Literature estimates of malaria attendance effects:
- Bleakley (2007): hookworm eradication raised attendance ~20% in affected areas
  of US South — but this is % of school days, not pp of completion
- Thuilliez et al. (2010): malaria reduces school attendance by ~4-10 days/year
  out of ~180-200 school days, i.e. ~2-5% of attendance
- Even generous upper bound: 5pp reduction in effective completion from endemic disease

If 5-year educational expansion rates exceed this by an order of magnitude (or even
several multiples), endemic disease is noise on the rupture signal.

Data source: wcde/data/processed/lower_sec_both.csv (WCDE v3, completion %)
"""

import os
import pandas as pd
import numpy as np

# ── Load data ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
WCDE_PROC = os.path.join(REPO_ROOT, "wcde", "data", "processed")

df = pd.read_csv(os.path.join(WCDE_PROC, "lower_sec_both.csv"), index_col="country")
df.columns = df.columns.astype(int)

# ── Focus on WCDE 5-year intervals 1975-2015 ──────────────────────
years = list(range(1975, 2016, 5))
df5 = df[[y for y in years if y in df.columns]]

# ── Compute 5-year gains ──────────────────────────────────────────
gains = pd.DataFrame()
for i in range(len(years) - 1):
    y0, y1 = years[i], years[i + 1]
    if y0 in df5.columns and y1 in df5.columns:
        col = f"{y0}-{y1}"
        gains[col] = df5[y1] - df5[y0]

# ── Filter to active expansion (baseline < 80%, gain > 0) ────────
# These are the countries actually making a rupture, not already-developed
active = []
for _, row in gains.iterrows():
    country = row.name
    for col in gains.columns:
        y0 = int(col.split("-")[0])
        baseline = df5.loc[country, y0] if y0 in df5.columns else None
        if baseline is not None and baseline < 80 and row[col] > 0:
            active.append({
                "country": country,
                "interval": col,
                "baseline": baseline,
                "gain_5yr": row[col],
            })

active_df = pd.DataFrame(active)

print("=" * 70)
print("5-YEAR EDUCATIONAL EXPANSION RATES vs ENDEMIC DISEASE EFFECTS")
print("=" * 70)
print()

# ── Distribution of 5-year gains ──────────────────────────────────
print("Distribution of 5-year completion gains (pp)")
print(f"  (countries with baseline < 80% and positive gain, N={len(active_df)})")
print()
percentiles = [10, 25, 50, 75, 90]
for p in percentiles:
    val = np.percentile(active_df["gain_5yr"], p)
    print(f"  P{p:2d}: {val:5.1f} pp / 5 years")
print(f"  Mean: {active_df['gain_5yr'].mean():5.1f} pp / 5 years")
print(f"  Min:  {active_df['gain_5yr'].min():5.1f} pp / 5 years")
print()

# ── Endemic disease attendance effect (generous upper bound) ──────
DISEASE_EFFECT_PP = 5.0  # generous upper bound in pp of completion
print(f"Generous upper bound for endemic disease attendance effect: {DISEASE_EFFECT_PP} pp")
print()

# ── Ratio ─────────────────────────────────────────────────────────
print("Ratio of educational expansion to disease effect:")
for p in percentiles:
    val = np.percentile(active_df["gain_5yr"], p)
    ratio = val / DISEASE_EFFECT_PP
    print(f"  P{p:2d}: {ratio:5.1f}x")
print(f"  Mean: {active_df['gain_5yr'].mean() / DISEASE_EFFECT_PP:5.1f}x")
print()

# ── Key slow-rupture countries ────────────────────────────────────
slow_countries = ["India", "Bangladesh", "Nepal", "Ethiopia", "Nigeria",
                  "Tanzania, United Republic of", "Kenya", "Uganda",
                  "Mozambique", "Mali", "Niger", "Chad", "Burkina Faso",
                  "Myanmar"]

print("=" * 70)
print("SLOW-RUPTURE COUNTRIES: 5-year gains by interval")
print("=" * 70)
print()

for c in slow_countries:
    if c in gains.index:
        cdata = []
        for col in gains.columns:
            y0 = int(col.split("-")[0])
            baseline = df5.loc[c, y0] if y0 in df5.columns else None
            gain = gains.loc[c, col]
            if baseline is not None and gain > 0:
                cdata.append((col, baseline, gain))
        if cdata:
            print(f"  {c}")
            for interval, baseline, gain in cdata:
                ratio = gain / DISEASE_EFFECT_PP
                print(f"    {interval}: baseline {baseline:5.1f}%, "
                      f"gain {gain:5.1f} pp  ({ratio:.1f}x disease effect)")
            print()

# ── How many country-intervals have gain < disease effect? ────────
below = active_df[active_df["gain_5yr"] < DISEASE_EFFECT_PP]
print("=" * 70)
pct_below = len(below) / len(active_df) * 100
print(f"Country-intervals where 5yr gain < {DISEASE_EFFECT_PP}pp: "
      f"{len(below)}/{len(active_df)} ({pct_below:.1f}%)")
print()
if len(below) > 0:
    print("  These are:")
    for _, row in below.sort_values("gain_5yr").head(20).iterrows():
        print(f"    {row['country']:30s} {row['interval']}  "
              f"baseline {row['baseline']:5.1f}%  gain {row['gain_5yr']:4.1f} pp")
