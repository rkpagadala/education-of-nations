"""
Education Integral Anchor: cumulative educated-adult-years preceding
each country's development crossing date.

Hypothesis: a fixed completion threshold doesn't predict development
crossing, but the integral (cumulative educated-adult-years over the
preceding 25 years) might converge to a consistent value across countries.

The integral is the area under the lower secondary completion curve,
measured in percentage-point-years (pp-years).  If completion were
constant at 50% for 25 years, the integral would be 1250 pp-years.

Data sources:
  - Education (1950-2020): wcde/data/processed/lower_sec_both.csv
  - Long-run (1870-2015): wcde/data/processed/cohort_lower_sec_both.csv

Development crossing dates from Table 4 of the paper.
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROC_DIR = os.path.join(REPO_ROOT, "wcde", "data", "processed")

# ── Constants ─────────────────────────────────────────────────────
INTEGRAL_WINDOWS = [20, 25, 30]        # years preceding crossing date

CROSSING_DATES = {
    "Taiwan Province of China": 1972,
    "Republic of Korea":        1987,
    "Cuba":                     1974,
    "Bangladesh":               2014,
    "Sri Lanka":                1993,
    "China":                    1994,
}

SHORT_NAMES = {
    "Taiwan Province of China": "Taiwan",
    "Republic of Korea":        "South Korea",
    "Cuba":                     "Cuba",
    "Bangladesh":               "Bangladesh",
    "Sri Lanka":                "Sri Lanka",
    "China":                    "China",
}

# ── Load data ─────────────────────────────────────────────────────
# Prefer the long-run cohort file (back to 1870) for maximum coverage.
# Fall back to the shorter file for any missing country.
cohort = pd.read_csv(os.path.join(PROC_DIR, "cohort_lower_sec_both.csv"),
                     index_col="country")
short = pd.read_csv(os.path.join(PROC_DIR, "lower_sec_both.csv"),
                    index_col="country")


def get_time_series(country):
    """Return (years, values) arrays from the best available source."""
    for df in [cohort, short]:
        if country in df.index:
            row = df.loc[country].dropna()
            years = np.array([int(c) for c in row.index], dtype=float)
            vals = row.values.astype(float)
            order = np.argsort(years)
            return years[order], vals[order]
    raise KeyError(f"Country not found: {country}")


def interpolate_annual(years, vals):
    """Linearly interpolate 5-year data to annual resolution."""
    first, last = int(years[0]), int(years[-1])
    annual_years = np.arange(first, last + 1, dtype=float)
    annual_vals = np.interp(annual_years, years, vals)
    return annual_years, annual_vals


def compute_integral(annual_years, annual_vals, crossing, window):
    """
    Sum of annual completion values over [crossing - window, crossing - 1].
    Returns the integral in pp-years, or NaN if data doesn't cover the range.
    """
    start = crossing - window
    end = crossing - 1
    if start < annual_years[0] or end > annual_years[-1]:
        return np.nan
    mask = (annual_years >= start) & (annual_years <= end)
    return float(np.sum(annual_vals[mask]))


def completion_at_year(annual_years, annual_vals, year):
    """Return interpolated completion at a given year, or NaN."""
    if year < annual_years[0] or year > annual_years[-1]:
        return np.nan
    return float(np.interp(year, annual_years, annual_vals))


# ── Compute ───────────────────────────────────────────────────────
results = []

for country, crossing in CROSSING_DATES.items():
    years, vals = get_time_series(country)
    ay, av = interpolate_annual(years, vals)

    row = {
        "Country": SHORT_NAMES[country],
        "Crossing": crossing,
        "Completion at T-25": completion_at_year(ay, av, crossing - 25),
        "Completion at T": completion_at_year(ay, av, crossing),
    }
    for w in INTEGRAL_WINDOWS:
        row[f"Integral {w}yr"] = compute_integral(ay, av, crossing, w)
    results.append(row)

df = pd.DataFrame(results)

# ── Print table ───────────────────────────────────────────────────
print("=" * 90)
print("EDUCATION INTEGRAL ANCHOR")
print("Cumulative lower secondary completion (pp-years) preceding development crossing")
print("=" * 90)

fmt = "{:<14s} {:>8s} {:>10s} {:>10s} {:>14s} {:>14s} {:>14s}"
print(fmt.format("Country", "Cross", "Comp T-25", "Comp T",
                  "Int 20yr", "Int 25yr", "Int 30yr"))
print("-" * 90)

for _, r in df.iterrows():
    print(fmt.format(
        r["Country"],
        str(r["Crossing"]),
        f"{r['Completion at T-25']:.1f}%",
        f"{r['Completion at T']:.1f}%",
        f"{r['Integral 20yr']:.0f}" if not np.isnan(r["Integral 20yr"]) else "n/a",
        f"{r['Integral 25yr']:.0f}" if not np.isnan(r["Integral 25yr"]) else "n/a",
        f"{r['Integral 30yr']:.0f}" if not np.isnan(r["Integral 30yr"]) else "n/a",
    ))

print("-" * 90)

# ── Convergence assessment ────────────────────────────────────────
print("\n" + "=" * 90)
print("CONVERGENCE ASSESSMENT")
print("=" * 90)

for w in INTEGRAL_WINDOWS:
    col = f"Integral {w}yr"
    valid = df[col].dropna()
    if len(valid) < 2:
        print(f"\n{w}-year window: insufficient data")
        continue
    mean = valid.mean()
    std = valid.std()
    cv = std / mean if mean > 0 else np.nan
    rng = valid.max() - valid.min()
    print(f"\n{w}-year window (n={len(valid)}):")
    print(f"  Mean     = {mean:.0f} pp-years")
    print(f"  Std      = {std:.0f} pp-years")
    print(f"  CV       = {cv:.2f}")
    print(f"  Range    = {rng:.0f}  (min={valid.min():.0f}, max={valid.max():.0f})")
    print(f"  Ratio    = {valid.max() / valid.min():.1f}x")

# ── Also show completion at crossing (threshold check) ────────────
print("\n" + "=" * 90)
print("THRESHOLD COMPARISON (completion % at crossing date)")
print("=" * 90)
comp_t = df["Completion at T"].dropna()
print(f"  Mean     = {comp_t.mean():.1f}%")
print(f"  Std      = {comp_t.std():.1f}%")
print(f"  CV       = {comp_t.std()/comp_t.mean():.2f}")
print(f"  Range    = {comp_t.min():.1f}% – {comp_t.max():.1f}%")
print(f"\nIf the integral CV < threshold CV, the integral is a better predictor.")
integral_cv = df["Integral 25yr"].dropna().std() / df["Integral 25yr"].dropna().mean()
threshold_cv = comp_t.std() / comp_t.mean()
print(f"  Integral 25yr CV = {integral_cv:.2f}")
print(f"  Threshold CV     = {threshold_cv:.2f}")
if integral_cv < threshold_cv:
    print("  --> Integral converges BETTER than threshold.")
else:
    print("  --> Threshold converges better than integral.")
