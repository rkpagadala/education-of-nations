"""
Separate education anchors for TFR and LE threshold crossings.

Hypothesis: TFR drops fast once education reaches some level (household
decision by educated women — responds quickly). LE rises slowly (requires
workforce, infrastructure, time). Separating them may reveal a tighter
anchor for one or both.

Uses WCDE v3 lower-secondary completion (age 20-24, both sexes) and
interpolates 5-year data to annual for lag search at 1-year resolution.
"""

import pandas as pd
import numpy as np

# ── Data paths ──────────────────────────────────────────────────────────
COHORT_BOTH = "../wcde/data/processed/cohort_lower_sec_both.csv"
PERIOD_BOTH = "../wcde/data/processed/lower_sec_both.csv"
COHORT_FEMALE = None  # cohort female may not exist
PERIOD_FEMALE = "../wcde/data/processed/lower_sec_female.csv"

# ── Country name mapping (display → CSV) ────────────────────────────────
COUNTRY_MAP = {
    "Taiwan":     "Taiwan Province of China",
    "S. Korea":   "Republic of Korea",
    "Cuba":       "Cuba",
    "Bangladesh": "Bangladesh",
    "Sri Lanka":  "Sri Lanka",
    "China":      "China",
}

# ── Crossing dates from Table 4 ────────────────────────────────────────
TFR_CROSS = {
    "Taiwan": 1972, "S. Korea": 1975, "Cuba": 1972,
    "Bangladesh": 1995, "Sri Lanka": 1981, "China": 1975,
}

LE_CROSS = {
    "Taiwan": 1972, "S. Korea": 1987, "Cuba": 1974,
    "Bangladesh": 2014, "Sri Lanka": 1993, "China": 1994,
}

# Combined threshold crossing (for comparison — from paper)
COMBINED_CROSS = {
    "Taiwan": 1972, "S. Korea": 1987, "Cuba": 1974,
    "Bangladesh": 2014, "Sri Lanka": 1993, "China": 1994,
}


def load_and_interpolate(path):
    """Load CSV with 5-year columns, return DataFrame with annual interpolation."""
    df = pd.read_csv(path, index_col="country")
    df.columns = df.columns.astype(int)
    # Transpose so rows=years, columns=countries, then interpolate
    dft = df.T
    dft.index = dft.index.astype(int)
    # Reindex to annual
    full_idx = range(dft.index.min(), dft.index.max() + 1)
    dft = dft.reindex(full_idx).interpolate(method="linear")
    return dft  # rows=years, columns=country_names


def get_education_at_lag(annual_df, crossing_dates, lag):
    """For each country, get education completion `lag` years before crossing."""
    values = {}
    for short_name, csv_name in COUNTRY_MAP.items():
        cross_year = crossing_dates[short_name]
        lookup_year = cross_year - lag
        if csv_name not in annual_df.columns:
            values[short_name] = np.nan
            continue
        if lookup_year < annual_df.index.min() or lookup_year > annual_df.index.max():
            values[short_name] = np.nan
        else:
            values[short_name] = annual_df.loc[lookup_year, csv_name]
    return values


def compute_cv(values):
    """Coefficient of variation (std/mean) from dict of values, ignoring NaN."""
    v = [x for x in values.values() if not np.isnan(x)]
    if len(v) < 3 or np.mean(v) == 0:
        return np.nan
    return np.std(v, ddof=0) / np.mean(v)


def scan_lags(annual_df, crossing_dates, max_lag=50):
    """Scan lags 0..max_lag, return list of (lag, cv, values_dict)."""
    results = []
    for lag in range(max_lag + 1):
        vals = get_education_at_lag(annual_df, crossing_dates, lag)
        cv = compute_cv(vals)
        results.append((lag, cv, vals))
    return results


def print_table(results, label, step=5):
    """Print CV table at every `step` years, plus minimum."""
    countries = list(COUNTRY_MAP.keys())
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"{'Lag':>4}  {'CV':>6}  ", end="")
    for c in countries:
        print(f"{c:>12}", end="")
    print()
    print("-" * (12 + 6 + len(countries) * 12))

    for lag, cv, vals in results:
        if lag % step == 0:
            print(f"{lag:>4}  {cv:>6.3f}  ", end="")
            for c in countries:
                v = vals.get(c, np.nan)
                print(f"{v:>12.1f}" if not np.isnan(v) else f"{'N/A':>12}", end="")
            print()

    # Find minimum CV
    valid = [(lag, cv, vals) for lag, cv, vals in results if not np.isnan(cv)]
    if valid:
        best_lag, best_cv, best_vals = min(valid, key=lambda x: x[1])
        print(f"\n  *** Minimum CV = {best_cv:.4f} at lag = {best_lag} years ***")
        print(f"  Completion values at optimal lag:")
        for c in countries:
            v = best_vals.get(c, np.nan)
            print(f"    {c:>12}: {v:.1f}%")
        mean_val = np.nanmean([best_vals[c] for c in countries])
        print(f"    {'Mean':>12}: {mean_val:.1f}%")
    return valid


def main():
    # ── Load both-sex data ──────────────────────────────────────────────
    # Use cohort data (goes back to 1870) as primary; fall back to period
    try:
        annual_both = load_and_interpolate(COHORT_BOTH)
        source_both = "cohort (1870-2015)"
    except FileNotFoundError:
        annual_both = load_and_interpolate(PERIOD_BOTH)
        source_both = "period (1950-2020)"

    print(f"Data source (both sexes): {source_both}")
    print(f"Year range: {annual_both.index.min()}-{annual_both.index.max()}")

    # ── Load female data ────────────────────────────────────────────────
    try:
        annual_female = load_and_interpolate(PERIOD_FEMALE)
        has_female = True
        print(f"Data source (female): period (1950-2020)")
    except FileNotFoundError:
        has_female = False
        print("Female data: not available")

    # Also check for cohort female
    try:
        cohort_female = load_and_interpolate(
            "../wcde/data/processed/cohort_lower_sec_female.csv"
        )
        has_cohort_female = True
        print(f"Data source (female cohort): cohort (1870-2015)")
    except FileNotFoundError:
        has_cohort_female = False

    female_df = cohort_female if has_cohort_female else (annual_female if has_female else None)

    # ── TFR crossing analysis (both sexes) ──────────────────────────────
    tfr_results = scan_lags(annual_both, TFR_CROSS, max_lag=50)
    tfr_valid = print_table(tfr_results, "TFR < 3.65 CROSSING — Education anchor (both sexes)")

    # ── LE crossing analysis (both sexes) ───────────────────────────────
    le_results = scan_lags(annual_both, LE_CROSS, max_lag=50)
    le_valid = print_table(le_results, "LE > 69.8 CROSSING — Education anchor (both sexes)")

    # ── Combined crossing (for comparison) ──────────────────────────────
    comb_results = scan_lags(annual_both, COMBINED_CROSS, max_lag=50)
    comb_valid = print_table(comb_results, "COMBINED CROSSING (latest of TFR/LE) — for comparison")

    # ── TFR crossing with FEMALE education ──────────────────────────────
    if female_df is not None:
        tfr_female_results = scan_lags(female_df, TFR_CROSS, max_lag=50)
        tfr_f_valid = print_table(
            tfr_female_results,
            "TFR < 3.65 CROSSING — Education anchor (FEMALE lower secondary)"
        )
    else:
        tfr_f_valid = None

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")

    def best_of(valid):
        if not valid:
            return None, None
        best = min(valid, key=lambda x: x[1])
        return best[0], best[1]

    tfr_lag, tfr_cv = best_of(tfr_valid)
    le_lag, le_cv = best_of(le_valid)
    comb_lag, comb_cv = best_of(comb_valid)

    print(f"\n  {'Threshold':<35} {'Best lag':>10} {'Min CV':>10}")
    print(f"  {'-'*55}")
    print(f"  {'TFR < 3.65 (both sexes)':<35} {tfr_lag:>8} yr {tfr_cv:>10.4f}")
    print(f"  {'LE > 69.8 (both sexes)':<35} {le_lag:>8} yr {le_cv:>10.4f}")
    print(f"  {'Combined (both sexes)':<35} {comb_lag:>8} yr {comb_cv:>10.4f}")

    if tfr_f_valid:
        tf_lag, tf_cv = best_of(tfr_f_valid)
        print(f"  {'TFR < 3.65 (female)':<35} {tf_lag:>8} yr {tf_cv:>10.4f}")

    # Completion levels at best lag
    print(f"\n  Completion levels at optimal lag:")
    countries = list(COUNTRY_MAP.keys())

    for label, lag, cross_dates, df in [
        ("TFR (both)", tfr_lag, TFR_CROSS, annual_both),
        ("LE (both)", le_lag, LE_CROSS, annual_both),
        ("Combined", comb_lag, COMBINED_CROSS, annual_both),
    ]:
        vals = get_education_at_lag(df, cross_dates, lag)
        mean_v = np.nanmean(list(vals.values()))
        print(f"\n  {label} (lag={lag}): mean = {mean_v:.1f}%")
        for c in countries:
            print(f"    {c:>12}: {vals[c]:.1f}%")

    if female_df is not None and tfr_f_valid:
        tf_lag, _ = best_of(tfr_f_valid)
        vals = get_education_at_lag(female_df, TFR_CROSS, tf_lag)
        mean_v = np.nanmean(list(vals.values()))
        print(f"\n  TFR (female) (lag={tf_lag}): mean = {mean_v:.1f}%")
        for c in countries:
            print(f"    {c:>12}: {vals[c]:.1f}%")

    # Which is tighter?
    print(f"\n  CONCLUSION:")
    if tfr_cv < le_cv:
        print(f"  TFR has the TIGHTER anchor (CV={tfr_cv:.4f} vs LE CV={le_cv:.4f})")
        print(f"  TFR crossing is anchored at ~{np.nanmean(list(get_education_at_lag(annual_both, TFR_CROSS, tfr_lag).values())):.0f}% lower-secondary completion, {tfr_lag} years before crossing.")
    else:
        print(f"  LE has the TIGHTER anchor (CV={le_cv:.4f} vs TFR CV={tfr_cv:.4f})")
        print(f"  LE crossing is anchored at ~{np.nanmean(list(get_education_at_lag(annual_both, LE_CROSS, le_lag).values())):.0f}% lower-secondary completion, {le_lag} years before crossing.")

    if tfr_f_valid:
        tf_lag, tf_cv = best_of(tfr_f_valid)
        if tf_cv < tfr_cv:
            print(f"  Female education tightens TFR anchor further: CV={tf_cv:.4f} (vs {tfr_cv:.4f} both sexes)")
        else:
            print(f"  Female education does NOT tighten TFR anchor: CV={tf_cv:.4f} (vs {tfr_cv:.4f} both sexes)")


if __name__ == "__main__":
    main()
