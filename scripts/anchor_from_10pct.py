"""
anchor_from_10pct.py
====================
Uses 10% lower-secondary completion as a common starting anchor for all
countries, then tests what predicts the development crossing from that anchor.

Countries: Taiwan, South Korea, Cuba, Bangladesh, Sri Lanka, China, India (Kerala proxy)

Development crossing = combined TFR < 3.65 AND LE > 69.8
TFR crossing = TFR < 3.65 alone
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
COHORT = BASE / "wcde/data/processed/cohort_lower_sec_both.csv"
SHORT  = BASE / "wcde/data/processed/lower_sec_both.csv"

# ── Country name mapping ─────────────────────────────────────────────────
DISPLAY = {
    "Taiwan Province of China": "Taiwan",
    "Republic of Korea": "South Korea",
    "Cuba": "Cuba",
    "Bangladesh": "Bangladesh",
    "Sri Lanka": "Sri Lanka",
    "China": "China",
    "India": "India (Kerala)",
}

# ── Development crossing dates (given) ───────────────────────────────────
DEV_CROSSING = {
    "Taiwan": 1972,
    "South Korea": 1987,
    "Cuba": 1974,
    "Bangladesh": 2014,
    "Sri Lanka": 1993,
    "China": 1994,
    "India (Kerala)": 1982,
}

TFR_CROSSING = {
    "Taiwan": 1972,
    "South Korea": 1975,
    "Cuba": 1972,
    "Bangladesh": 1995,
    "Sri Lanka": 1981,
    "China": 1975,
    "India (Kerala)": 1973,
}


def load_and_merge():
    """Load cohort (long-run) and short (1950-2020) data, prefer cohort for
    earlier years, fill gaps from short file."""
    cohort = pd.read_csv(COHORT, index_col=0)
    short = pd.read_csv(SHORT, index_col=0)

    # Cohort columns are years as strings; convert to int
    cohort.columns = [int(c) for c in cohort.columns]
    short.columns = [int(c) for c in short.columns]

    # Use cohort as base; for overlapping countries+years, fill NaN from short
    combined = cohort.copy()
    for col in short.columns:
        if col not in combined.columns:
            combined[col] = np.nan
    # Update NaN cells with short data where available
    common_idx = combined.index.intersection(short.index)
    common_cols = combined.columns.intersection(short.columns)
    for col in common_cols:
        na_mask = combined.loc[common_idx, col].isna()
        fill_idx = na_mask[na_mask].index
        if len(fill_idx) > 0:
            combined.loc[fill_idx, col] = short.loc[fill_idx, col]

    combined = combined.sort_index(axis=1)
    return combined


def interpolate_series(years, values):
    """Given 5-year spaced data, interpolate to annual resolution."""
    years = np.array(years, dtype=float)
    values = np.array(values, dtype=float)

    # Remove NaN
    mask = ~np.isnan(values)
    years = years[mask]
    values = values[mask]

    if len(years) < 2:
        return np.array([]), np.array([])

    annual_years = np.arange(int(years[0]), int(years[-1]) + 1)
    annual_values = np.interp(annual_years, years, values)
    return annual_years, annual_values


def find_crossing_year(annual_years, annual_values, threshold):
    """Find the first year the series crosses a threshold (from below)."""
    for i, (y, v) in enumerate(zip(annual_years, annual_values)):
        if v >= threshold:
            return int(y)
    return None


def compute_integral(annual_years, annual_values, start_year, end_year):
    """Compute cumulative percentage-point-years from start to end."""
    mask = (annual_years >= start_year) & (annual_years <= end_year)
    return float(np.sum(annual_values[mask]))


def get_value_at_year(annual_years, annual_values, year):
    """Get interpolated value at a specific year."""
    if year < annual_years[0] or year > annual_years[-1]:
        return np.nan
    return float(np.interp(year, annual_years, annual_values))


def cv(values):
    """Coefficient of variation."""
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0 or np.mean(arr) == 0:
        return np.nan
    return float(np.std(arr, ddof=1) / np.abs(np.mean(arr)))


def main():
    data = load_and_merge()

    THRESHOLD = 10.0  # 10% lower secondary completion

    # Build annual interpolated series for each country
    country_series = {}
    for raw_name, display_name in DISPLAY.items():
        if raw_name not in data.index:
            print(f"WARNING: {raw_name} not found in data")
            continue
        row = data.loc[raw_name]
        years = [int(c) for c in row.index]
        values = [float(v) if pd.notna(v) else np.nan for v in row.values]
        ay, av = interpolate_series(years, values)
        country_series[display_name] = (ay, av)

    # ── Step 1: Find 10% crossing year ───────────────────────────────────
    anchor_year = {}
    print("=" * 80)
    print("STEP 1: Year each country first reached 10% lower-secondary completion")
    print("=" * 80)
    print(f"{'Country':<20} {'10% Year':>10} {'Completion at 10% year':>25}")
    print("-" * 55)

    for name in DISPLAY.values():
        ay, av = country_series[name]
        yr = find_crossing_year(ay, av, THRESHOLD)
        anchor_year[name] = yr
        val_at_yr = get_value_at_year(ay, av, yr) if yr else np.nan
        print(f"{name:<20} {yr if yr else 'N/A':>10} {val_at_yr:>25.1f}")

    # ── Step 2: Compute metrics from 10% anchor ─────────────────────────
    print("\n" + "=" * 80)
    print("STEP 2: Metrics from 10% anchor to development crossing (combined TFR<3.65 & LE>69.8)")
    print("=" * 80)

    metrics = {}
    header = (f"{'Country':<16} {'10%yr':>6} {'DevXyr':>7} {'Gap':>5} "
              f"{'Rate':>7} {'@Cross':>7} {'@-25yr':>7} {'@-50yr':>7} {'Integral':>10}")
    print(header)
    print("-" * len(header))

    for name in DISPLAY.values():
        ay, av = country_series[name]
        a_yr = anchor_year[name]
        d_yr = DEV_CROSSING[name]

        if a_yr is None:
            continue

        gap = d_yr - a_yr
        val_at_cross = get_value_at_year(ay, av, d_yr)
        val_at_10 = get_value_at_year(ay, av, a_yr)
        rate = (val_at_cross - val_at_10) / gap if gap > 0 else np.nan
        val_25_before = get_value_at_year(ay, av, d_yr - 25)
        val_50_before = get_value_at_year(ay, av, d_yr - 50)
        integral = compute_integral(ay, av, a_yr, d_yr)

        metrics[name] = {
            "anchor_yr": a_yr,
            "dev_yr": d_yr,
            "gap": gap,
            "rate": rate,
            "at_cross": val_at_cross,
            "at_25_before": val_25_before,
            "at_50_before": val_50_before,
            "integral": integral,
        }

        print(f"{name:<16} {a_yr:>6} {d_yr:>7} {gap:>5} "
              f"{rate:>7.2f} {val_at_cross:>7.1f} {val_25_before:>7.1f} {val_50_before:>7.1f} {integral:>10.0f}")

    # ── Step 2b: Metrics from 10% anchor to TFR crossing ────────────────
    print("\n" + "=" * 80)
    print("STEP 2b: Metrics from 10% anchor to TFR crossing (TFR < 3.65)")
    print("=" * 80)

    tfr_metrics = {}
    header = (f"{'Country':<16} {'10%yr':>6} {'TFRXyr':>7} {'Gap':>5} "
              f"{'Rate':>7} {'@Cross':>7} {'@-25yr':>7} {'@-50yr':>7} {'Integral':>10}")
    print(header)
    print("-" * len(header))

    for name in DISPLAY.values():
        ay, av = country_series[name]
        a_yr = anchor_year[name]
        t_yr = TFR_CROSSING[name]

        if a_yr is None:
            continue

        gap = t_yr - a_yr
        val_at_cross = get_value_at_year(ay, av, t_yr)
        val_at_10 = get_value_at_year(ay, av, a_yr)
        rate = (val_at_cross - val_at_10) / gap if gap > 0 else np.nan
        val_25_before = get_value_at_year(ay, av, t_yr - 25)
        val_50_before = get_value_at_year(ay, av, t_yr - 50)
        integral = compute_integral(ay, av, a_yr, t_yr)

        tfr_metrics[name] = {
            "anchor_yr": a_yr,
            "tfr_yr": t_yr,
            "gap": gap,
            "rate": rate,
            "at_cross": val_at_cross,
            "at_25_before": val_25_before,
            "at_50_before": val_50_before,
            "integral": integral,
        }

        print(f"{name:<16} {a_yr:>6} {t_yr:>7} {gap:>5} "
              f"{rate:>7.2f} {val_at_cross:>7.1f} {val_25_before:>7.1f} {val_50_before:>7.1f} {integral:>10.0f}")

    # ── Step 3: PTE convergence test (grandmother vs mother) ─────────────
    print("\n" + "=" * 80)
    print("STEP 3: PTE convergence — grandmother (50yr) vs mother (25yr) before TFR crossing")
    print("=" * 80)

    vals_50 = [tfr_metrics[n]["at_50_before"] for n in tfr_metrics if not np.isnan(tfr_metrics[n]["at_50_before"])]
    vals_25 = [tfr_metrics[n]["at_25_before"] for n in tfr_metrics if not np.isnan(tfr_metrics[n]["at_25_before"])]

    print(f"\n{'Country':<20} {'@TFR-50yr':>12} {'@TFR-25yr':>12}")
    print("-" * 44)
    for name in tfr_metrics:
        m = tfr_metrics[name]
        print(f"{name:<20} {m['at_50_before']:>12.1f} {m['at_25_before']:>12.1f}")

    cv_50 = cv(vals_50)
    cv_25 = cv(vals_25)
    print(f"\n{'Mean':<20} {np.nanmean(vals_50):>12.1f} {np.nanmean(vals_25):>12.1f}")
    print(f"{'StdDev':<20} {np.nanstd(vals_50, ddof=1):>12.1f} {np.nanstd(vals_25, ddof=1):>12.1f}")
    print(f"{'CV':<20} {cv_50:>12.3f} {cv_25:>12.3f}")
    flag_50 = " *** ANCHOR ***" if cv_50 < 0.20 else ""
    flag_25 = " *** ANCHOR ***" if cv_25 < 0.20 else ""
    print(f"\nGrandmother (50yr) CV: {cv_50:.3f}{flag_50}")
    print(f"Mother (25yr) CV:      {cv_25:.3f}{flag_25}")
    if cv_50 < cv_25:
        print(">>> Grandmother lag converges TIGHTER than mother lag")
    else:
        print(">>> Mother lag converges tighter than grandmother lag")

    # ── Step 4: CV summary for development crossing ──────────────────────
    print("\n" + "=" * 80)
    print("STEP 4: CV summary for development crossing (combined TFR<3.65 & LE>69.8)")
    print("=" * 80)

    cv_tests = {
        "Years from 10% to crossing": [metrics[n]["gap"] for n in metrics],
        "Rate (pp/yr) from 10% to crossing": [metrics[n]["rate"] for n in metrics],
        "Integral (pp-years) 10% to crossing": [metrics[n]["integral"] for n in metrics],
        "Completion at crossing": [metrics[n]["at_cross"] for n in metrics],
        "Completion 25yr before crossing": [metrics[n]["at_25_before"] for n in metrics],
        "Completion 50yr before crossing": [metrics[n]["at_50_before"] for n in metrics],
    }

    print(f"\n{'Metric':<45} {'Mean':>10} {'StdDev':>10} {'CV':>8} {'Flag':>15}")
    print("-" * 88)
    for label, vals in cv_tests.items():
        clean = [v for v in vals if not np.isnan(v)]
        m = np.mean(clean)
        s = np.std(clean, ddof=1)
        c = cv(clean)
        flag = "*** ANCHOR ***" if c < 0.20 else ""
        print(f"{label:<45} {m:>10.2f} {s:>10.2f} {c:>8.3f} {flag:>15}")

    # ── Step 4b: CV summary for TFR crossing ─────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 4b: CV summary for TFR crossing (TFR < 3.65)")
    print("=" * 80)

    cv_tests_tfr = {
        "Years from 10% to TFR crossing": [tfr_metrics[n]["gap"] for n in tfr_metrics],
        "Rate (pp/yr) from 10% to TFR crossing": [tfr_metrics[n]["rate"] for n in tfr_metrics],
        "Integral (pp-years) 10% to TFR crossing": [tfr_metrics[n]["integral"] for n in tfr_metrics],
        "Completion at TFR crossing": [tfr_metrics[n]["at_cross"] for n in tfr_metrics],
        "Completion 25yr before TFR crossing": [tfr_metrics[n]["at_25_before"] for n in tfr_metrics],
        "Completion 50yr before TFR crossing": [tfr_metrics[n]["at_50_before"] for n in tfr_metrics],
    }

    print(f"\n{'Metric':<45} {'Mean':>10} {'StdDev':>10} {'CV':>8} {'Flag':>15}")
    print("-" * 88)
    for label, vals in cv_tests_tfr.items():
        clean = [v for v in vals if not np.isnan(v)]
        m = np.mean(clean)
        s = np.std(clean, ddof=1)
        c = cv(clean)
        flag = "*** ANCHOR ***" if c < 0.20 else ""
        print(f"{label:<45} {m:>10.2f} {s:>10.2f} {c:>8.3f} {flag:>15}")

    # ── Step 5: Integral CV sweep ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 5: Integral CV sweep — development crossing")
    print("For each window W (years from 10% anchor), compute integral and CV across countries")
    print("=" * 80)

    # Find the max gap
    max_gap = max(metrics[n]["gap"] for n in metrics)

    print(f"\n{'Window':>8} {'CV':>8} {'Flag':>15}   Values per country")
    print("-" * 80)

    best_cv = 999
    best_w = 0

    for w in range(1, max_gap + 1):
        integrals = []
        country_vals = []
        for name in metrics:
            ay, av = country_series[name]
            a_yr = anchor_year[name]
            end_yr = a_yr + w
            # Only include if this country's crossing is >= end_yr (still en route)
            # Actually, compute for all countries that have data up to a_yr + w
            if end_yr <= ay[-1]:
                integ = compute_integral(ay, av, a_yr, end_yr)
                integrals.append(integ)
                country_vals.append(f"{name[:3]}={integ:.0f}")

        if len(integrals) >= 4:  # Need at least 4 countries
            c = cv(integrals)
            flag = "*** ANCHOR ***" if c < 0.20 else ""
            if c < best_cv:
                best_cv = c
                best_w = w
            if w <= 10 or w % 5 == 0 or c < 0.20:
                print(f"{w:>8} {c:>8.3f} {flag:>15}   {', '.join(country_vals)}")

    print(f"\nBest integral window: {best_w} years, CV = {best_cv:.3f}")
    if best_cv < 0.20:
        print("*** This is a genuine anchor (CV < 0.20) ***")

    # ── Step 5b: Integral CV sweep — TFR crossing ────────────────────────
    print("\n" + "=" * 80)
    print("STEP 5b: Integral CV sweep — TFR crossing")
    print("For windows backward from TFR crossing: integral of education over W years before crossing")
    print("=" * 80)

    max_gap_tfr = max(tfr_metrics[n]["gap"] for n in tfr_metrics)

    print(f"\n{'Window':>8} {'CV':>8} {'Flag':>15}   Values per country")
    print("-" * 80)

    best_cv_tfr = 999
    best_w_tfr = 0

    for w in range(1, max_gap_tfr + 1):
        integrals = []
        country_vals = []
        for name in tfr_metrics:
            ay, av = country_series[name]
            a_yr = anchor_year[name]
            end_yr = a_yr + w
            if end_yr <= ay[-1]:
                integ = compute_integral(ay, av, a_yr, end_yr)
                integrals.append(integ)
                country_vals.append(f"{name[:3]}={integ:.0f}")

        if len(integrals) >= 4:
            c = cv(integrals)
            flag = "*** ANCHOR ***" if c < 0.20 else ""
            if c < best_cv_tfr:
                best_cv_tfr = c
                best_w_tfr = w
            if w <= 10 or w % 5 == 0 or c < 0.20:
                print(f"{w:>8} {c:>8.3f} {flag:>15}   {', '.join(country_vals)}")

    print(f"\nBest integral window: {best_w_tfr} years, CV = {best_cv_tfr:.3f}")
    if best_cv_tfr < 0.20:
        print("*** This is a genuine anchor (CV < 0.20) ***")

    # ── Step 5c: Backward integral sweep from crossing ───────────────────
    print("\n" + "=" * 80)
    print("STEP 5c: Backward integral sweep — integrate W years BACKWARD from TFR crossing")
    print("=" * 80)

    print(f"\n{'Window':>8} {'CV':>8} {'Flag':>15}   Values per country")
    print("-" * 80)

    best_cv_back = 999
    best_w_back = 0

    for w in range(5, 81, 1):
        integrals = []
        country_vals = []
        for name in tfr_metrics:
            ay, av = country_series[name]
            t_yr = TFR_CROSSING[name]
            start_yr = t_yr - w
            if start_yr >= ay[0]:
                integ = compute_integral(ay, av, start_yr, t_yr)
                integrals.append(integ)
                country_vals.append(f"{name[:3]}={integ:.0f}")

        if len(integrals) >= 4:
            c = cv(integrals)
            flag = "*** ANCHOR ***" if c < 0.20 else ""
            if c < best_cv_back:
                best_cv_back = c
                best_w_back = w
            if w <= 15 or w % 5 == 0 or c < 0.20:
                print(f"{w:>8} {c:>8.3f} {flag:>15}   {', '.join(country_vals)}")

    print(f"\nBest backward integral window: {best_w_back} years, CV = {best_cv_back:.3f}")
    if best_cv_back < 0.20:
        print("*** This is a genuine anchor (CV < 0.20) ***")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: All metrics with CV < 0.20 (genuine anchors)")
    print("=" * 80)

    all_cvs = {}
    for label, vals in cv_tests.items():
        clean = [v for v in vals if not np.isnan(v)]
        all_cvs[f"[Dev] {label}"] = cv(clean)
    for label, vals in cv_tests_tfr.items():
        clean = [v for v in vals if not np.isnan(v)]
        all_cvs[f"[TFR] {label}"] = cv(clean)
    all_cvs["[TFR] Grandmother (50yr lag) completion"] = cv_50
    all_cvs["[TFR] Mother (25yr lag) completion"] = cv_25
    all_cvs[f"[Dev] Best integral window ({best_w}yr from 10%)"] = best_cv
    all_cvs[f"[TFR] Best integral window ({best_w_tfr}yr from 10%)"] = best_cv_tfr
    all_cvs[f"[TFR] Best backward integral ({best_w_back}yr before crossing)"] = best_cv_back

    found_anchor = False
    for label, c in sorted(all_cvs.items(), key=lambda x: x[1]):
        flag = "*** ANCHOR ***" if c < 0.20 else ""
        if flag:
            found_anchor = True
        print(f"  CV = {c:.3f}  {flag:>15}  {label}")

    if not found_anchor:
        print("\n  No metric achieved CV < 0.20. No genuine anchor found.")


if __name__ == "__main__":
    main()
