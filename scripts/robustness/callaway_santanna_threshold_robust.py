"""
robustness/callaway_santanna_threshold_robust.py
=================================================
Runs the Callaway-Sant'Anna (2021) estimator at thresholds 5%, 10%, 15%,
20% to check robustness of the headline event-study result to the choice
of discretisation cutoff.

Reuses the exact estimator logic from callaway_santanna.py. Point
estimates only at each threshold (no bootstrap, for speed). Outputs a
threshold-by-outcome grid plus a compact event-study table per threshold.

Output: console print + checkin/callaway_santanna_threshold_robust.json
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import PROC, REGIONS, write_checkin, load_wb, get_wb_val

THRESHOLDS = [5, 10, 15, 20]
PERIODS = list(range(1950, 1995, 5))
LAG = 25

print("Loading data...")
edu_wide = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"), index_col="country")
edu_wide = edu_wide[~edu_wide.index.isin(REGIONS)]
le_df = load_wb("life_expectancy_years.csv")

treat_cols = [str(y) for y in PERIODS]
out_cols = [str(y + LAG) for y in PERIODS]
complete = edu_wide.dropna(subset=sorted(set(treat_cols + out_cols)))
countries = sorted(complete.index)
T = len(PERIODS)
period_idx = {t: j for j, t in enumerate(PERIODS)}


def build_cohort_arr(threshold):
    """Return (child_edu_arr, le_arr, cohort_vec, timing_vals, counts)."""
    rows = []
    for c in countries:
        for t in PERIODS:
            edu_t = float(complete.loc[c, str(t)])
            child_edu = float(complete.loc[c, str(t + LAG)])
            le_val = get_wb_val(le_df, c, t + LAG)
            rows.append({
                "country": c, "t": t, "edu": edu_t,
                "d": int(edu_t >= threshold),
                "child_edu": child_edu, "le": le_val,
            })
    panel = pd.DataFrame(rows)

    # Absorbing treatment
    for c in countries:
        d = panel.loc[panel["country"] == c, "d"].values.copy()
        if np.any(np.diff(d) < 0):
            first_on = np.argmax(d == 1)
            if d[first_on] == 1:
                d[first_on:] = 1
                panel.loc[panel["country"] == c, "d"] = d

    # Cohort assignment
    cohort_map = {}
    for c in countries:
        dv = panel.loc[panel["country"] == c].sort_values("t")
        tr = dv[dv["d"] == 1]["t"]
        if len(tr) == 0:
            cohort_map[c] = None
        elif len(tr) == T:
            cohort_map[c] = "always"
        else:
            cohort_map[c] = int(tr.min())

    cohort_arr = np.array([
        cohort_map[c] if isinstance(cohort_map[c], int)
        else (0 if cohort_map[c] == "always" else 9999)
        for c in countries
    ])
    timing_vals = sorted(set(
        g for g in cohort_map.values() if g not in (None, "always")
    ))
    counts = {
        "always": sum(1 for v in cohort_map.values() if v == "always"),
        "never":  sum(1 for v in cohort_map.values() if v is None),
        "timing": sum(1 for v in cohort_map.values()
                       if isinstance(v, int)),
    }

    country_idx = {c: i for i, c in enumerate(countries)}
    child_edu_arr = np.full((len(countries), T), np.nan)
    le_arr = np.full((len(countries), T), np.nan)
    for _, row in panel.iterrows():
        i = country_idx[row["country"]]
        j = period_idx[row["t"]]
        child_edu_arr[i, j] = row["child_edu"]
        le_arr[i, j] = row["le"]

    return child_edu_arr, le_arr, cohort_arr, timing_vals, counts


def compute_att_gt(outcome_arr, cohort_vec, timing_vals):
    results = []
    for g in timing_vals:
        base_j = period_idx.get(g - 5)
        if base_j is None:
            continue
        treat_mask = cohort_vec == g
        if treat_mask.sum() < 2:
            continue
        for t in PERIODS:
            j = period_idx[t]
            event_time = (t - g) // 5
            ctrl_mask = cohort_vec > t
            if ctrl_mask.sum() < 2:
                continue

            y_tb = outcome_arr[treat_mask, base_j]
            y_tt = outcome_arr[treat_mask, j]
            y_cb = outcome_arr[ctrl_mask, base_j]
            y_ct = outcome_arr[ctrl_mask, j]

            t_ok = ~np.isnan(y_tb) & ~np.isnan(y_tt)
            c_ok = ~np.isnan(y_cb) & ~np.isnan(y_ct)
            if t_ok.sum() < 2 or c_ok.sum() < 2:
                continue

            att = (y_tt[t_ok].mean() - y_tb[t_ok].mean()) - (
                y_ct[c_ok].mean() - y_cb[c_ok].mean())
            results.append({"g": g, "t": t, "event_time": event_time,
                             "att": att})
    return results


def aggregate(att_list):
    if not att_list:
        return np.nan, {}
    df = pd.DataFrame(att_list)
    post = df[df["event_time"] >= 0]
    att_agg = post["att"].mean() if len(post) > 0 else np.nan
    dynamic = {int(e): grp["att"].mean() for e, grp in df.groupby("event_time")}
    return att_agg, dynamic


# ── Run across thresholds ────────────────────────────────────────────
N = len(countries)
print(f"Panel: {N} countries × {T} periods ({PERIODS[0]}–{PERIODS[-1]})\n")

print(f"{'Threshold':>10}  {'Always':>7}  {'Timing':>7}  {'Never':>6}  "
      f"{'Cohorts':>7}  {'Edu ATT':>9}  {'LE ATT':>8}  {'Edu e=0':>8}  "
      f"{'Edu e=5':>8}  {'Edu e=7':>8}")
print("-" * 105)

all_results = {}
for thr in THRESHOLDS:
    ced, le, coh, tv, counts = build_cohort_arr(thr)
    att_edu = compute_att_gt(ced, coh, tv)
    att_le  = compute_att_gt(le,  coh, tv)
    agg_edu, dyn_edu = aggregate(att_edu)
    agg_le,  dyn_le  = aggregate(att_le)

    e0 = dyn_edu.get(0, np.nan)
    e5 = dyn_edu.get(5, np.nan)  # 25 years post-treatment
    e7 = dyn_edu.get(7, np.nan)  # 35 years post-treatment

    print(f"{thr:>9}%  {counts['always']:>7}  {counts['timing']:>7}  "
          f"{counts['never']:>6}  {len(tv):>7}  "
          f"{agg_edu:>9.2f}  {agg_le:>8.2f}  "
          f"{e0:>8.2f}  {e5:>8.2f}  {e7:>8.2f}")

    all_results[thr] = {
        "cohorts_timing": counts["timing"],
        "cohorts_always": counts["always"],
        "cohorts_never":  counts["never"],
        "n_cohort_years": len(tv),
        "edu_att_aggregate": round(float(agg_edu), 3) if not np.isnan(agg_edu) else None,
        "le_att_aggregate":  round(float(agg_le),  3) if not np.isnan(agg_le)  else None,
        "edu_event_study": {
            str(e): round(float(v), 3) for e, v in sorted(dyn_edu.items())
        },
        "le_event_study": {
            str(e): round(float(v), 3) for e, v in sorted(dyn_le.items())
        },
    }

# ── Event-study grid (child edu) ─────────────────────────────────────
print(f"\n{'=' * 80}")
print("Child education event study (ATT, percentage points) across thresholds")
print(f"{'=' * 80}")
all_events = sorted({int(e) for thr in THRESHOLDS
                      for e in all_results[thr]["edu_event_study"]})
print(f"  {'e':>3} {'yrs':>5}  " + "  ".join(f"{thr:>6}%" for thr in THRESHOLDS))
print(f"  {'-' * (11 + 9 * len(THRESHOLDS))}")
for e in all_events:
    vals = []
    for thr in THRESHOLDS:
        d = all_results[thr]["edu_event_study"].get(str(e))
        vals.append(f"{d:>6.2f}" if d is not None else "   -  ")
    mark = "+" if e >= 0 else " "
    print(f"  {mark}{e:>2} {e*5:>+4}y  " + "  ".join(vals))

# ── Event-study grid (life expectancy) ───────────────────────────────
print(f"\n{'=' * 80}")
print("Life expectancy event study (ATT, years) across thresholds")
print(f"{'=' * 80}")
all_events_le = sorted({int(e) for thr in THRESHOLDS
                         for e in all_results[thr]["le_event_study"]})
print(f"  {'e':>3} {'yrs':>5}  " + "  ".join(f"{thr:>6}%" for thr in THRESHOLDS))
print(f"  {'-' * (11 + 9 * len(THRESHOLDS))}")
for e in all_events_le:
    vals = []
    for thr in THRESHOLDS:
        d = all_results[thr]["le_event_study"].get(str(e))
        vals.append(f"{d:>6.2f}" if d is not None else "   -  ")
    mark = "+" if e >= 0 else " "
    print(f"  {mark}{e:>2} {e*5:>+4}y  " + "  ".join(vals))

write_checkin("callaway_santanna_threshold_robust.json", {
    "method": (
        f"Callaway-Sant'Anna (2021) event study at thresholds "
        f"{THRESHOLDS}. Point estimates only (no bootstrap). "
        f"Panel: {N} countries, {PERIODS[0]}-{PERIODS[-1]}, 5yr."
    ),
    "thresholds": THRESHOLDS,
    "results_by_threshold": {str(k): v for k, v in all_results.items()},
}, script_path="scripts/robustness/callaway_santanna_threshold_robust.py")
print("\nDone.")
