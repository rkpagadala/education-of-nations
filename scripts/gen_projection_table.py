"""
Generate education_projection_90pct.md  (and _upper.md)

Design:
  - Historical crossing years (Yr10, Yr90) from annual data/ data (1875–2015).
  - Projected Yr90 from WCDE SSP2 projections (2020–2100, 5-yr steps).
    No linear extrapolation of our own — WCDE numbers only.
    If 90% not reached by 2100 in WCDE data, report ">2100".
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[1]
RAW   = ROOT / "wcde" / "data" / "raw" / "prop_both.csv"

WCDE_PROC = ROOT / "wcde" / "data" / "processed"

level = (sys.argv[1].lower() if len(sys.argv) > 1 else "lower")
if level in ("upper", "higher"):
    # Use cohort completion data, interpolated to annual
    COHORT_FILE = WCDE_PROC / "cohort_completion_both_long.csv"
    COHORT_COL  = "upper_sec"
    OUT         = ROOT / "analysis" / "education_projection_90pct_upper.md"
    LABEL       = "Upper Secondary"
    WCDE_CATS   = ["Upper Secondary", "Post Secondary"]
else:
    COHORT_FILE = WCDE_PROC / "cohort_lower_sec_both.csv"
    COHORT_COL  = None  # wide format, no column selection needed
    OUT         = ROOT / "analysis" / "education_projection_90pct.md"
    LABEL       = "Lower Secondary"
    WCDE_CATS   = ["Lower Secondary", "Upper Secondary", "Post Secondary"]

# No name mapping needed — WCDE processed files already use WCDE country names
NAME_MAP = {}


# ── load historical data (WCDE v3 cohort, 5-yr intervals → annual) ───────────
if COHORT_COL:
    # Long format: pivot to wide
    _raw_cohort = pd.read_csv(COHORT_FILE)
    hist = _raw_cohort.pivot(index="country", columns="cohort_year", values=COHORT_COL)
else:
    # Already wide format
    hist = pd.read_csv(COHORT_FILE, index_col="country")

hist.columns = [int(c) for c in hist.columns]
# Interpolate 5-year → annual
all_years = list(range(min(hist.columns), max(hist.columns) + 1))
hist = hist.reindex(columns=all_years).interpolate(axis=1, method="linear")
hist = hist.bfill(axis=1).ffill(axis=1)
hist.index.name = "Country"
hist = hist.clip(lower=0)


# ── load WCDE projection data (1950–2100, 5-yr steps) ────────────────────────
raw = pd.read_csv(RAW)
raw = raw[(raw["age"] == "20--24") & (raw["sex"] == "Both")].copy()

# Always use only the basic-6 categories (they sum to ~100% at all years).
# Short Post Secondary / Bachelor / Master and higher are breakdowns of Post Secondary
# that appear in the 9-cat schema from 2020 onward — using them would double-count.
BASIC6 = {"No Education", "Incomplete Primary", "Primary",
          "Lower Secondary", "Upper Secondary", "Post Secondary"}
raw = raw[raw["education"].isin(BASIC6)]

wcde_pct = (raw[raw["education"].isin(WCDE_CATS)]
            .groupby(["name", "year"])["prop"]
            .sum()
            .unstack("year"))
wcde_pct.index.name = "wcde_name"
WCDE_YEARS = sorted(wcde_pct.columns.tolist())   # 1950, 1955, …, 2100


# ── helpers ───────────────────────────────────────────────────────────────────
def first_year_gte_annual(series, threshold):
    """First year in annual series where value >= threshold. Returns (year, value) or None."""
    mask = series >= threshold
    if not mask.any():
        return None
    yr = int(mask.idxmax())
    return yr, float(series[yr])


def interpolate_crossing_5yr(wcde_row, threshold, start_year):
    """
    Given a WCDE 5-yr row, find the projected year when value crosses `threshold`,
    starting search from `start_year`. Linearly interpolates between 5-yr points.
    Returns integer year or None if not reached by 2100.
    """
    years = [y for y in WCDE_YEARS if y >= start_year]
    prev_yr, prev_val = None, None
    for yr in years:
        val = wcde_row.get(yr, np.nan)
        if pd.isna(val):
            continue
        if val >= threshold:
            if prev_yr is None:
                return yr   # already above at first point
            # linear interpolation
            frac = (threshold - prev_val) / (val - prev_val)
            return int(round(prev_yr + frac * (yr - prev_yr)))
        prev_yr, prev_val = yr, val
    return None   # not reached by 2100


# ── build rows ────────────────────────────────────────────────────────────────
rows = []

for country, s in hist.iterrows():
    wcde_name = NAME_MAP.get(country, country)
    wcde_row  = wcde_pct.loc[wcde_name] if wcde_name in wcde_pct.index else None

    ls2015 = round(float(s[2015]), 2)

    # ── Yr10: from historical annual data ────────────────────────────────────
    already_above_10 = float(s[1875]) >= 10.0
    yr10_result = first_year_gte_annual(s, 10.0)
    if already_above_10:
        yr10_str = "≤1875"
        ls10     = round(float(s[1875]), 2)
    elif yr10_result:
        yr10_str = str(yr10_result[0])
        ls10     = round(yr10_result[1], 2)
    else:
        yr10_str = ">2015"   # hasn't crossed 10% even by 2015
        ls10     = "—"

    # ── Yr90: historical or WCDE projection ──────────────────────────────────
    already_above_90 = float(s[1875]) >= 90.0
    yr90_hist = first_year_gte_annual(s, 90.0)

    if already_above_90:
        yr90_str = "≤1875"
        ls90     = round(float(s[1875]), 2)
        status   = "Completed"
    elif yr90_hist is not None:
        yr90_str = str(yr90_hist[0])
        ls90     = round(yr90_hist[1], 2)
        status   = "Completed"
    else:
        # Need WCDE projection
        status = "In progress" if yr10_result or already_above_10 else "Below 10%"
        if wcde_row is not None:
            proj_yr = interpolate_crossing_5yr(wcde_row, 90.0, start_year=2016)
            if proj_yr is not None:
                yr90_str = f"~{proj_yr}"
                ls90     = "90.0"
            else:
                yr90_str = ">2100"
                ls90     = f"{round(float(wcde_row.get(2100, np.nan)), 1) if wcde_row is not None else '—'}"
        else:
            yr90_str = "no WCDE data"
            ls90     = "—"

    # ── Years 10→90 and Avg rate ──────────────────────────────────────────────
    if already_above_10 or already_above_90:
        yrs      = "n/a"
        avg_rate = "n/a"
        yrs_sort = -1
    elif yr90_str in (">2100", "no WCDE data", ">2015"):
        yrs      = ">2100" if yr90_str == ">2100" else "—"
        avg_rate = "—"
        yrs_sort = 99999
    else:
        try:
            yr10_int = int(yr10_str)
            yr90_int = int(yr90_str.lstrip("~"))
            yrs      = yr90_int - yr10_int
            avg_rate = round(80 / yrs, 2) if yrs > 0 else "—"
            yrs_sort = yrs
        except ValueError:
            yrs      = "—"
            avg_rate = "—"
            yrs_sort = 99999

    # WCDE 2025 value (latest projection point in processed data)
    wcde_2025 = "—"
    if wcde_row is not None and 2025 in wcde_row.index:
        v = wcde_row[2025]
        if not pd.isna(v):
            wcde_2025 = round(float(v), 1)

    rows.append({
        "Country":            country,
        "LS 2015 (%)":        ls2015,
        "WCDE 2025 (%)":      wcde_2025,
        "Yr crossed 10%":     yr10_str,
        "LS at 10% yr (%)":   ls10,
        "Yr crossed/est 90%": yr90_str,
        "LS at 90% yr (%)":   ls90,
        "Years 10→90":        yrs,
        "Avg rate (pp/yr)":   avg_rate,
        "Status":             status,
        "_yrs_sort":          yrs_sort,
    })

result = pd.DataFrame(rows)


# ── section sort ──────────────────────────────────────────────────────────────
def section_sort_key(row):
    if row["Status"] == "Completed":
        s = row["_yrs_sort"]
        return (0, s if s >= 0 else 9999)
    elif row["Status"] == "In progress":
        yr90 = row["Yr crossed/est 90%"]
        try:    return (1, int(yr90.lstrip("~")))
        except: return (1, 9999)
    else:
        yr90 = row["Yr crossed/est 90%"]
        try:    return (2, int(yr90.lstrip("~")))
        except: return (2, 9999)

result["_sect"] = result.apply(section_sort_key, axis=1)
result = result.sort_values("_sect").drop(columns="_sect").reset_index(drop=True)

completed   = result[result["Status"] == "Completed"]
in_progress = result[result["Status"] == "In progress"]
below10     = result[result["Status"] == "Below 10%"]

# combined table: all countries sorted by Years 10→90
def yrs_combined_key(row):
    s = row["_yrs_sort"]
    if s == -1:   return 99998   # pre-1875 at end
    return s

result["_yrs_c"] = result.apply(yrs_combined_key, axis=1)
combined = result.sort_values("_yrs_c").drop(columns=["_yrs_sort","_yrs_c"]).reset_index(drop=True)
result   = result.drop(columns=["_yrs_sort","_yrs_c"])


# ── markdown ──────────────────────────────────────────────────────────────────
COLS = [
    "Country",
    "LS 2015 (%)",
    "WCDE 2025 (%)",
    "Yr crossed 10%",
    "LS at 10% yr (%)",
    "Yr crossed/est 90%",
    "LS at 90% yr (%)",
    "Years 10→90",
    "Avg rate (pp/yr)",
    "Status",
]

def table_block(df_sub):
    header = "| " + " | ".join(COLS) + " |"
    sep    = "| " + " | ".join(["---"] * len(COLS)) + " |"
    lines  = [header, sep]
    for _, r in df_sub.iterrows():
        vals = [str(r[c]) for c in COLS]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def spot_row(name):
    m = result[result["Country"].str.contains(name, case=False, regex=False)]
    if m.empty:
        return f"  {name}: not found"
    r = m.iloc[0]
    return (f"  {r['Country']}: 2015={r['LS 2015 (%)']}, 2025={r['WCDE 2025 (%)']}, "
            f"Yr10={r['Yr crossed 10%']}, Yr90={r['Yr crossed/est 90%']}, "
            f"Yrs={r['Years 10→90']}, Avg={r['Avg rate (pp/yr)']} pp/yr")


# count ">2100" countries
over2100 = (result["Yr crossed/est 90%"] == ">2100").sum()

md_lines = [
    f"# Education Projection: 10% → 90% {LABEL} Completion",
    "",
    f"*Source: historical annual data (WCDE v3 reconstruction, 1875–2015) + WCDE SSP2 projections (2020–2100, 5-yr steps).*  ",
    "*Projected Yr90 (prefix `~`) = WCDE SSP2 projection, interpolated between 5-yr points.*  ",
    "*No independent extrapolation — WCDE numbers only. `>2100` = not reached in WCDE horizon.*  ",
    "*`≤1875` = already above threshold at first data point. `n/a` = crossing pre-dates 1875 data.*",
    "",
    "## Summary",
    "",
    "| Status | N | Definition |",
    "|--------|---|------------|",
    f"| Completed | {len(completed)} | Crossed 90% by 2015 (historical data) |",
    f"| In progress | {len(in_progress)} | Crossed 10% but not yet 90% by 2015 |",
    f"| Below 10% | {len(below10)} | Still below 10% in 2015 |",
    f"| &nbsp;&nbsp;of which >2100 | {over2100} | WCDE projects do not reach 90% by 2100 |",
    "",
    "**Spot checks:**",
    "```",
    spot_row("South Korea"),
    spot_row("Singapore"),
    spot_row("United States"),
    spot_row("Argentina"),
    spot_row("Niger"),
    "```",
    "",
    "---",
    "",
    f"## Completed ({len(completed)} countries)",
    "*Sorted by Years 10→90 ascending (fastest first). `n/a` = pre-1875 data.*",
    "",
    table_block(completed),
    "",
    "---",
    "",
    f"## In Progress ({len(in_progress)} countries)",
    "*Sorted by projected Yr 90% ascending. `~YYYY` = WCDE SSP2 interpolated.*",
    "",
    table_block(in_progress),
    "",
    "---",
    "",
    f"## Below 10% in 2015 ({len(below10)} countries)",
    "*Sorted by projected Yr 90% ascending.*",
    "",
    table_block(below10),
    "",
    "---",
    "",
    "## All Countries: Ranked by Years 10→90",
    "*All countries sorted fastest to slowest. `>2100` and `n/a` at end.*",
    "",
    table_block(combined),
]

OUT.write_text("\n".join(md_lines) + "\n")
print(f"Written: {OUT}")
print(f"  Completed:   {len(completed)}")
print(f"  In progress: {len(in_progress)}")
print(f"  Below 10%:   {len(below10)}")
print(f"  >2100:       {over2100}")
print(f"  No WCDE:     {(result['Yr crossed/est 90%'] == 'no WCDE data').sum()}")

print("\nSpot checks:")
for name in ["South Korea", "Singapore", "United States", "Argentina", "Niger"]:
    m = result[result["Country"].str.contains(name, case=False, regex=False)]
    if not m.empty:
        r = m.iloc[0]
        print(f"  {r['Country']}: 2015={r['LS 2015 (%)']}, 2025={r['WCDE 2025 (%)']}, "
              f"Yr10={r['Yr crossed 10%']}, Yr90={r['Yr crossed/est 90%']}, Yrs={r['Years 10→90']}")
