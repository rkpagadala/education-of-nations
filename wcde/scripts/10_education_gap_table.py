"""
10_education_gap_table.py

For every country, using WCDE v3 population data (age 20-24, both sexes),
compute for a given year:
  - Total cohort size
  - Number (and %) not completing primary (= no education + incomplete primary)
  - Number (and %) not completing lower secondary (= all below lower secondary)
  - Number (and %) not completing upper secondary (= all below upper secondary)

Usage:
  python3 10_education_gap_table.py          # defaults to 2025
  python3 10_education_gap_table.py 2035     # specify year

Output: analysis/education_gap_table_{year}.md
"""

import pandas as pd
import os
import sys

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2025

# ── Load ──────────────────────────────────────────────────────────────────────
pop = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "../data/raw/pop_both.csv")
)

# Filter: scenario 2 (SSP2 medium), age 20-24, both sexes
# 20-24 reflects completed education; 15-19 are still in school
df = pop[
    (pop["scenario"] == 2) &
    (pop["age"] == "20--24") &
    (pop["year"] == YEAR)
].copy()

# Aggregate both sexes
df = df.groupby(["name", "education"], as_index=False)["pop"].sum()

edu_order = [
    "Under 15",
    "No Education",
    "Incomplete Primary",
    "Primary",
    "Lower Secondary",
    "Upper Secondary",
    "Post Secondary",
]

piv = df.pivot_table(index="name", columns="education", values="pop", aggfunc="sum").fillna(0)

for col in edu_order:
    if col not in piv.columns:
        piv[col] = 0.0

piv["total"]        = piv[edu_order].sum(axis=1)
piv["miss_primary"] = piv["No Education"] + piv["Incomplete Primary"] + piv["Under 15"]
piv["miss_lower_sec"] = piv["miss_primary"] + piv["Primary"]
piv["miss_upper_sec"] = piv["miss_lower_sec"] + piv["Lower Secondary"]

piv["pct_miss_primary"]   = piv["miss_primary"]   / piv["total"] * 100
piv["pct_miss_lower_sec"] = piv["miss_lower_sec"] / piv["total"] * 100
piv["pct_miss_upper_sec"] = piv["miss_upper_sec"] / piv["total"] * 100

region_keywords = [
    "Africa", "Asia", "Europe", "America", "Caribbean", "Oceania",
    "World", "Eastern", "Western", "Northern", "Southern", "Central",
    "Sub-Saharan", "Latin", "Middle East", "North Africa",
    "Developed", "Developing", "Less developed", "Least developed",
    "Low income", "Lower middle", "Upper middle", "High income",
    "OECD", "EU", "G20",
]

def is_region(name):
    for kw in region_keywords:
        if kw.lower() in name.lower():
            return True
    return False

result = piv.reset_index()
result = result[~result["name"].apply(is_region)].copy()
result = result.sort_values("miss_lower_sec", ascending=False)

def fmt_M(x):
    m = x / 1000.0
    if m >= 0.05:
        return f"{m:.1f}M"
    return "<0.1M"

def fmt_pct(x):
    return f"{x:.0f}%"

# ── Build Markdown ─────────────────────────────────────────────────────────────
lines = []

lines.append(f"# Education Gap by Country — {YEAR} (Age 20–24 Cohort)")
lines.append("")
lines.append(f"*WCDE v3, scenario SSP2, age group 20–24, both sexes, year {YEAR}. Age 20–24 reflects completed education — people old enough to have finished secondary if they were going to.*")
lines.append("")
lines.append("**Definitions:**")
lines.append("- **Not completing primary** = no education + incomplete primary")
lines.append("- **Not completing lower secondary** = above + completed primary only")
lines.append("- **Not completing upper secondary** = above + completed lower secondary only")
lines.append("")
lines.append("Sorted by number not completing lower secondary, largest first.")
lines.append("")

g_total        = result["total"].sum()
g_miss_primary = result["miss_primary"].sum()
g_miss_lower   = result["miss_lower_sec"].sum()
g_miss_upper   = result["miss_upper_sec"].sum()

lines.append("## Global Totals")
lines.append("")
lines.append(f"| | Millions | % of 20–24 cohort |")
lines.append(f"|---|---:|---:|")
lines.append(f"| **Total 20–24 cohort** | **{g_total/1000:.0f}M** | |")
lines.append(f"| Not completing primary | {g_miss_primary/1000:.0f}M | {g_miss_primary/g_total*100:.0f}% |")
lines.append(f"| Not completing lower secondary | {g_miss_lower/1000:.0f}M | {g_miss_lower/g_total*100:.0f}% |")
lines.append(f"| Not completing upper secondary | {g_miss_upper/1000:.0f}M | {g_miss_upper/g_total*100:.0f}% |")
lines.append("")

lines.append("## All Countries")
lines.append("")
lines.append("| Country | Total 20–24 | Not completing primary | | Not completing lower sec | | Not completing upper sec | |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

for _, row in result.iterrows():
    if row["total"] < 10:
        continue
    lines.append(
        f"| {row['name']} "
        f"| {fmt_M(row['total'])} "
        f"| {fmt_M(row['miss_primary'])} | {fmt_pct(row['pct_miss_primary'])} "
        f"| {fmt_M(row['miss_lower_sec'])} | {fmt_pct(row['pct_miss_lower_sec'])} "
        f"| {fmt_M(row['miss_upper_sec'])} | {fmt_pct(row['pct_miss_upper_sec'])} |"
    )

lines.append("")

# ── Write output ───────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), f"../../analysis/education_gap_table_{YEAR}.md")
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Written: {out_path}")
print(f"Countries included: {len(result[result['total'] >= 10])}")
print(f"\nGlobal totals (thousands):")
print(f"  Total 20-24 cohort:           {g_total:,.0f}k = {g_total/1000:.0f}M")
print(f"  Not completing primary:       {g_miss_primary:,.0f}k = {g_miss_primary/1000:.0f}M ({g_miss_primary/g_total*100:.0f}%)")
print(f"  Not completing lower sec:     {g_miss_lower:,.0f}k = {g_miss_lower/1000:.0f}M ({g_miss_lower/g_total*100:.0f}%)")
print(f"  Not completing upper sec:     {g_miss_upper:,.0f}k = {g_miss_upper/1000:.0f}M ({g_miss_upper/g_total*100:.0f}%)")
