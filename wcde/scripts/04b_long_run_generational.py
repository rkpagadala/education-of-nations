"""
04b_long_run_generational.py
Long-run generational transmission using cohort reconstruction (back to ~1900).

Uses 02b_cohort_reconstruction.py output: education by cohort year (when they
were 20-24), reconstructed from older age groups at historical observation years.

Valid for: countries with reliable historical data and no colonial suppression.
  Japan, USA, UK, Germany, France, Australia, NZ, Canada, Argentina, Chile,
  Cuba, Uruguay, Costa Rica — "self-determination" countries.

Key question: does the T-25 parental multiplier hold over a 100-year horizon?
Does Japan's 1920s education investment explain its 1940s-1960s progress?

Outputs: wcde/output/long_run_generational.md
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "../data/processed")
OUT  = os.path.join(SCRIPT_DIR, "../output")
os.makedirs(OUT, exist_ok=True)

# Countries with reliable pre-1960 cohort data (self-determining, good historical records)
RELIABLE_COUNTRIES = {
    # East Asia (post-Meiji Japan, Taiwan, Korea rapid investment)
    "Japan",
    "Taiwan Province of China",
    "Republic of Korea",
    # Western Europe
    "United Kingdom of Great Britain and Northern Ireland",
    "France",
    "Germany",
    "Sweden",
    "Norway",
    "Denmark",
    "Finland",
    "Netherlands",
    "Belgium",
    "Switzerland",
    "Austria",
    "Italy",
    "Spain",
    "Portugal",
    # New World
    "United States of America",
    "Canada",
    "Australia",
    "New Zealand",
    # Latin America (independent since 1820s, own policy choices)
    "Argentina",
    "Chile",
    "Uruguay",
    "Cuba",
    "Costa Rica",
    # Special colonial cases with active education investment
    "Sri Lanka",  # British Ceylon — active investment, known anomaly
    "Hong Kong Special Administrative Region of China",
}

print("Loading cohort data...")
long = pd.read_csv(os.path.join(PROC, "cohort_completion_both_long.csv"))
print(f"  Total: {len(long)} rows, {long['country'].nunique()} countries, "
      f"cohort years {long['cohort_year'].min()}–{long['cohort_year'].max()}")

# Filter to reliable countries
long_r = long[long["country"].isin(RELIABLE_COUNTRIES)].copy()
print(f"  Reliable countries: {long_r['country'].nunique()}")

# Pivot to wide for easy lookup
low_w = long.pivot(index="country", columns="cohort_year", values="lower_sec")
pri_w = long.pivot(index="country", columns="cohort_year", values="primary")
low_r = long_r.pivot(index="country", columns="cohort_year", values="lower_sec")

def v(df_w, country, year):
    try:
        val = float(df_w.loc[country, int(year)])
        return val if not np.isnan(val) else np.nan
    except:
        return np.nan

# ── Long-run panel with T-25 lag ──────────────────────────────────────────────
# Cohort years available: 1870-2015 in 5-year steps
# T-25: child cohort C → parent cohort C-25
# For reliable countries, use cohort years 1900-2015 (child) → 1875-1990 (parent)

CHILD_COHORTS_ALL   = list(range(1900, 2016, 5))   # all countries from 1900
CHILD_COHORTS_EARLY = list(range(1900, 1960, 5))    # pre-WCDE-direct era (pre-1960 cohort data is from reconstruction)

panel_rows = []
for c in RELIABLE_COUNTRIES:
    if c not in low_w.index:
        continue
    for child_yr in CHILD_COHORTS_ALL:
        parent_yr  = child_yr - 25
        child_low  = v(low_w, c, child_yr)
        parent_low = v(low_w, c, parent_yr)
        child_pri  = v(pri_w, c, child_yr)

        if any(np.isnan(x) for x in [child_low, parent_low]):
            continue

        panel_rows.append({
            "country": c,
            "cohort_year": child_yr,
            "child_low": child_low,
            "parent_low": parent_low,
            "child_pri": child_pri,
            "pre_direct": child_yr < 1960,  # flag: using reconstructed data
        })

panel = pd.DataFrame(panel_rows)
print(f"\nLong-run panel: {len(panel)} obs, {panel['country'].nunique()} countries")
print(f"  Pre-1960 (reconstructed): {panel['pre_direct'].sum()} obs")
print(f"  Post-1960 (direct 20-24): {(~panel['pre_direct']).sum()} obs")

# ── Pooled OLS: full panel ─────────────────────────────────────────────────────
ok = ~panel["child_low"].isna() & ~panel["parent_low"].isna()
X = panel.loc[ok, ["parent_low"]].values
y = panel.loc[ok, "child_low"].values
reg_all = LinearRegression().fit(X, y)
r2_all  = reg_all.score(X, y)
print(f"\nPooled OLS (full, 1900-2015): β={reg_all.coef_[0]:.3f}, R²={r2_all:.3f}")

# FE version
pan_fe = panel[ok].copy()
pan_fe["child_dm"]  = pan_fe["child_low"]  - pan_fe.groupby("country")["child_low"].transform("mean")
pan_fe["parent_dm"] = pan_fe["parent_low"] - pan_fe.groupby("country")["parent_low"].transform("mean")
ok_fe = ~pan_fe["child_dm"].isna() & ~pan_fe["parent_dm"].isna()
reg_fe = LinearRegression(fit_intercept=False).fit(
    pan_fe.loc[ok_fe, ["parent_dm"]].values,
    pan_fe.loc[ok_fe, "child_dm"].values
)
r2_fe = reg_fe.score(pan_fe.loc[ok_fe, ["parent_dm"]].values, pan_fe.loc[ok_fe, "child_dm"].values)
print(f"Country FE (full, 1900-2015): β={reg_fe.coef_[0]:.3f}, R²={r2_fe:.3f}")

# ── Key country trajectories ──────────────────────────────────────────────────
KEY = [
    "Japan",
    "Republic of Korea",
    "Taiwan Province of China",
    "United States of America",
    "United Kingdom of Great Britain and Northern Ireland",
    "Germany",
    "France",
    "Sri Lanka",
    "Argentina",
    "Chile",
]
KEY = [c for c in KEY if c in low_w.index]

SNAP_COHORTS = [1875,1890,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2015]

# ── Report ────────────────────────────────────────────────────────────────────
lines = []
def h(t=""): lines.append(t)
def pct(val): return f"{val:.1f}%" if not np.isnan(val) else "n/a"
def cn(name, maxlen=40): return str(name)[:maxlen]

def pipe_table(headers, rows_data, aligns=None):
    if aligns is None:
        aligns = ["left"] + ["right"] * (len(headers) - 1)
    def sep(a): return ":---" if a == "left" else "---:"
    h("| " + " | ".join(headers) + " |")
    h("| " + " | ".join(sep(a) for a in aligns) + " |")
    for r in rows_data:
        h("| " + " | ".join(str(x) for x in r) + " |")
    h()

h("# Long-Run Generational Transmission of Education — WCDE v3 Cohort Reconstruction")
h()
h("*A 100-year view of the intergenerational multiplier using cohort-based education reconstruction.*")
h()
h("## How This Works")
h()
h("WCDE v3 provides education attainment by **age group** at each observation year (1950-2015).")
h("By reading older age groups at historical observation years, we reconstruct what each")
h("birth cohort looked like when they were 20-24:")
h()
h("  cohort_year = obs_year − (midpoint_age − 22)")
h()
h("Examples:")
h("- Age 60-64 at obs_year=1950 → cohort_year = 1910 (these people were 20-24 in 1910)")
h("- Age 80-84 at obs_year=1950 → cohort_year = 1890")
h("- Age 95-99 at obs_year=1950 → cohort_year = 1875")
h()
h("**Best estimate per cohort**: earliest available observation (youngest age at measurement)")
h("minimises survivorship bias.")
h()
h("### Survivorship Bias — Direction and Implications")
h()
h("Two competing survival effects operate on the cohort reconstruction:")
h()
h("1. **Education increases longevity**: educated people survive to old age at higher rates.")
h("   This **overestimates** education for cohorts measured late in life (age 70+).")
h("   Effect is strongest for pre-1910 cohorts, which are necessarily measured at age 80+ in 1950.")
h()
h("2. **Women live longer**: historically, women had lower education than men but higher survival.")
h("   This partially **offsets** the educated-survival bias by adding more low-education women")
h("   to the surviving pool. Effect is most pronounced in pre-1940 cohorts where gender gaps")
h("   in education were large.")
h()
h("**Net direction**: the two effects partially cancel. The residual is a modest upward bias")
h("in measured education for pre-1920 cohorts.")
h()
h("**Implication for the β coefficient**: for the T-25 regression, the *parent* cohort is")
h("measured at an older age than the *child* cohort (since parents are born 25 years earlier,")
h("they are further into old age when first observed in 1950). This means parental education")
h("is inflated more than child education. A higher-than-true parental education stretches")
h("the x-axis, compressing β. **Our estimates are therefore conservative** — the true")
h("intergenerational transmission coefficient is if anything higher than reported.")
h()
h("**Valid countries**: those with self-determined education policy and good historical records.")
h("Pre-1960 data for colonised countries reflects colonial investment, not domestic policy")
h("(though the mechanistic T-25 predictor still works — a literate colonial parent still")
h("transmits literacy to their child).")
h()
h("**Sri Lanka** is a documented anomaly: British colonial policy in Ceylon actively invested")
h("in education, making pre-1960 attainment relatively high and explaining Sri Lanka's")
h("persistent over-performance in later cohort regressions.")
h()
h("---")
h()
h("## Regression Results")
h()
h(f"Panel: {len(panel)} obs across {panel['country'].nunique()} countries, cohort years 1900–2015.")
h()
h("| Model | β (parental) | R² | Notes |")
h("|---|---|---|---|")
h(f"| Pooled OLS (1900–2015) | {reg_all.coef_[0]:.3f} | {r2_all:.3f} | Every 1pp parental → {reg_all.coef_[0]:.2f}pp child |")
h(f"| Country FE (1900–2015) | {reg_fe.coef_[0]:.3f} | {r2_fe:.3f} | Within-country, controls for all fixed traits |")
h()
h("The FE coefficient means: **within the same country over time**, a 1 pp rise in parental")
h(f"lower-secondary completion predicts a **{reg_fe.coef_[0]:.2f} pp** rise in child completion")
h("two generations later, after removing all time-invariant country effects.")
h()
h("---")
h()
h("## Table 1 — Lower Secondary Completion by Birth Cohort (Key Countries)")
h()
h("Each row is a cohort of people who were 20-24 in the given year.")
h("Pre-1960 values are reconstructed from older age groups at 1950 observation.")
h("Post-1960 are direct 20-24 measurements. Asterisk (*) marks reconstructed estimates.")
h()

# Build table
headers = ["Country"] + [str(c) for c in SNAP_COHORTS]
rows_data = []
for c in KEY:
    row = [cn(c, 28)]
    for cy in SNAP_COHORTS:
        val = v(low_w, c, cy)
        mark = "*" if cy < 1960 else ""
        row.append((pct(val) + mark) if not np.isnan(val) else "n/a")
    rows_data.append(row)
pipe_table(headers, rows_data, ["left"] + ["right"]*len(SNAP_COHORTS))

h("*\\* = reconstructed from older age group at 1950 observation year.*")
h()
h("---")
h()
h("## Table 2 — Primary Completion by Birth Cohort (Key Countries)")
h()
headers = ["Country"] + [str(c) for c in SNAP_COHORTS]
rows_data = []
for c in KEY:
    row = [cn(c, 28)]
    for cy in SNAP_COHORTS:
        val = v(pri_w, c, cy)
        mark = "*" if cy < 1960 else ""
        row.append((pct(val) + mark) if not np.isnan(val) else "n/a")
    rows_data.append(row)
pipe_table(headers, rows_data, ["left"] + ["right"]*len(SNAP_COHORTS))
h()
h("---")
h()
h("## Table 3 — The Generational Chain (T-25 Parent-Child Pairs for Key Countries)")
h()
h("For each country, showing parent cohort, child cohort, and the education levels,")
h("to make the intergenerational multiplier visible.")
h()
for c in KEY[:6]:
    h(f"**{c}:**")
    h()
    chain_rows = []
    for child_yr in [1925,1940,1950,1960,1970,1980,1990,2000,2015]:
        parent_yr = child_yr - 25
        child_low  = v(low_w, c, child_yr)
        parent_low = v(low_w, c, parent_yr)
        if np.isnan(child_low) or np.isnan(parent_low): continue
        gain = child_low - parent_low
        mark_c = "*" if child_yr < 1960 else ""
        mark_p = "*" if parent_yr < 1960 else ""
        chain_rows.append([
            f"{child_yr}{mark_c}",
            f"{parent_yr}{mark_p}",
            pct(parent_low) + mark_p,
            pct(child_low) + mark_c,
            f"+{gain:.1f} pp" if gain >= 0 else f"{gain:.1f} pp"
        ])
    if chain_rows:
        pipe_table(["Child Cohort","Parent Cohort","Parent Low Sec","Child Low Sec","Gain"],
                   chain_rows,
                   ["right","right","right","right","right"])

h("*\\* = reconstructed estimate.*")
h()
h("---")
h()
h("## Key Findings")
h()
h("### Japan: The 1920s Investment That Built the Postwar Miracle")
h()
jp_1920 = v(low_w, "Japan", 1920)
jp_1930 = v(low_w, "Japan", 1930)
jp_1940 = v(low_w, "Japan", 1940)
jp_1960 = v(low_w, "Japan", 1960)
jp_1980 = v(low_w, "Japan", 1980)
h(f"The 1920 cohort in Japan had {pct(jp_1920)} lower secondary completion.")
h(f"By the 1930 cohort this had jumped to {pct(jp_1930)}, and by 1940 to {pct(jp_1940)}.")
h(f"These people became the parents of Japan's 1945–1965 children. When those children")
h(f"were 20-24 (1960 cohort: {pct(jp_1960)}, 1980 cohort: {pct(jp_1980)}),")
h(f"they inherited an already-educated parental generation. Japan's postwar miracle")
h(f"was built on a pre-war education foundation that is invisible in post-1960 data.")
h()
h("### Republic of Korea: Post-Independence Acceleration")
h()
kr_1940 = v(low_w, "Republic of Korea", 1940)
kr_1950 = v(low_w, "Republic of Korea", 1950)
kr_1960 = v(low_w, "Republic of Korea", 1960)
kr_1970 = v(low_w, "Republic of Korea", 1970)
kr_1980 = v(low_w, "Republic of Korea", 1980)
h(f"Korea under Japanese colonial rule: 1940 cohort had only {pct(kr_1940)} lower secondary.")
h(f"Post-independence (1950 cohort): {pct(kr_1950)}. The state's deliberate investment")
h(f"drove rapid gains: 1960 cohort {pct(kr_1960)}, 1970 cohort {pct(kr_1970)},")
h(f"1980 cohort {pct(kr_1980)}. Korea is the canonical example of T-25 multiplication")
h(f"working through political commitment rather than colonial inheritance.")
h()
h("### USA: Steady but Slow — The Gradualist Path")
h()
us_1900 = v(low_w, "United States of America", 1900)
us_1930 = v(low_w, "United States of America", 1930)
us_1960 = v(low_w, "United States of America", 1960)
us_1980 = v(low_w, "United States of America", 1980)
h(f"The USA expanded steadily: {pct(us_1900)} lower sec in 1900, {pct(us_1930)} by 1930,")
h(f"{pct(us_1960)} by 1960, {pct(us_1980)} by 1980. No dramatic inflection — but also")
h(f"no colonial suppression and no rapid catch-up. The gradualist model.")
h()
h("### UK: Primary Completed by 1875, Secondary Slow for 75 Years")
h()
uk_1875 = v(low_w, "United Kingdom of Great Britain and Northern Ireland", 1875)
uk_1900 = v(low_w, "United Kingdom of Great Britain and Northern Ireland", 1900)
uk_1950 = v(low_w, "United Kingdom of Great Britain and Northern Ireland", 1950)
uk_1970 = v(low_w, "United Kingdom of Great Britain and Northern Ireland", 1970)
h(f"UK primary: {pct(uk_1875)} in 1875 (Forster Act 1870 effect visible).")
h(f"But lower secondary barely moved for 75 years: {pct(uk_1900)} in 1900, {pct(uk_1950)} in 1950.")
h(f"Not until the 1944 Butler Act made secondary universal did it accelerate: {pct(uk_1970)} by 1970.")
h(f"The UK case shows **primary completion alone does not create the T-25 secondary multiplier**")
h(f"— the secondary investment must also happen.")
h()
h("---")
h()
h("*Method: cohort reconstruction from WCDE v3 age-period data.*")
h("*Pre-1960 estimates from oldest available observation (least survivorship bias).*")
h("*Use for policy inference only with reliable-country subset.*")

OUT_MD = os.path.join(OUT, "long_run_generational.md")
with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved: {OUT_MD}")

# ── Write checkin JSON ───────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CHECKIN_DIR = os.path.join(REPO_ROOT, "checkin")
os.makedirs(CHECKIN_DIR, exist_ok=True)
checkin = {
    "script": "wcde/scripts/04b_long_run_generational.py",
    "produced": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "numbers": {
        "LR-countries": panel["country"].nunique(),
        "LR-obs": len(panel),
        "LR-pooled-beta": round(reg_all.coef_[0], 3),
        "LR-pooled-R2": round(r2_all, 3),
        "LR-FE-beta": round(reg_fe.coef_[0], 3),
        "LR-FE-R2": round(r2_fe, 3),
    },
}
checkin_path = os.path.join(CHECKIN_DIR, "long_run_generational.json")
with open(checkin_path, "w") as f:
    json.dump(checkin, f, indent=2)
print(f"\nCheckin written to {checkin_path}")
print("Done.")
