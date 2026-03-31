"""
05_alternative_rankings.py
Alternative education rankings on WCDE v3 data.

Questions:
  A. Mean Years of Schooling (Barro-Lee style)
  B. Bottleneck — where does each country lose the most students?
  C. Transition efficiency (% continuing to next level)
  D. Velocity — 2005→2015 growth rate per level
  E. Speed to 50% lower secondary
  F. Female deficit (primary and lower sec)

Outputs: wcde/output/alternative_rankings.md
"""

import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "../data/processed")
OUT  = os.path.join(SCRIPT_DIR, "../output")
os.makedirs(OUT, exist_ok=True)

REGIONS = {
    "Africa","Asia","Europe","World","Oceania","Caribbean",
    "Central America","South America","Latin America and the Caribbean",
    "Central Asia","Eastern Africa","Eastern Asia","Eastern Europe",
    "Northern Africa","Northern America","Northern Europe",
    "Southern Africa","Southern Asia","Southern Europe",
    "Western Africa","Western Asia","Western Europe",
    "Middle Africa","South-Eastern Asia",
}

OBS_YEARS = [1960,1965,1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2020,2025]

print("Loading data...")
both   = pd.read_csv(os.path.join(PROC, "completion_both_long.csv"))
female = pd.read_csv(os.path.join(PROC, "completion_female_long.csv"))

both   = both[~both["country"].isin(REGIONS)].copy()
female = female[~female["country"].isin(REGIONS)].copy()

for col in ["primary","lower_sec","upper_sec","college"]:
    both[col]   = both[col].clip(upper=100)
    female[col] = female[col].clip(upper=100)

pri_w   = both.pivot(index="country", columns="year", values="primary")
low_w   = both.pivot(index="country", columns="year", values="lower_sec")
upp_w   = both.pivot(index="country", columns="year", values="upper_sec")
col_w   = both.pivot(index="country", columns="year", values="college")
f_pri_w = female.pivot(index="country", columns="year", values="primary")
f_low_w = female.pivot(index="country", columns="year", values="lower_sec")

countries = sorted(pri_w.index)
print(f"  Countries: {len(countries)}")

def v(df_w, country, year):
    try:
        val = float(df_w.loc[country, year])
        return val if not np.isnan(val) else np.nan
    except:
        return np.nan

def pct(val): return f"{val:.1f}%" if not np.isnan(val) else "n/a"
def signed(val):
    if np.isnan(val): return "n/a"
    return f"+{val:.1f}" if val >= 0 else f"{val:.1f}"
def cn(name, maxlen=34): return str(name)[:maxlen]

# ── Build metrics ─────────────────────────────────────────────────────────────
rows = []
for c in countries:
    p15 = v(pri_w,c,2015); l15 = v(low_w,c,2015)
    u15 = v(upp_w,c,2015); c15 = v(col_w,c,2015)
    p05 = v(pri_w,c,2005); l05 = v(low_w,c,2005)
    u05 = v(upp_w,c,2005); c05 = v(col_w,c,2005)
    fp15 = v(f_pri_w,c,2015); fl15 = v(f_low_w,c,2015)

    # A. Mean Years of Schooling (Barro-Lee approximation)
    if not any(np.isnan(x) for x in [p15, l15, u15, c15]):
        no_edu   = max(100 - p15, 0)
        pri_only = max(p15 - l15, 0)
        low_only = max(l15 - u15, 0)
        upp_only = max(u15 - c15, 0)
        col_comp = max(c15, 0)
        mys = (no_edu*0 + pri_only*6 + low_only*9 + upp_only*12 + col_comp*16) / 100.0
    else:
        mys = np.nan

    # B. Bottleneck
    gaps = {}
    if not (np.isnan(p15) or np.isnan(l15)): gaps["pri→low"] = p15 - l15
    if not (np.isnan(l15) or np.isnan(u15)): gaps["low→upp"] = l15 - u15
    if not (np.isnan(u15) or np.isnan(c15)): gaps["upp→col"] = u15 - c15
    bottleneck    = max(gaps, key=gaps.get) if gaps else "n/a"
    bottleneck_pp = max(gaps.values()) if gaps else np.nan

    # C. Transition efficiency
    eff_pl = (l15/p15*100) if not (np.isnan(p15) or np.isnan(l15)) and p15 > 0 else np.nan
    eff_lu = (u15/l15*100) if not (np.isnan(l15) or np.isnan(u15)) and l15 > 0 else np.nan
    eff_uc = (c15/u15*100) if not (np.isnan(u15) or np.isnan(c15)) and u15 > 0 else np.nan

    # D. Velocity 2005-2015
    vel_p = (p15 - p05) / 10.0 if not (np.isnan(p15) or np.isnan(p05)) else np.nan
    vel_l = (l15 - l05) / 10.0 if not (np.isnan(l15) or np.isnan(l05)) else np.nan
    vel_u = (u15 - u05) / 10.0 if not (np.isnan(u15) or np.isnan(u05)) else np.nan
    vel_c = (c15 - c05) / 10.0 if not (np.isnan(c15) or np.isnan(c05)) else np.nan
    vel_mean = np.nanmean([x for x in [vel_p, vel_l, vel_u, vel_c] if not np.isnan(x)]) if any(not np.isnan(x) for x in [vel_p, vel_l, vel_u, vel_c]) else np.nan

    # E. Speed to 50% lower secondary
    speed_low50 = np.nan
    for yr in OBS_YEARS:
        lv = v(low_w, c, yr)
        if not np.isnan(lv) and lv >= 50:
            speed_low50 = yr
            break

    # F. Female deficit
    fem_def_pri = (fp15 - p15) if not (np.isnan(fp15) or np.isnan(p15)) else np.nan
    fem_def_low = (fl15 - l15) if not (np.isnan(fl15) or np.isnan(l15)) else np.nan

    rows.append({
        "country": c,
        "pri_2015": p15, "low_2015": l15, "upp_2015": u15, "col_2015": c15,
        "mys": mys,
        "bottleneck": bottleneck, "bottleneck_pp": bottleneck_pp,
        "gap_pl": gaps.get("pri→low", np.nan),
        "gap_lu": gaps.get("low→upp", np.nan),
        "gap_uc": gaps.get("upp→col", np.nan),
        "eff_pl": eff_pl, "eff_lu": eff_lu, "eff_uc": eff_uc,
        "vel_p": vel_p, "vel_l": vel_l, "vel_u": vel_u, "vel_c": vel_c,
        "vel_mean": vel_mean,
        "speed_low50": speed_low50,
        "fem_def_pri": fem_def_pri, "fem_def_low": fem_def_low,
    })

df = pd.DataFrame(rows)
df["mys_rank"] = df["mys"].rank(ascending=False, na_option="bottom").astype(int)

# ── Report ────────────────────────────────────────────────────────────────────
lines = []
def h(t=""): lines.append(t)

def pipe_table(headers, rows_data, aligns=None):
    if aligns is None:
        aligns = ["left"] + ["right"] * (len(headers) - 1)
    def sep(a): return ":---" if a == "left" else "---:"
    h("| " + " | ".join(headers) + " |")
    h("| " + " | ".join(sep(a) for a in aligns) + " |")
    for r in rows_data:
        h("| " + " | ".join(str(x) for x in r) + " |")
    h()

n_countries = len(df)
h("# Alternative Education Rankings — WCDE v3")
h()
h(f"*{n_countries} countries, 1960–2025. WCDE v3. Companion to rankings.md.*")
h()
h("---")
h()

# A. Mean Years of Schooling
h("## A. Mean Years of Schooling (2015)")
h()
h("Population-distribution view. No education=0yr, primary only=6yr, lower sec only=9yr,")
h("upper sec only=12yr, college=16yr.")
h()
mys_sorted = df.dropna(subset=["mys"]).sort_values("mys", ascending=False)
h(f"**Global mean years of schooling 2015: {df['mys'].mean():.2f} years ({n_countries} countries)**")
h()
h("**Top 40 — highest mean years of schooling:**")
h()
pipe_table(
    ["Rank","Country","MYS","Primary","Lower Sec","Upper Sec","College"],
    [[i+1, cn(r.country), f"{r.mys:.2f}",
      pct(r.pri_2015), pct(r.low_2015), pct(r.upp_2015), pct(r.col_2015)]
     for i, (_, r) in enumerate(mys_sorted.head(40).iterrows())],
    ["right","left","right","right","right","right","right"]
)
h("**Bottom 30 — lowest mean years of schooling:**")
h()
pipe_table(
    ["Rank","Country","MYS","Primary","Lower Sec","Upper Sec","College"],
    [[len(mys_sorted)-29+i, cn(r.country), f"{r.mys:.2f}",
      pct(r.pri_2015), pct(r.low_2015), pct(r.upp_2015), pct(r.col_2015)]
     for i, (_, r) in enumerate(mys_sorted.tail(30).iterrows())],
    ["right","left","right","right","right","right","right"]
)
h("---")
h()

# B. Bottleneck
h("## B. Bottleneck Analysis — Where Does Each Country Lose Students?")
h()
h("The largest drop on the education ladder for each country in 2015.")
h()
bn_counts = df["bottleneck"].value_counts()
h("**Bottleneck distribution:**")
h()
h("| Bottleneck | Countries | Description |")
h("|---|---|---|")
h(f"| pri→low | {bn_counts.get('pri→low', 0)} | Biggest loss at primary→lower secondary |")
h(f"| low→upp | {bn_counts.get('low→upp', 0)} | Biggest loss at lower→upper secondary |")
h(f"| upp→col | {bn_counts.get('upp→col', 0)} | Biggest loss at upper secondary→college |")
h()
for bn in ["pri→low", "low→upp", "upp→col"]:
    sub = df[df["bottleneck"] == bn].sort_values("bottleneck_pp", ascending=False)
    label = {"pri→low":"primary to lower secondary","low→upp":"lower to upper secondary","upp→col":"upper secondary to college"}[bn]
    h(f"**{bn} — worst {min(25,len(sub))} countries (biggest gap at {label}):**")
    h()
    pipe_table(
        ["Country","Primary","Lower Sec","Upper Sec","College","Gap"],
        [[cn(r.country), pct(r.pri_2015), pct(r.low_2015), pct(r.upp_2015), pct(r.col_2015),
          f"{r.bottleneck_pp:.1f} pp"]
         for _, r in sub.head(25).iterrows()],
        ["left","right","right","right","right","right"]
    )
h("---")
h()

# C. Transition Efficiency
h("## C. Transition Efficiency (2015)")
h()
h("Of those who completed a level, what % continued to the next?")
h()
eff_df = df.dropna(subset=["eff_pl"]).sort_values("eff_pl", ascending=False)
h("**Primary→Lower Secondary — top and bottom 20:**")
h()
h("*Top 20:*")
h()
pipe_table(
    ["Rank","Country","Primary","Lower Sec","Continuation %"],
    [[i+1, cn(r.country), pct(r.pri_2015), pct(r.low_2015), f"{r.eff_pl:.1f}%"]
     for i, (_, r) in enumerate(eff_df.head(20).iterrows())],
    ["right","left","right","right","right"]
)
h("*Bottom 20:*")
h()
pipe_table(
    ["Rank","Country","Primary","Lower Sec","Continuation %"],
    [[len(eff_df)-19+i, cn(r.country), pct(r.pri_2015), pct(r.low_2015), f"{r.eff_pl:.1f}%"]
     for i, (_, r) in enumerate(eff_df.tail(20).iterrows())],
    ["right","left","right","right","right"]
)
h("**Lower Secondary→Upper Secondary — bottom 20:**")
h()
eff_lu_df = df.dropna(subset=["eff_lu"]).sort_values("eff_lu")
pipe_table(
    ["Rank","Country","Lower Sec","Upper Sec","Continuation %"],
    [[len(eff_lu_df)-19+i, cn(r.country), pct(r.low_2015), pct(r.upp_2015), f"{r.eff_lu:.1f}%"]
     for i, (_, r) in enumerate(eff_lu_df.head(20).iterrows())],
    ["right","left","right","right","right"]
)
h("---")
h()

# D. Velocity
h("## D. Velocity — 2005→2015 Progress Rate (pp per year)")
h()
h("Where is momentum now? Ranked by mean annual gain across all 4 levels.")
h()
vel_df = df.dropna(subset=["vel_mean"]).sort_values("vel_mean", ascending=False)
h("**Top 30 — fastest current climbers:**")
h()
pipe_table(
    ["Rank","Country","Pri/yr","Low/yr","Upp/yr","Col/yr","Mean/yr","Primary 2015","Low Sec 2015"],
    [[i+1, cn(r.country),
      f"{r.vel_p:.2f}" if not np.isnan(r.vel_p) else "n/a",
      f"{r.vel_l:.2f}" if not np.isnan(r.vel_l) else "n/a",
      f"{r.vel_u:.2f}" if not np.isnan(r.vel_u) else "n/a",
      f"{r.vel_c:.2f}" if not np.isnan(r.vel_c) else "n/a",
      f"{r.vel_mean:.2f}", pct(r.pri_2015), pct(r.low_2015)]
     for i, (_, r) in enumerate(vel_df.head(30).iterrows())],
    ["right","left","right","right","right","right","right","right","right"]
)
h("**Top 20 — fastest lower-secondary growth 2005–2015:**")
h()
low_vel = df.dropna(subset=["vel_l"]).sort_values("vel_l", ascending=False)
pipe_table(
    ["Rank","Country","Low Sec 2005","Low Sec 2015","Growth/yr","Primary 2015"],
    [[i+1, cn(r.country),
      pct(v(low_w, r.country, 2005)),
      pct(r.low_2015),
      f"{r.vel_l:.2f} pp/yr",
      pct(r.pri_2015)]
     for i, (_, r) in enumerate(low_vel.head(20).iterrows())],
    ["right","left","right","right","right","right"]
)
h("---")
h()

# E. Speed to 50% lower secondary
h("## E. Speed to 50% Lower Secondary Completion")
h()
h("When did each country cross 50% lower secondary completion (20–24 cohort)?")
h("Tests the leapfrog thesis: did countries that invested in primary early also cross secondary early?")
h()
crossed_low = df.dropna(subset=["speed_low50"]).sort_values("speed_low50")
never_low   = df[df["speed_low50"].isna()].sort_values("low_2015", ascending=False)
h(f"{len(crossed_low)} countries crossed 50% lower secondary. {len(never_low)} had not by 2025.")
h()
pipe_table(
    ["Rank","Country","Crossed 50% Low Sec","Low Sec 2015","Primary 2015"],
    [[i+1, cn(r.country), int(r.speed_low50), pct(r.low_2015), pct(r.pri_2015)]
     for i, (_, r) in enumerate(crossed_low.iterrows())],
    ["right","left","right","right","right"]
)
h("**Countries that never crossed 50% lower secondary by 2025:**")
h()
pipe_table(
    ["Country","Low Sec 2015","Low Sec 2025","Primary 2015"],
    [[cn(r.country), pct(r.low_2015), pct(v(low_w, r.country, 2025)), pct(r.pri_2015)]
     for _, r in never_low.iterrows()],
    ["left","right","right","right"]
)
h("---")
h()

# F. Female Deficit
h("## F. Female Education Deficit (2015)")
h()
h("Female completion minus overall completion. **Negative** = girls behind.")
h("Female data available for primary and lower secondary.")
h()
fem_df = df.dropna(subset=["fem_def_pri","fem_def_low"]).copy()
fem_df["fem_def_combined"] = (fem_df["fem_def_pri"] + fem_df["fem_def_low"]) / 2
fem_sorted = fem_df.sort_values("fem_def_combined")
h("**Worst 30 combined female deficits:**")
h()
pipe_table(
    ["Rank","Country","F−Overall Primary","F−Overall Low Sec","Combined Deficit","Low Sec 2015"],
    [[i+1, cn(r.country),
      signed(r.fem_def_pri), signed(r.fem_def_low),
      f"{r.fem_def_combined:.1f} pp",
      pct(r.low_2015)]
     for i, (_, r) in enumerate(fem_sorted.head(30).iterrows())],
    ["right","left","right","right","right","right"]
)
fem_lead = fem_df[fem_df["fem_def_combined"] > 0].sort_values("fem_def_combined", ascending=False)
h("**Best 20 — countries where girls lead at both levels:**")
h()
pipe_table(
    ["Rank","Country","F−Overall Primary","F−Overall Low Sec","Combined Surplus","Low Sec 2015"],
    [[i+1, cn(r.country),
      signed(r.fem_def_pri), signed(r.fem_def_low),
      f"+{r.fem_def_combined:.1f} pp",
      pct(r.low_2015)]
     for i, (_, r) in enumerate(fem_lead.head(20).iterrows())],
    ["right","left","right","right","right","right"]
)
h("---")
h()
h("*WCDE v3 data. Age group 20–24. Female data from female cohort completion rates.*")

with open(os.path.join(OUT, "alternative_rankings.md"), "w") as f:
    f.write("\n".join(lines))
print(f"  Saved: {os.path.join(OUT, 'alternative_rankings.md')}")
print("Done.")
