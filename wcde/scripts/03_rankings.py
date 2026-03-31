"""
03_rankings.py
Global Education Achievement Rankings — WCDE v3 data, 228 entities, 1960-2025.

Outputs:
  wcde/output/rankings.md
  wcde/output/rankings.csv
"""

import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "../data/processed")
OUT  = os.path.join(SCRIPT_DIR, "../output")
os.makedirs(OUT, exist_ok=True)

# WCDE regional aggregates to exclude
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
SNAP_YEARS = [1960,1975,1990,2005,2015,2025]

def cap100(x):
    if np.isnan(x): return np.nan
    return min(x, 100.0)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
both   = pd.read_csv(os.path.join(PROC, "completion_both_long.csv"))
female = pd.read_csv(os.path.join(PROC, "completion_female_long.csv"))

# Filter to countries only, cap at 100
both = both[~both["country"].isin(REGIONS)].copy()
female = female[~female["country"].isin(REGIONS)].copy()

for col in ["primary","lower_sec","upper_sec","college"]:
    both[col]   = both[col].clip(upper=100)
    female[col] = female[col].clip(upper=100)

countries = sorted(both["country"].unique())
print(f"  Countries: {len(countries)}")

def v(df_wide, country, year):
    """Lookup value from pivoted wide df (country as index, year as int columns)."""
    try:
        val = float(df_wide.loc[country, year])
        return val if not np.isnan(val) else np.nan
    except:
        return np.nan

# Pivot to wide format per level
def pivot_level(df, level):
    p = df.pivot(index="country", columns="year", values=level)
    return p

pri_w   = pivot_level(both,   "primary")
low_w   = pivot_level(both,   "lower_sec")
upp_w   = pivot_level(both,   "upper_sec")
col_w   = pivot_level(both,   "college")
f_pri_w = pivot_level(female, "primary")
f_low_w = pivot_level(female, "lower_sec")

def safe_mean(*vals):
    vals = [x for x in vals if not np.isnan(x)]
    return np.mean(vals) if vals else np.nan

def wscore(p, l, u, cl):
    """Ladder score: college 2.5×, upper-sec 2×, lower-sec 1.5×, primary 1×."""
    vals = [(p,1),(l,1.5),(u,2),(cl,2.5)]
    num = sum(val*w for val,w in vals if not np.isnan(val))
    den = sum(w for val,w in vals if not np.isnan(val))
    return num/den if den>0 else np.nan

# ── Build country metrics ─────────────────────────────────────────────────────
print("Computing metrics...")
rows = []
for c in countries:
    def snap(df_w, yr): return v(df_w, c, yr)

    p60, p75, p90, p05, p15, p25 = snap(pri_w,1960),snap(pri_w,1975),snap(pri_w,1990),snap(pri_w,2005),snap(pri_w,2015),snap(pri_w,2025)
    l60, l75, l90, l05, l15, l25 = snap(low_w,1960),snap(low_w,1975),snap(low_w,1990),snap(low_w,2005),snap(low_w,2015),snap(low_w,2025)
    u60, u75, u90, u05, u15, u25 = snap(upp_w,1960),snap(upp_w,1975),snap(upp_w,1990),snap(upp_w,2005),snap(upp_w,2015),snap(upp_w,2025)
    c60, c75, c90, c05, c15, c25 = snap(col_w,1960),snap(col_w,1975),snap(col_w,1990),snap(col_w,2005),snap(col_w,2015),snap(col_w,2025)

    edu60  = safe_mean(p60, l60, u60, c60)
    edu15  = safe_mean(p15, l15, u15, c15)
    gain   = edu15 - edu60 if not (np.isnan(edu60) or np.isnan(edu15)) else np.nan

    ladder15 = wscore(p15, l15, u15, c15)
    ladder60 = wscore(p60, l60, u60, c60)
    ladder25 = wscore(p25, l25, u25, c25)

    # Speed: first year crossing 60% primary
    speed_year = np.nan
    for yr in OBS_YEARS:
        pv = v(pri_w, c, yr)
        if not np.isnan(pv) and pv >= 60:
            speed_year = yr
            break

    # Gender gaps 2015
    fp15 = snap(f_pri_w, 2015)
    fl15 = snap(f_low_w, 2015)
    gender_gap     = fp15 - p15 if not (np.isnan(fp15) or np.isnan(p15)) else np.nan
    gender_gap_low = fl15 - l15 if not (np.isnan(fl15) or np.isnan(l15)) else np.nan

    # Sequential gap 2015
    seq_gap15 = p15 - l15 if not (np.isnan(p15) or np.isnan(l15)) else np.nan

    # Peak decade (10-year periods)
    decade_gains = {}
    for i in range(0, len(OBS_YEARS)-2, 2):
        y0, y1 = OBS_YEARS[i], OBS_YEARS[i+2]
        s0 = safe_mean(v(pri_w,c,y0),v(low_w,c,y0),v(upp_w,c,y0),v(col_w,c,y0))
        s1 = safe_mean(v(pri_w,c,y1),v(low_w,c,y1),v(upp_w,c,y1),v(col_w,c,y1))
        if not (np.isnan(s0) or np.isnan(s1)):
            decade_gains[f"{y0}–{y1}"] = s1 - s0
    peak_decade = max(decade_gains, key=decade_gains.get) if decade_gains else "n/a"
    peak_gain   = max(decade_gains.values()) if decade_gains else np.nan

    # Archetype 2015
    if not (np.isnan(p15) or np.isnan(l15)):
        if   p15 >= 92 and l15 >= 85 and (np.isnan(u15) or u15 >= 70):
            archetype = "Universal"
        elif p15 >= 85 and l15 >= 65:
            archetype = "Secondary Building"
        elif p15 >= 80 and l15 >= 40:
            archetype = "Secondary Transition"
        elif p15 >= 70 and l15 < 40:
            archetype = "Primary Complete"
        elif p15 >= 45:
            archetype = "Primary Building"
        else:
            archetype = "Low Access"
    else:
        archetype = "No Data"

    # Trajectory
    if not np.isnan(gain):
        if gain <= -10:
            trajectory = "Regression"
        elif edu60 >= 65:
            trajectory = "Early Achiever"
        elif gain >= 40:
            trajectory = "Large Gain"
        elif gain >= 22:
            trajectory = "Strong Progress"
        elif gain >= 10:
            trajectory = "Moderate Progress"
        else:
            trajectory = "Minimal Progress"
    else:
        trajectory = "Insufficient Data"

    rows.append({
        "country": c,
        "pri_2015": p15, "low_2015": l15, "upp_2015": u15, "col_2015": c15,
        "pri_2025": p25, "low_2025": l25, "upp_2025": u25, "col_2025": c25,
        "pri_1960": p60, "low_1960": l60, "upp_1960": u60, "col_1960": c60,
        "pri_1975": p75, "low_1975": l75,
        "pri_1990": p90, "low_1990": l90,
        "pri_2005": p05, "low_2005": l05,
        "edu_score_2015": edu15, "edu_score_1960": edu60,
        "ladder_score_2015": ladder15, "ladder_score_1960": ladder60,
        "ladder_score_2025": ladder25,
        "gain_1960_2015": gain,
        "year_crossed_60pct_primary": speed_year,
        "gender_gap_primary_2015": gender_gap,
        "gender_gap_lowsec_2015": gender_gap_low,
        "sequential_gap_2015": seq_gap15,
        "peak_decade": peak_decade, "peak_decade_gain": peak_gain,
        "archetype": archetype, "trajectory": trajectory,
    })

df = pd.DataFrame(rows)
df["world_rank_2015"] = df["ladder_score_2015"].rank(ascending=False, na_option="bottom").astype(int)
df["world_rank_gain"]  = df["gain_1960_2015"].rank(ascending=False, na_option="bottom").astype(int)
df["world_rank_speed"] = df["year_crossed_60pct_primary"].rank(ascending=True, na_option="bottom").astype(int)
df.sort_values("world_rank_2015", inplace=True)
df.reset_index(drop=True, inplace=True)

OUT_MD  = os.path.join(OUT, "rankings.md")
OUT_CSV = os.path.join(OUT, "rankings.csv")
df.to_csv(OUT_CSV, index=False, float_format="%.1f")
print(f"  CSV saved: {OUT_CSV}")

# ── Format helpers ────────────────────────────────────────────────────────────
def pct(val):
    return f"{val:.1f}%" if not np.isnan(val) else "n/a"

def signed(val):
    if np.isnan(val): return "n/a"
    return f"+{val:.1f}" if val >= 0 else f"{val:.1f}"

def cn(name, maxlen=34):
    return str(name)[:maxlen]

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

archetype_order = ["Universal","Secondary Building","Secondary Transition",
                   "Primary Complete","Primary Building","Low Access"]
archetype_desc  = {
    "Universal":            "Primary ≥92%, Lower Sec ≥85%, Upper Sec ≥70%",
    "Secondary Building":   "Primary ≥85%, Lower Sec 65–85% — primary mostly done, secondary still incomplete",
    "Secondary Transition": "Primary ≥80%, Lower Sec 40–65%",
    "Primary Complete":     "Primary ≥70%, Lower Sec <40%",
    "Primary Building":     "Primary 45–70%",
    "Low Access":           "Primary <45%",
}
traj_order = ["Regression","Early Achiever","Large Gain","Strong Progress",
              "Moderate Progress","Minimal Progress","Insufficient Data"]
traj_desc  = {
    "Regression":        "Edu Score declined ≥10 pp over the period",
    "Early Achiever":    "Edu Score ≥65 already in 1960, maintained",
    "Large Gain":        "Gained ≥40 pp — endpoint quality varies widely",
    "Strong Progress":   "Gained 22–40 pp",
    "Moderate Progress": "Gained 10–22 pp",
    "Minimal Progress":  "Gained <10 pp",
    "Insufficient Data": "Missing 1960 data",
}

n_countries = len(df)
h("# Global Education Achievement Rankings — WCDE v3 (1960–2025)")
h()
h(f"**{n_countries} countries** with complete data. Source: Wittgenstein Centre for Demography and Global Human Capital (WCDE v3).")
h("SSP2 (medium) scenario. Age group 20–24. 2020 and 2025 values are SSP2 projections.")
h()
h("| Metric | Definition |")
h("|---|---|")
h("| **Ladder Score** | Primary ranking metric. Weighted mean — college 2.5×, upper-sec 2×, lower-sec 1.5×, primary 1×. |")
h("| **Edu Score** | Simple mean of 4 completion levels. Secondary reference metric. |")
h("| **Gain** | Edu Score 2015 minus Edu Score 1960 |")
h("| **Speed** | First year country crossed 60% primary completion |")
h("| **Sequential Gap** | Primary minus lower-secondary 2015 |")
h()
h("---")
h()
h("## Summary Statistics")
h()
h("| | |")
h("|---|---|")
h(f"| Countries in dataset | {n_countries} |")
h(f"| Global Ladder Score 1960 | {df['ladder_score_1960'].mean():.1f} |")
h(f"| Global Ladder Score 2015 | {df['ladder_score_2015'].mean():.1f} |")
h(f"| Global Edu Score 1960 | {df['edu_score_1960'].mean():.1f} |")
h(f"| Global Edu Score 2015 | {df['edu_score_2015'].mean():.1f} |")
h(f"| Global Gain 1960→2015 | +{df['gain_1960_2015'].mean():.1f} pp |")
h(f"| Countries never reaching 60% primary by 2025 | {df['year_crossed_60pct_primary'].isna().sum()} |")
h(f"| Countries with gender gap >5 pp (girls behind) | {(df['gender_gap_primary_2015'] < -5).sum()} |")
h()
pipe_table(["Archetype","Countries","Definition"],
           [[a, len(df[df["archetype"]==a]), archetype_desc.get(a,"")] for a in archetype_order])
pipe_table(["Trajectory","Countries","Definition"],
           [[t, len(df[df["trajectory"]==t]), traj_desc.get(t,"")] for t in traj_order])
h("---")
h()

h("## Table 1 — World Ranking by 2015 Ladder Score")
h()
h("All countries ranked by Ladder Score (college 2.5×, upper-sec 2×, lower-sec 1.5×, primary 1×).")
h()
pipe_table(
    ["Rank","Country","Primary","Lower Sec","Upper Sec","College","Ladder Score","Edu Score","Archetype"],
    [[r.world_rank_2015, cn(r.country),
      pct(r.pri_2015), pct(r.low_2015), pct(r.upp_2015), pct(r.col_2015),
      f"{r.ladder_score_2015:.1f}", f"{r.edu_score_2015:.1f}", r.archetype]
     for _, r in df.iterrows()],
    ["right","left","right","right","right","right","right","right","left"]
)

h("---")
h()
h("## Table 2 — Most Improved 1960→2015")
h()
top_gain = df.sort_values("gain_1960_2015", ascending=False).head(60)
pipe_table(
    ["Rank","Country","Edu Score 1960","Edu Score 2015","Gain","Peak Decade","Peak Gain","Archetype"],
    [[i+1, cn(r.country),
      f"{r.edu_score_1960:.1f}" if not np.isnan(r.edu_score_1960) else "n/a",
      f"{r.edu_score_2015:.1f}" if not np.isnan(r.edu_score_2015) else "n/a",
      signed(r.gain_1960_2015), r.peak_decade,
      f"{r.peak_decade_gain:.1f}" if not np.isnan(r.peak_decade_gain) else "n/a",
      r.archetype]
     for i, (_, r) in enumerate(top_gain.iterrows())],
    ["right","left","right","right","right","left","right","left"]
)

h("---")
h()
h("## Table 3 — Archetype Groups")
h()
for arch in archetype_order:
    sub = df[df["archetype"] == arch].sort_values("edu_score_2015", ascending=False)
    if len(sub) == 0: continue
    h(f"### {arch} ({len(sub)} countries)")
    h(f"*{archetype_desc[arch]}*")
    h()
    pipe_table(
        ["Country","Primary","Lower Sec","Upper Sec","College","Edu Score","Gain","Trajectory"],
        [[cn(r.country), pct(r.pri_2015), pct(r.low_2015), pct(r.upp_2015), pct(r.col_2015),
          f"{r.edu_score_2015:.1f}", signed(r.gain_1960_2015), r.trajectory]
         for _, r in sub.iterrows()],
        ["left","right","right","right","right","right","right","left"]
    )

h("---")
h()
h("## Table 4 — Speed Rankings: First to Cross 60% Primary Completion")
h()
crossed = df.dropna(subset=["year_crossed_60pct_primary"]).sort_values("year_crossed_60pct_primary")
never   = df[df["year_crossed_60pct_primary"].isna()].sort_values("pri_2015", ascending=False)
h(f"{len(crossed)} countries crossed the threshold. {len(never)} had not crossed 60% primary by 2025.")
h()
pipe_table(
    ["Rank","Country","Crossed 60%","Primary 2015","Edu Score 2015"],
    [[i+1, cn(r.country), int(r.year_crossed_60pct_primary), pct(r.pri_2015), f"{r.edu_score_2015:.1f}"]
     for i, (_, r) in enumerate(crossed.iterrows())],
    ["right","left","right","right","right"]
)
h("**Countries that never crossed 60% primary by 2025:**")
h()
pipe_table(
    ["Country","Primary 2015","Primary 2025","Edu Score 2015"],
    [[cn(r.country), pct(r.pri_2015), pct(r.pri_2025), f"{r.edu_score_2015:.1f}"] for _, r in never.iterrows()],
    ["left","right","right","right"]
)

h("---")
h()
h("## Table 5 — Gender Gap Rankings (2015 Primary)")
h()
h("Female primary completion minus overall. **Negative** = girls behind.")
h()
gdf = df.dropna(subset=["gender_gap_primary_2015"]).sort_values("gender_gap_primary_2015")
h("**Worst 30 gender gaps:**")
h()
pipe_table(
    ["Country","Female Primary","Overall Primary","Gap (pp)"],
    [[cn(r.country), pct(v(f_pri_w, r.country, 2015)), pct(r.pri_2015), signed(r.gender_gap_primary_2015)]
     for _, r in gdf.head(30).iterrows()],
    ["left","right","right","right"]
)
h("**Best 20 — girls leading:**")
h()
pipe_table(
    ["Country","Female Primary","Overall Primary","Gap (pp)"],
    [[cn(r.country), pct(v(f_pri_w, r.country, 2015)), pct(r.pri_2015), signed(r.gender_gap_primary_2015)]
     for _, r in gdf.tail(20).iterrows()],
    ["left","right","right","right"]
)

h("---")
h()
h("## Table 6 — Sequential vs Simultaneous Expansion (2015)")
h()
sdf2 = df.dropna(subset=["sequential_gap_2015"]).sort_values("sequential_gap_2015")
simul_high = sdf2[(sdf2["sequential_gap_2015"] <= 15) & (sdf2["pri_2015"] >= 80)]
simul_low  = sdf2[(sdf2["sequential_gap_2015"] <= 15) & (sdf2["pri_2015"] < 60)]
h("**Simultaneous at scale — primary ≥80%, gap ≤15 pp:**")
h()
pipe_table(
    ["Country","Primary","Lower Sec","Gap (pp)","Edu Score"],
    [[cn(r.country), pct(r.pri_2015), pct(r.low_2015), signed(r.sequential_gap_2015), f"{r.edu_score_2015:.1f}"]
     for _, r in simul_high.iterrows()],
    ["left","right","right","right","right"]
)
h("**Low-base tied — primary <60%, gap ≤15 pp (both levels equally underdeveloped):**")
h()
pipe_table(
    ["Country","Primary","Lower Sec","Gap (pp)","Edu Score"],
    [[cn(r.country), pct(r.pri_2015), pct(r.low_2015), signed(r.sequential_gap_2015), f"{r.edu_score_2015:.1f}"]
     for _, r in simul_low.sort_values("pri_2015", ascending=False).iterrows()],
    ["left","right","right","right","right"]
)
h("**Most sequential — gap >30 pp:**")
h()
pipe_table(
    ["Country","Primary","Lower Sec","Gap (pp)","Edu Score"],
    [[cn(r.country), pct(r.pri_2015), pct(r.low_2015), signed(r.sequential_gap_2015), f"{r.edu_score_2015:.1f}"]
     for _, r in sdf2[sdf2["sequential_gap_2015"] > 30].sort_values("sequential_gap_2015", ascending=False).iterrows()],
    ["left","right","right","right","right"]
)

h("---")
h()
h("## Table 7 — Decade-by-Decade Trajectories (Key Countries)")
h()
key_countries = [
    "Republic of Korea","Singapore","Japan","China","Malaysia","Thailand",
    "Indonesia","Viet Nam","Philippines","Myanmar",
    "India","Bangladesh","Pakistan","Nepal","Sri Lanka",
    "Kenya","Ghana","Nigeria","Ethiopia","United Republic of Tanzania","Senegal",
    "Mozambique","Mali","Niger","Burkina Faso","South Africa","Rwanda",
    "Egypt","Morocco","Algeria","Iran (Islamic Republic of)","Turkey","Saudi Arabia",
    "Brazil","Mexico","Colombia","Peru","Bolivia (Plurinational State of)","Guatemala",
    "United States of America","Germany","Finland","France",
    "Taiwan Province of China","Hong Kong Special Administrative Region of China",
]
key_countries = [c for c in key_countries if c in df["country"].values]
kdf = df[df["country"].isin(key_countries)].copy()
kdf = kdf.set_index("country").loc[[c for c in key_countries if c in kdf["country"].values]].reset_index()
pipe_table(
    ["Country",
     "Pri 60","Pri 75","Pri 90","Pri 05","Pri 15","Pri 25",
     "Low 60","Low 75","Low 90","Low 05","Low 15","Low 25",
     "Ladder 15","Archetype"],
    [[cn(r.country,28),
      pct(r.pri_1960),pct(r.pri_1975),pct(r.pri_1990),pct(r.pri_2005),pct(r.pri_2015),pct(r.pri_2025),
      pct(r.low_1960),pct(r.low_1975),pct(r.low_1990),pct(r.low_2005),pct(r.low_2015),pct(r.low_2025),
      f"{r.ladder_score_2015:.1f}", r.archetype]
     for _, r in kdf.iterrows()],
    ["left"] + ["right"]*12 + ["right","left"]
)

h("---")
h()
h("## Table 8 — Trajectory Classification")
h()
for traj in traj_order:
    sub = df[df["trajectory"] == traj].sort_values("edu_score_2015", ascending=False)
    if len(sub) == 0: continue
    h(f"**{traj} ({len(sub)})** — *{traj_desc[traj]}*")
    h()
    h(", ".join(cn(c) for c in sub["country"]))
    h()

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"  Markdown saved: {OUT_MD}")
print(f"  Lines: {len(lines)}")
print("Done.")
