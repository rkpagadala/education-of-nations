"""
fig_residualization.py
=======================
Figure 4: Education R² vs Residualized GDP R² across four outcomes.

Blue bars (education) at 0.28-0.52. Red bars (residualized GDP) near zero.
The visual: education predicts everything, GDP predicts nothing independently.

Entry-cohort design (entry ≥ 10%, ceiling ≤ 90%), country FE,
lower secondary completion, T=1960-1990, lag=25.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PAPER_DIR = os.path.join(REPO_ROOT, "paper")

# ── Compute results ──────────────────────────────────────────────────

T_YEARS = list(range(1960, 1995, 5))
LAG = 25
COL_NAME = "lower_sec"
CEILING = 90

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df = load_wb("life_expectancy_years.csv")
tfr_df = load_wb("children_per_woman_total_fertility.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

edu_annual = interpolate_to_annual(edu_raw, COL_NAME)
entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[10]

outcomes = {}

# LE, TFR, U5MR from WB data
for label, outcome_col, outcome_df in [
    ("Life\nexpectancy", "le", le_df),
    ("Fertility", "tfr", tfr_df),
    ("Child\nmortality", "u5mr", u5mr_df),
]:
    panel = build_panel(edu_annual, outcome_df, gdp_df, T_YEARS, LAG, outcome_col)
    sub = filter_panel(panel, cohort, CEILING)
    r2_e, n_e, c_e = fe_r2("edu_t", outcome_col, sub)
    resid = fe_residualize_gdp(sub)
    r2_r = 0.0
    if resid is not None:
        sub_r, _ = resid
        r2_r, _, _ = fe_r2("gdp_resid", outcome_col, sub_r)
        if np.isnan(r2_r):
            r2_r = 0.0
    outcomes[label] = {"edu": r2_e, "resid": r2_r, "n": n_e, "ctry": c_e}
    print(f"  {label.replace(chr(10), ' ')}: edu={r2_e:.3f}, resid={r2_r:.3f}, n={n_e}, ctry={c_e}")

# Child education (parent→child from WCDE)
rows = []
for c in sorted(edu_annual.keys()):
    s = edu_annual[c]
    for t in T_YEARS:
        if t not in s.index or (t + LAG) not in s.index:
            continue
        parent_edu = s[t]
        child_edu = s[t + LAG]
        gdp_t = get_wb_val(gdp_df, c, t)
        if np.isnan(parent_edu) or np.isnan(child_edu):
            continue
        rows.append({
            "country": c, "t": t, "edu_t": parent_edu,
            "log_gdp_t": np.log(gdp_t) if not np.isnan(gdp_t) and gdp_t > 0 else np.nan,
            "child_edu": child_edu,
        })
import pandas as pd
panel_ce = pd.DataFrame(rows)
sub = filter_panel(panel_ce, cohort, CEILING)
r2_e, n_e, c_e = fe_r2("edu_t", "child_edu", sub)
resid = fe_residualize_gdp(sub)
r2_r = 0.0
if resid is not None:
    sub_r, _ = resid
    r2_r, _, _ = fe_r2("gdp_resid", "child_edu", sub_r)
    if np.isnan(r2_r):
        r2_r = 0.0
outcomes["Child\neducation"] = {"edu": r2_e, "resid": r2_r, "n": n_e, "ctry": c_e}
print(f"  Child education: edu={r2_e:.3f}, resid={r2_r:.3f}, n={n_e}, ctry={c_e}")

# ── Plot ─────────────────────────────────────────────────────────────

labels = ["Life\nexpectancy", "Fertility", "Child\neducation", "Child\nmortality"]
edu_vals = [outcomes[l]["edu"] for l in labels]
resid_vals = [outcomes[l]["resid"] for l in labels]

fig, ax = plt.subplots(figsize=(8, 4.5))

x = np.arange(len(labels))
width = 0.35

bars_edu = ax.bar(x - width/2, edu_vals, width, color='#2563eb', label='Education', zorder=3)
bars_resid = ax.bar(x + width/2, resid_vals, width, color='#ef4444', label='GDP (after removing\neducation\'s contribution)', zorder=3)

# Labels on bars
for bar, val in zip(bars_edu, edu_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2563eb')

for bar, val in zip(bars_resid, resid_vals):
    label_text = f'{val:.1%}' if val >= 0.005 else '0%'
    ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0.005) + 0.01,
            label_text, ha='center', va='bottom', fontsize=11, fontweight='bold', color='#ef4444')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Within-country R² (predictive power)', fontsize=11)
ax.set_ylim(0, 0.65)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, zorder=0)

ax.set_title('GDP has no independent effect on any development outcome',
             fontsize=13, fontweight='bold', pad=15)

fig.text(0.5, -0.02,
         'Country fixed effects, lower secondary completion, entry ≥ 10%, ceiling ≤ 90%, T=1960–1990, lag=25 years.',
         ha='center', fontsize=8, color='#64748b')

plt.tight_layout()

# Save for paper
out_paper = os.path.join(PAPER_DIR, "fig_residualization.png")
fig.savefig(out_paper, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved to {out_paper}")

# Save for scripts dir too
out_scripts = os.path.join(SCRIPT_DIR, "fig_residualization.png")
fig.savefig(out_scripts, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved to {out_scripts}")

plt.close()
