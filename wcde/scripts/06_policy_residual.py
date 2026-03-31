"""
06_policy_residual.py
Policy-adjusted education ranking on WCDE v3 data.

Which countries delivered more lower-secondary education than their
parental education history predicts?

Method:
  For each country-year, regress lower secondary completion on:
    - parental lower secondary (T-25)
  The residual = actual - predicted = policy over/under-performance.

GDP is NOT used as a predictor: education causes GDP (education_outcomes.md),
so controlling for GDP would block part of the education signal via the income
channel (bad control / mediation problem). The residual here measures pure
policy contribution above the intergenerational inheritance baseline.

GDP is loaded for display in the output tables only.
Country name matching: WCDE names → World Bank lowercase names.

Outputs: wcde/output/policy_residual.md
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "../data/processed")
OUT  = os.path.join(SCRIPT_DIR, "../output")
ROOT_DS = os.path.join(SCRIPT_DIR, "../../datasets/")
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

OBS_YEARS_STR = ["1985","1990","1995","2000","2005","2010","2015"]

# Manual WCDE → World Bank name mapping (only cases needing translation)
NAME_MAP = {
    "viet nam": "vietnam",
    "iran (islamic republic of)": "iran",
    "bolivia (plurinational state of)": "bolivia",
    "republic of korea": "south korea",
    "democratic republic of the congo": "congo, dem. rep.",
    "congo": "congo, rep.",
    "united republic of tanzania": "tanzania",
    "united states of america": "united states",
    "united kingdom of great britain and northern ireland": "united kingdom",
    "russian federation": "russia",
    "venezuela (bolivarian republic of)": "venezuela",
    "republic of moldova": "moldova",
    "syrian arab republic": "syria",
    "libyan arab jamahiriya": "libya",
    "taiwan province of china": "taiwan",
    "lao people's democratic republic": "laos",
    "hong kong special administrative region of china": "hong kong",
    "macao special administrative region of china": "macao",
    "côte d'ivoire": "cote d'ivoire",
    "democratic people's republic of korea": "north korea",
    "myanmar": "myanmar",
    "cabo verde": "cape verde",
    "czechia": "czech republic",
    "north macedonia": "macedonia",
    "türkiye": "turkey",
    "eswatini": "swaziland",
}

print("Loading data...")
both = pd.read_csv(os.path.join(PROC, "completion_both_long.csv"))
both = both[~both["country"].isin(REGIONS)].copy()
for col in ["primary","lower_sec","upper_sec","college"]:
    both[col] = both[col].clip(upper=100)

low_w = both.pivot(index="country", columns="year", values="lower_sec")
pri_w = both.pivot(index="country", columns="year", values="primary")

# Load GDP
try:
    gdp = pd.read_csv(ROOT_DS + "gdppercapita_us_inflation_adjusted.csv")
    gdp["Country"] = gdp["Country"].str.lower()
    gdp = gdp.set_index("Country")
    for c in gdp.columns:
        gdp[c] = pd.to_numeric(gdp[c], errors="coerce")
    print(f"  GDP: {len(gdp)} countries")
except Exception as e:
    print(f"  GDP load failed: {e}")
    gdp = None

def v(df_w, country, year):
    try:
        val = float(df_w.loc[country, int(year)])
        return val if not np.isnan(val) else np.nan
    except:
        return np.nan

def get_gdp(country_wcde, year_str):
    if gdp is None: return np.nan
    c_lower = country_wcde.lower()
    # Try direct match
    if c_lower in gdp.index:
        try:
            val = float(gdp.loc[c_lower, year_str])
            return val if not np.isnan(val) and val > 0 else np.nan
        except:
            return np.nan
    # Try name map
    mapped = NAME_MAP.get(c_lower)
    if mapped and mapped in gdp.index:
        try:
            val = float(gdp.loc[mapped, year_str])
            return val if not np.isnan(val) and val > 0 else np.nan
        except:
            return np.nan
    return np.nan

# Build panel: T-25 lag
CHILD_YRS  = [1985,1990,1995,2000,2005,2010,2015]
PARENT_YRS = [1960,1965,1970,1975,1980,1985,1990]

panel_rows = []
for c in low_w.index:
    for child_yr, parent_yr in zip(CHILD_YRS, PARENT_YRS):
        child_low  = v(low_w, c, child_yr)
        parent_low = v(low_w, c, parent_yr)
        pri_val    = v(pri_w, c, child_yr)
        gdp_val    = get_gdp(c, str(child_yr))

        if any(np.isnan(x) for x in [child_low, parent_low]):
            continue

        panel_rows.append({
            "country": c,
            "year": child_yr,
            "child_low": child_low,
            "parent_low": parent_low,
            "log_gdp": np.log(gdp_val) if not np.isnan(gdp_val) else np.nan,  # display only
            "pri": pri_val,
        })

panel = pd.DataFrame(panel_rows)
print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")

if len(panel) < 10:
    print("  WARNING: too few observations for regression. Check GDP data.")
    exit(1)

# Country FE regression — parental education only (no GDP: bad control / mediation)
panel["child_low_dm"]  = panel["child_low"]  - panel.groupby("country")["child_low"].transform("mean")
panel["parent_low_dm"] = panel["parent_low"] - panel.groupby("country")["parent_low"].transform("mean")

X_fe = panel[["parent_low_dm"]].values
y_fe = panel["child_low_dm"].values
ok   = ~np.isnan(X_fe).any(axis=1) & ~np.isnan(y_fe)
reg_fe = LinearRegression(fit_intercept=False).fit(X_fe[ok], y_fe[ok])
panel.loc[ok, "fitted_dm"] = reg_fe.predict(X_fe[ok])

panel["child_mean"] = panel.groupby("country")["child_low"].transform("mean")
panel["fitted"]     = panel["child_mean"] + panel["fitted_dm"].fillna(0)
panel["residual"]   = panel["child_low"] - panel["fitted"]

print(f"  FE coef: parental={reg_fe.coef_[0]:.3f}")

# Pooled OLS residual — parental education only
X_ols = panel[["parent_low"]].values
y_ols = panel["child_low"].values
ok2   = ~np.isnan(X_ols).any(axis=1) & ~np.isnan(y_ols)
reg_ols = LinearRegression().fit(X_ols[ok2], y_ols[ok2])
panel.loc[ok2, "fitted_ols"] = reg_ols.predict(X_ols[ok2])
panel["resid_ols"] = panel["child_low"] - panel["fitted_ols"]

print(f"  OLS coef: parental={reg_ols.coef_[0]:.3f}")

# Country summaries
summary_rows = []
for c, grp in panel.groupby("country"):
    grp15 = grp[grp["year"] == 2015]
    resid_2015    = grp15["residual"].values[0]    if len(grp15) > 0 and not grp15["residual"].isna().all()    else np.nan
    resid_ols2015 = grp15["resid_ols"].values[0]   if len(grp15) > 0 and not grp15["resid_ols"].isna().all()  else np.nan
    low2015 = grp15["child_low"].values[0] if len(grp15) > 0 else np.nan
    pri2015 = grp15["pri"].values[0]       if len(grp15) > 0 else np.nan
    gdp2015 = np.exp(grp15["log_gdp"].values[0]) if len(grp15) > 0 and not grp15["log_gdp"].isna().all() else np.nan
    mean_resid_ols = grp["resid_ols"].mean()
    mean_resid_fe  = grp["residual"].mean()
    summary_rows.append({
        "country": c,
        "resid_fe_2015":   resid_2015,
        "resid_ols_2015":  resid_ols2015,
        "mean_resid_fe":   mean_resid_fe,
        "mean_resid_ols":  mean_resid_ols,
        "low_2015": low2015,
        "pri_2015": pri2015,
        "gdp_2015": gdp2015,
    })

sdf = pd.DataFrame(summary_rows)

def cn(name, maxlen=32): return str(name)[:maxlen]
def pct(val): return f"{val:.1f}%" if not np.isnan(val) else "n/a"
def pp(val):
    if np.isnan(val): return "n/a"
    return f"+{val:.1f} pp" if val >= 0 else f"{val:.1f} pp"
def gdp_fmt(val): return f"${val:,.0f}" if not np.isnan(val) else "n/a"

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

h("# Policy-Adjusted Education Ranking — WCDE v3")
h()
h("*Which countries delivered more lower-secondary education than their parental education history predicts?*")
h()
h("## Method")
h()
h("**Target:** lower secondary completion rate (% of 20–24 cohort) at year T")
h("**Predictor:** parental lower secondary completion at T−25")
h("**Model:** country fixed effects (within-country variation only)")
h()
h("GDP is intentionally excluded as a predictor. Because education causes GDP")
h("(see education_outcomes.md), controlling for current income would block part of the")
h("education signal via the income channel — a bad control / mediation problem. Countries")
h("like Korea and Singapore that invested early in education, grew rich as a result, and")
h("continued investing would otherwise show as under-performers because the model assigns")
h("credit for their education gains to their income. The residual here measures policy")
h("contribution above the intergenerational inheritance baseline only. GDP is shown in")
h("the tables for context.")
h()
h(f"Fixed effects coefficient:")
h(f"- Parental lower secondary: **{reg_fe.coef_[0]:.3f}** pp per 1 pp of parental completion")
h(f"- Panel: {len(panel)} obs, {panel['country'].nunique()} countries")
h()
h("**Residual = actual − predicted.** Positive residual = policy over-performance.")
h()
h("---")
h()

h("## Table 1 — Biggest Over-Performers in 2015 (FE Residual)")
h()
over = sdf.dropna(subset=["resid_fe_2015"]).sort_values("resid_fe_2015", ascending=False)
pipe_table(
    ["Rank","Country","Low Sec 2015","FE Residual","OLS Residual","GDP/capita 2015"],
    [[i+1, cn(r.country), pct(r.low_2015), pp(r.resid_fe_2015), pp(r.resid_ols_2015), gdp_fmt(r.gdp_2015)]
     for i, (_, r) in enumerate(over.head(30).iterrows())],
    ["right","left","right","right","right","right"]
)

h("## Table 2 — Biggest Under-Performers in 2015 (FE Residual)")
h()
under = sdf.dropna(subset=["resid_fe_2015"]).sort_values("resid_fe_2015")
pipe_table(
    ["Rank","Country","Low Sec 2015","FE Residual","OLS Residual","GDP/capita 2015"],
    [[i+1, cn(r.country), pct(r.low_2015), pp(r.resid_fe_2015), pp(r.resid_ols_2015), gdp_fmt(r.gdp_2015)]
     for i, (_, r) in enumerate(under.head(30).iterrows())],
    ["right","left","right","right","right","right"]
)

h("## Table 3 — Chronic Over-Performers (Mean OLS Residual Across All Years)")
h()
h("Countries that consistently outperformed the global cross-country prediction across all years.")
h()
chronic_over = sdf.dropna(subset=["mean_resid_ols"]).sort_values("mean_resid_ols", ascending=False)
pipe_table(
    ["Rank","Country","Low Sec 2015","Mean OLS Residual","2015 OLS Residual","2015 FE Residual"],
    [[i+1, cn(r.country), pct(r.low_2015), pp(r.mean_resid_ols), pp(r.resid_ols_2015), pp(r.resid_fe_2015)]
     for i, (_, r) in enumerate(chronic_over.head(30).iterrows())],
    ["right","left","right","right","right","right"]
)

h("## Table 4 — Chronic Under-Performers (Mean OLS Residual)")
h()
chronic_under = sdf.dropna(subset=["mean_resid_ols"]).sort_values("mean_resid_ols")
pipe_table(
    ["Rank","Country","Low Sec 2015","Mean OLS Residual","2015 OLS Residual","2015 FE Residual"],
    [[i+1, cn(r.country), pct(r.low_2015), pp(r.mean_resid_ols), pp(r.resid_ols_2015), pp(r.resid_fe_2015)]
     for i, (_, r) in enumerate(chronic_under.head(30).iterrows())],
    ["right","left","right","right","right","right"]
)

h("## Table 5 — Full Country Ranking by 2015 FE Residual")
h()
full = sdf.dropna(subset=["resid_fe_2015"]).sort_values("resid_fe_2015", ascending=False).reset_index(drop=True)
pipe_table(
    ["Rank","Country","Low Sec 2015","FE Residual","OLS Residual"],
    [[i+1, cn(r.country), pct(r.low_2015), pp(r.resid_fe_2015), pp(r.resid_ols_2015)]
     for i, r in full.iterrows()],
    ["right","left","right","right","right"]
)

h("---")
h()
h("*Method: country fixed effects regression of lower secondary completion on parental lower secondary (T−25) only.*")
h("*GDP excluded to avoid bad-control bias (education → GDP). GDP shown for context only. WCDE v3 education data.*")

with open(os.path.join(OUT, "policy_residual.md"), "w") as f:
    f.write("\n".join(lines))
print(f"  Saved: {os.path.join(OUT, 'policy_residual.md')}")
print("Done.")
