"""
Fixed-effects and first-differences analysis to address the two methodological gaps:

1. Country fixed-effects: does within-country change in parental education
   predict within-country change in child education, after removing all
   time-invariant country-level factors (colonial history, institutions,
   culture, geography)?

2. First-differences: regress Δchild_edu on Δparental_edu + ΔGDP
   — eliminates all level-based correlation, tests only co-movement

3. Within-country R² decomposition: how much variation is between countries
   vs within countries, and which predictor explains the within-country part?

4. Placebo test: does a random 25-year lag of an unrelated variable (CO2)
   predict child education as well as parental education?
   If yes — the generational finding is spurious time-trend correlation.
   If no — the finding is specific to education transmission.

5. Heterogeneity: does the within-country effect hold equally for
   tigers, non-tigers, rich, poor?
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
WCDE_PROC = os.path.join(REPO_ROOT, "wcde", "data", "processed")
WB_DIR = os.path.join(REPO_ROOT, "data")  # World Bank WDI downloads

# WCDE v3 education data (completion %, 5-year intervals)
# Converted to OL (out-of-school) and interpolated to annual for this script
def load_wcde_as_ol(filename):
    """Load WCDE completion CSV, convert to OL rate, interpolate to annual."""
    df = pd.read_csv(os.path.join(WCDE_PROC, filename), index_col="country")
    df.columns = [int(c) for c in df.columns]
    ol = 100.0 - df  # completion → out-of-school
    # Interpolate to annual
    all_years = list(range(min(df.columns), max(df.columns) + 1))
    ol = ol.reindex(columns=all_years).interpolate(axis=1, method="linear")
    ol = ol.bfill(axis=1).ffill(axis=1)
    ol.columns = [str(c) for c in ol.columns]
    ol.index = ol.index.str.lower()
    return ol

def load_wb(filename):
    """Load World Bank WDI CSV (Country x Year wide format)."""
    df = pd.read_csv(os.path.join(WB_DIR, filename))
    df["Country"] = df["Country"].str.lower()
    return df.set_index("Country")

DATASETS = {}  # populated below after function defs

PARENTAL_LAG = 25
SCHOOLING_LAG = 12
TIGERS = ["south korea", "singapore", "malaysia", "thailand"]

def load(path):
    df = pd.read_csv(path)
    df["Country"] = df["Country"].str.lower()
    return df.set_index("Country")

print("Loading data...")
# WCDE education data (converted to OL rate, interpolated to annual, lowercased)
dfs = {
    "Primary_OL":         load_wcde_as_ol("primary_both.csv"),
    "Lower_Secondary_OL": load_wcde_as_ol("lower_sec_both.csv"),
    "Higher_Secondary":   load_wcde_as_ol("upper_sec_both.csv"),
    "female_Primary_OL":  load_wcde_as_ol("primary_female.csv"),
}
# World Bank data
for name, filename in [
    ("gdp",             "gdppercapita_us_inflation_adjusted.csv"),
    ("co2",             "co2_emissions_tonnes_per_person.csv"),
    ("tfr",             "children_per_woman_total_fertility.csv"),
    ("life_expectancy", "life_expectancy_years.csv"),
]:
    dfs[name] = load_wb(filename)

all_countries = sorted(
    set(dfs["gdp"].index) &
    set(dfs["Primary_OL"].index) &
    set(dfs["co2"].index)
)

def get(name, country, year):
    try:
        return float(dfs[name].loc[country, str(year)])
    except (KeyError, ValueError):
        return np.nan

def comp(ol): return 100 - ol

# ── Build panel ───────────────────────────────────────────────────────────────
print("Building panel...")
rows = []
for country in all_countries:
    for yr in range(1985, 2016):
        child_pri    = comp(get("Primary_OL",         country, yr))
        child_low    = comp(get("Lower_Secondary_OL", country, yr))
        parent_pri   = comp(get("Primary_OL",         country, yr - PARENTAL_LAG))
        parent_f_pri = comp(get("female_Primary_OL",  country, yr - PARENTAL_LAG))
        gdp_school   = get("gdp",  country, yr - SCHOOLING_LAG)
        co2_lag      = get("co2",  country, yr - PARENTAL_LAG)   # placebo
        if any(np.isnan(v) for v in [child_pri, parent_pri, gdp_school]):
            continue
        rows.append({
            "country":       country,
            "year":          yr,
            "child_pri":     child_pri,
            "child_low":     child_low if not np.isnan(child_low) else np.nan,
            "parent_pri":    parent_pri,
            "parent_f_pri":  parent_f_pri,
            "log_gdp":       np.log(gdp_school) if gdp_school > 0 else np.nan,
            "co2_lag":       co2_lag,
            "is_tiger":      1 if country in TIGERS else 0,
            "log_gdp_level": np.log(get("gdp", country, yr)) if get("gdp", country, yr) > 0 else np.nan,
        })

panel = pd.DataFrame(rows).dropna(subset=["child_pri","parent_pri","log_gdp"])
print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries\n")

def r2(y, y_hat):
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

def ols(X, y):
    reg = LinearRegression().fit(X, y)
    return reg, reg.score(X, y)

# ── SECTION 1: Variance decomposition ────────────────────────────────────────
print("="*68)
print("  SECTION 1: Variance decomposition")
print("  How much variation is between-country vs within-country?")
print("="*68)

for col, label in [("child_pri","child primary"), ("parent_pri","parental primary"),
                   ("log_gdp","log GDP at schooling age")]:
    total_var   = panel[col].var()
    country_means = panel.groupby("country")[col].transform("mean")
    between_var = country_means.var()
    within_var  = (panel[col] - country_means).var()
    print(f"\n  {label}:")
    print(f"    Total variance:   {total_var:.2f}")
    print(f"    Between-country:  {between_var:.2f}  ({between_var/total_var*100:.0f}%)")
    print(f"    Within-country:   {within_var:.2f}  ({within_var/total_var*100:.0f}%)")

# ── SECTION 2: Pooled OLS vs country fixed effects ───────────────────────────
print(f"\n{'='*68}")
print(f"  SECTION 2: Pooled OLS vs country fixed effects")
print(f"  Target: child primary completion")
print(f"{'='*68}")

p = panel.dropna(subset=["child_pri","parent_pri","log_gdp","parent_f_pri"])

# Pooled OLS
_, r2_pooled_par = ols(p[["parent_pri"]].values,   p["child_pri"].values)
_, r2_pooled_gdp = ols(p[["log_gdp"]].values,      p["child_pri"].values)
_, r2_pooled_both= ols(p[["parent_pri","log_gdp"]].values, p["child_pri"].values)

print(f"\n  Pooled OLS (no fixed effects):")
print(f"    Parental edu only:          R² = {r2_pooled_par:.3f}")
print(f"    GDP only:                   R² = {r2_pooled_gdp:.3f}")
print(f"    Both:                       R² = {r2_pooled_both:.3f}")

# Country fixed effects: demean within country
p2 = p.copy()
for col in ["child_pri","parent_pri","log_gdp","parent_f_pri","child_low"]:
    if col in p2.columns:
        means = p2.groupby("country")[col].transform("mean")
        p2[col+"_dm"] = p2[col] - means   # demeaned = within-country variation only

_, r2_fe_par  = ols(p2[["parent_pri_dm"]].values,              p2["child_pri_dm"].values)
_, r2_fe_gdp  = ols(p2[["log_gdp_dm"]].values,                 p2["child_pri_dm"].values)
_, r2_fe_both = ols(p2[["parent_pri_dm","log_gdp_dm"]].values, p2["child_pri_dm"].values)
_, r2_fe_fpar = ols(p2[["parent_f_pri_dm"]].values,            p2["child_pri_dm"].values)
_, r2_fe_fboth= ols(p2[["parent_f_pri_dm","log_gdp_dm"]].values,p2["child_pri_dm"].values)

print(f"\n  Country fixed effects (within-country variation only):")
print(f"    Parental edu only:          R² = {r2_fe_par:.3f}")
print(f"    GDP only:                   R² = {r2_fe_gdp:.3f}")
print(f"    Both:                       R² = {r2_fe_both:.3f}")
print(f"    Female parental edu only:   R² = {r2_fe_fpar:.3f}")
print(f"    Female parental + GDP:      R² = {r2_fe_fboth:.3f}")

incr_gdp = r2_fe_both - r2_fe_par
incr_par = r2_fe_both - r2_fe_gdp
print(f"\n  Within-country incremental R²:")
print(f"    Adding GDP to parental-edu model:   +{incr_gdp:.3f}")
print(f"    Adding parental edu to GDP model:   +{incr_par:.3f}")

# Same for lower secondary
p3 = p2.dropna(subset=["child_low_dm"])
_, r2_fe_par_low  = ols(p3[["parent_pri_dm"]].values,              p3["child_low_dm"].values)
_, r2_fe_gdp_low  = ols(p3[["log_gdp_dm"]].values,                 p3["child_low_dm"].values)
_, r2_fe_both_low = ols(p3[["parent_pri_dm","log_gdp_dm"]].values, p3["child_low_dm"].values)
_, r2_fe_fpar_low = ols(p3[["parent_f_pri_dm"]].values,            p3["child_low_dm"].values)

print(f"\n  Country fixed effects — target: child LOWER SECONDARY:")
print(f"    Parental primary only:      R² = {r2_fe_par_low:.3f}")
print(f"    GDP only:                   R² = {r2_fe_gdp_low:.3f}")
print(f"    Both:                       R² = {r2_fe_both_low:.3f}")
print(f"    Female parental only:       R² = {r2_fe_fpar_low:.3f}")
print(f"    Incremental GDP:            +{r2_fe_both_low - r2_fe_par_low:.3f}")
print(f"    Incremental parental edu:   +{r2_fe_both_low - r2_fe_gdp_low:.3f}")

# ── SECTION 3: First differences ─────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  SECTION 3: First differences (5-year intervals)")
print(f"  Δchild_edu ~ Δparental_edu + ΔGDP")
print(f"  Eliminates ALL levels — only tests co-movement of changes")
print(f"{'='*68}")

fd_rows = []
years_5 = list(range(1985, 2016, 5))
for country in all_countries:
    for i in range(len(years_5)-1):
        y0, y1 = years_5[i], years_5[i+1]
        d_child  = comp(get("Primary_OL", country, y1)) - comp(get("Primary_OL", country, y0))
        d_child_low = (comp(get("Lower_Secondary_OL", country, y1)) -
                       comp(get("Lower_Secondary_OL", country, y0)))
        d_parent = (comp(get("Primary_OL", country, y1-PARENTAL_LAG)) -
                    comp(get("Primary_OL", country, y0-PARENTAL_LAG)))
        d_fparent= (comp(get("female_Primary_OL", country, y1-PARENTAL_LAG)) -
                    comp(get("female_Primary_OL", country, y0-PARENTAL_LAG)))
        g0 = get("gdp", country, y0-SCHOOLING_LAG)
        g1 = get("gdp", country, y1-SCHOOLING_LAG)
        d_log_gdp = (np.log(g1) - np.log(g0)) if (g0 > 0 and g1 > 0) else np.nan
        c0 = get("co2", country, y0-PARENTAL_LAG)
        c1 = get("co2", country, y1-PARENTAL_LAG)
        d_co2 = (c1 - c0) if not (np.isnan(c0) or np.isnan(c1)) else np.nan

        if any(np.isnan(v) for v in [d_child, d_parent, d_log_gdp]):
            continue
        fd_rows.append({
            "country": country, "year": y1,
            "d_child": d_child, "d_child_low": d_child_low,
            "d_parent": d_parent, "d_fparent": d_fparent,
            "d_log_gdp": d_log_gdp, "d_co2": d_co2,
            "is_tiger": 1 if country in TIGERS else 0,
        })

fd = pd.DataFrame(fd_rows).dropna(subset=["d_child","d_parent","d_log_gdp"])
print(f"\n  First-differences panel: {len(fd)} obs, {fd['country'].nunique()} countries")

_, r2_fd_par  = ols(fd[["d_parent"]].values,             fd["d_child"].values)
_, r2_fd_gdp  = ols(fd[["d_log_gdp"]].values,            fd["d_child"].values)
_, r2_fd_both = ols(fd[["d_parent","d_log_gdp"]].values, fd["d_child"].values)
_, r2_fd_fpar = ols(fd[["d_fparent"]].values,            fd["d_child"].values)
_, r2_fd_fboth= ols(fd[["d_fparent","d_log_gdp"]].values,fd["d_child"].values)

print(f"\n  First differences — target: Δchild primary:")
print(f"    Δparental edu only:         R² = {r2_fd_par:.3f}")
print(f"    ΔGDP only:                  R² = {r2_fd_gdp:.3f}")
print(f"    Both:                       R² = {r2_fd_both:.3f}")
print(f"    Δfemale parental only:      R² = {r2_fd_fpar:.3f}")
print(f"    Δfemale parental + ΔGDP:    R² = {r2_fd_fboth:.3f}")
print(f"    Incremental ΔGDP:           +{r2_fd_both - r2_fd_par:.3f}")
print(f"    Incremental Δparental edu:  +{r2_fd_both - r2_fd_gdp:.3f}")

fd2 = fd.dropna(subset=["d_child_low"])
_, r2_fd_par_low  = ols(fd2[["d_parent"]].values,             fd2["d_child_low"].values)
_, r2_fd_gdp_low  = ols(fd2[["d_log_gdp"]].values,            fd2["d_child_low"].values)
_, r2_fd_both_low = ols(fd2[["d_parent","d_log_gdp"]].values, fd2["d_child_low"].values)

print(f"\n  First differences — target: Δchild lower secondary:")
print(f"    Δparental edu only:         R² = {r2_fd_par_low:.3f}")
print(f"    ΔGDP only:                  R² = {r2_fd_gdp_low:.3f}")
print(f"    Both:                       R² = {r2_fd_both_low:.3f}")
print(f"    Incremental ΔGDP:           +{r2_fd_both_low - r2_fd_par_low:.3f}")
print(f"    Incremental Δparental edu:  +{r2_fd_both_low - r2_fd_gdp_low:.3f}")

# ── SECTION 4: Placebo test ───────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  SECTION 4: Placebo test — CO2 emissions lagged 25 years")
print(f"  If time-trend drives the parental-edu finding, CO2(T-25)")
print(f"  should also predict child education (it also trends over time).")
print(f"  If the finding is specific: CO2 placebo should be much weaker.")
print(f"{'='*68}")

p_co2 = panel.dropna(subset=["co2_lag","child_pri","log_gdp"])
p_co2 = p_co2.copy()
for col in ["child_pri","co2_lag","log_gdp","parent_pri"]:
    means = p_co2.groupby("country")[col].transform("mean")
    p_co2[col+"_dm"] = p_co2[col] - means

_, r2_co2_pooled = ols(p_co2[["co2_lag"]].values,       p_co2["child_pri"].values)
_, r2_co2_fe     = ols(p_co2[["co2_lag_dm"]].values,    p_co2["child_pri_dm"].values)

print(f"\n  Pooled OLS:")
print(f"    CO2(T-25) alone:            R² = {r2_co2_pooled:.3f}  (compare: parental edu = {r2_pooled_par:.3f})")

p_co2_par = p_co2.dropna(subset=["parent_pri_dm"])
_, r2_par_fe_co2 = ols(p_co2_par[["parent_pri_dm"]].values, p_co2_par["child_pri_dm"].values)
print(f"\n  Fixed effects:")
print(f"    CO2(T-25) alone:            R² = {r2_co2_fe:.3f}  (compare: parental edu = {r2_par_fe_co2:.3f})")

fd_co2 = fd.dropna(subset=["d_co2"])
_, r2_co2_fd = ols(fd_co2[["d_co2"]].values, fd_co2["d_child"].values)
_, r2_par_fd_co2 = ols(fd_co2[["d_parent"]].values, fd_co2["d_child"].values)
print(f"\n  First differences:")
print(f"    ΔCO2(T-25) alone:           R² = {r2_co2_fd:.3f}  (compare: Δparental edu = {r2_par_fd_co2:.3f})")

# ── SECTION 5: Heterogeneity — tigers vs rest ─────────────────────────────────
print(f"\n{'='*68}")
print(f"  SECTION 5: Heterogeneity — tigers vs non-tigers, rich vs poor")
print(f"  Fixed-effects within each subgroup")
print(f"{'='*68}")

for group_name, mask in [
    ("Tigers (KOR, SGP, MYS, THA)", p2["country"].isin(TIGERS)),
    ("Non-tigers",                  ~p2["country"].isin(TIGERS)),
    ("Rich (GDP > world median)",   p2["log_gdp_level"] > p2["log_gdp_level"].median()),
    ("Poor (GDP < world median)",   p2["log_gdp_level"] <= p2["log_gdp_level"].median()),
]:
    sub = p2[mask].dropna(subset=["child_pri_dm","parent_pri_dm","log_gdp_dm"])
    if len(sub) < 20:
        print(f"\n  {group_name}: insufficient data")
        continue
    _, r2_sub_par  = ols(sub[["parent_pri_dm"]].values,              sub["child_pri_dm"].values)
    _, r2_sub_gdp  = ols(sub[["log_gdp_dm"]].values,                 sub["child_pri_dm"].values)
    _, r2_sub_both = ols(sub[["parent_pri_dm","log_gdp_dm"]].values, sub["child_pri_dm"].values)
    print(f"\n  {group_name}  (n={len(sub)}):")
    print(f"    Parental edu (FE):          R² = {r2_sub_par:.3f}")
    print(f"    GDP (FE):                   R² = {r2_sub_gdp:.3f}")
    print(f"    Both (FE):                  R² = {r2_sub_both:.3f}")
    print(f"    Incremental parental edu:   +{r2_sub_both - r2_sub_gdp:.3f}")
    print(f"    Incremental GDP:            +{r2_sub_both - r2_sub_par:.3f}")

# ── SECTION 6: Coefficients — magnitude of effect ────────────────────────────
print(f"\n{'='*68}")
print(f"  SECTION 6: Effect sizes — what does 1pp more parental education")
print(f"  predict for child education? (fixed-effects coefficients)")
print(f"{'='*68}")

reg_fe, _ = ols(p2[["parent_pri_dm","log_gdp_dm"]].values, p2["child_pri_dm"].values)
print(f"\n  Within-country OLS (demeaned), target: child primary:")
print(f"    1pp more parental primary edu → {reg_fe.coef_[0]:.3f}pp more child primary")
print(f"    1% more GDP at schooling age  → {reg_fe.coef_[1]:.3f}pp more child primary")

reg_fd, _ = ols(fd[["d_parent","d_log_gdp"]].values, fd["d_child"].values)
print(f"\n  First differences, target: Δchild primary:")
print(f"    1pp Δparental primary edu  → {reg_fd.coef_[0]:.3f}pp Δchild primary")
print(f"    1% ΔGDP at schooling age   → {reg_fd.coef_[1]:.3f}pp Δchild primary")

reg_fd_low, _ = ols(fd2[["d_parent","d_log_gdp"]].values, fd2["d_child_low"].values)
print(f"\n  First differences, target: Δchild lower secondary:")
print(f"    1pp Δparental primary edu  → {reg_fd_low.coef_[0]:.3f}pp Δchild lower secondary")
print(f"    1% ΔGDP at schooling age   → {reg_fd_low.coef_[1]:.3f}pp Δchild lower secondary")

print(f"\n{'='*68}")
print(f"  SUMMARY")
print(f"{'='*68}")
print(f"\n  Pooled OLS R²:   parental edu={r2_pooled_par:.3f}  GDP={r2_pooled_gdp:.3f}")
print(f"  Fixed effects R²: parental edu={r2_fe_par:.3f}  GDP={r2_fe_gdp:.3f}")
print(f"  First diff R²:    parental edu={r2_fd_par:.3f}  GDP={r2_fd_gdp:.3f}")
print(f"  Placebo (CO2) FE R²: {r2_co2_fe:.3f}  vs parental edu FE: {r2_par_fe_co2:.3f}")
print(f"\n  If the generational finding survives fixed effects and first differences")
print(f"  AND the CO2 placebo is weak, the finding is not a time-trend artefact.")
