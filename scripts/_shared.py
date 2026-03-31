"""
_shared.py
==========
Shared data loading, country mapping, and regression utilities
for the edu_vs_gdp analysis scripts.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROC = os.path.join(REPO_ROOT, "wcde", "data", "processed")
DATA = os.path.join(REPO_ROOT, "data")
CHECKIN = os.path.join(REPO_ROOT, "checkin")

REGIONS = {
    "Africa", "Asia", "Europe", "World", "Oceania", "Caribbean",
    "Central America", "South America",
    "Latin America and the Caribbean",
    "Central Asia", "Eastern Africa", "Eastern Asia", "Eastern Europe",
    "Northern Africa", "Northern America", "Northern Europe",
    "Southern Africa", "Southern Asia", "Southern Europe",
    "Western Africa", "Western Asia", "Western Europe",
    "Middle Africa", "South-Eastern Asia",
    "Melanesia", "Micronesia", "Polynesia",
    "Less developed regions", "More developed regions",
    "Least developed countries",
    "Australia and New Zealand", "Channel Islands", "Sub-Saharan Africa",
}

NAME_MAP = {
    "Viet Nam": "vietnam",
    "Iran (Islamic Republic of)": "iran",
    "Republic of Korea": "south korea",
    "United States of America": "united states",
    "United Kingdom of Great Britain and Northern Ireland": "united kingdom",
    "Russian Federation": "russia",
    "United Republic of Tanzania": "tanzania",
    "Democratic Republic of the Congo": "congo, dem. rep.",
    "Congo": "congo, rep.",
    "Bolivia (Plurinational State of)": "bolivia",
    "Venezuela (Bolivarian Republic of)": "venezuela",
    "Republic of Moldova": "moldova",
    "Syrian Arab Republic": "syria",
    "Taiwan Province of China": "taiwan",
    "Lao People's Democratic Republic": "laos",
    "Türkiye": "turkey",
    "Eswatini": "swaziland",
    "Cabo Verde": "cape verde",
    "Czechia": "czech republic",
    "North Macedonia": "macedonia",
}


def load_wb(filename):
    """Load a World Bank WDI CSV (Country × Year wide format)."""
    df = pd.read_csv(os.path.join(DATA, filename))
    df["Country"] = df["Country"].str.lower()
    df = df.set_index("Country")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_education(filename="completion_both_long.csv"):
    """Load WCDE education data, excluding regional aggregates."""
    edu = pd.read_csv(os.path.join(PROC, filename))
    edu = edu[~edu["country"].isin(REGIONS)].copy()
    return edu


def get_wb_val(df, wcde_name, year):
    """Look up a World Bank value for a WCDE country name."""
    key = NAME_MAP.get(wcde_name, wcde_name).lower()
    for k in [wcde_name.lower(), key]:
        if k in df.index:
            try:
                v = float(df.loc[k, str(year)])
                return v if not np.isnan(v) else np.nan
            except (KeyError, ValueError):
                pass
    return np.nan


def interpolate_to_annual(edu_df, col_name):
    """Interpolate 5-year WCDE data to annual values, per country."""
    edu_annual = {}
    for c, grp in edu_df.groupby("country"):
        s = grp.set_index("year")[col_name].sort_index()
        full_idx = range(s.index.min(), s.index.max() + 1)
        edu_annual[c] = s.reindex(full_idx).interpolate(method="linear")
    return edu_annual


def find_entry_year(series, threshold):
    """Find the first year where education >= threshold."""
    above = series[series >= threshold]
    return above.index[0] if len(above) > 0 else None


def build_panel(edu_annual, outcome_df, gdp_df, t_years, lag,
                outcome_name="outcome"):
    """
    Build a panel DataFrame with education, log GDP, and outcome at T+lag.
    Returns DataFrame with columns: country, t, edu_t, log_gdp_t, {outcome_name}.
    """
    rows = []
    for c in sorted(edu_annual.keys()):
        s = edu_annual[c]
        for t in t_years:
            if t not in s.index:
                continue
            edu_val = s[t]
            gdp_t = get_wb_val(gdp_df, c, t)
            out_val = get_wb_val(outcome_df, c, t + lag)
            if np.isnan(edu_val) or np.isnan(out_val):
                continue
            rows.append({
                "country": c, "t": t, "edu_t": edu_val,
                "log_gdp_t": np.log(gdp_t) if not np.isnan(gdp_t) and gdp_t > 0 else np.nan,
                outcome_name: out_val,
            })
    return pd.DataFrame(rows)


def precompute_entry_years(edu_annual, thresholds=range(10, 91)):
    """For each threshold, find entry year per country."""
    entry_years = {}
    for threshold in thresholds:
        cohort = {}
        for c, s in edu_annual.items():
            entry = find_entry_year(s, threshold)
            if entry is not None:
                cohort[c] = entry
        entry_years[threshold] = cohort
    return entry_years


def filter_panel(panel, cohort, ceiling, edu_col="edu_t"):
    """Filter panel to entry-cohort with ceiling."""
    mask = panel.apply(
        lambda r: (
            r["country"] in cohort
            and r["t"] >= cohort[r["country"]]
            and r[edu_col] <= ceiling
        ),
        axis=1
    )
    return panel[mask]


def fe_r2(x_col, y_col, data):
    """
    Country fixed-effects R² (within-R²).
    Demean by country, regress demeaned y on demeaned x, no intercept.
    Returns (r2, n_obs, n_countries).
    """
    sub = data.dropna(subset=[x_col, y_col]).copy()
    counts = sub.groupby("country").size()
    sub = sub[sub["country"].isin(counts[counts >= 2].index)]
    n_countries = sub["country"].nunique()
    if n_countries < 3 or len(sub) < 10:
        return np.nan, 0, 0
    xdm = sub[x_col] - sub.groupby("country")[x_col].transform("mean")
    ydm = sub[y_col] - sub.groupby("country")[y_col].transform("mean")
    X = xdm.values.reshape(-1, 1)
    y = ydm.values
    ok = ~np.isnan(X.ravel()) & ~np.isnan(y)
    if ok.sum() < 10:
        return np.nan, 0, 0
    reg = LinearRegression(fit_intercept=False).fit(X[ok], y[ok])
    return reg.score(X[ok], y[ok]), int(ok.sum()), n_countries


def fe_twoway_r2(x_col, y_col, data, time_col="t"):
    """
    Two-way fixed effects R² (country + time).
    Demean by country AND by time period, then regress.
    Returns (r2, n_obs, n_countries).
    """
    sub = data.dropna(subset=[x_col, y_col]).copy()
    counts = sub.groupby("country").size()
    sub = sub[sub["country"].isin(counts[counts >= 2].index)]
    n_countries = sub["country"].nunique()
    if n_countries < 3 or len(sub) < 10:
        return np.nan, 0, 0
    # Demean by country
    xdm = sub[x_col] - sub.groupby("country")[x_col].transform("mean")
    ydm = sub[y_col] - sub.groupby("country")[y_col].transform("mean")
    # Demean by time period
    sub["_xdm"] = xdm
    sub["_ydm"] = ydm
    xdm2 = sub["_xdm"] - sub.groupby(time_col)["_xdm"].transform("mean")
    ydm2 = sub["_ydm"] - sub.groupby(time_col)["_ydm"].transform("mean")
    X = xdm2.values.reshape(-1, 1)
    y = ydm2.values
    ok = ~np.isnan(X.ravel()) & ~np.isnan(y)
    if ok.sum() < 10:
        return np.nan, 0, 0
    reg = LinearRegression(fit_intercept=False).fit(X[ok], y[ok])
    return reg.score(X[ok], y[ok]), int(ok.sum()), n_countries


def fe_residualize_gdp(data, fe_func=fe_r2):
    """
    Residualize GDP against education with country FE.
    Returns (sub_with_gdp_resid, edu_gdp_r2) or None.
    """
    sub = data.dropna(subset=["edu_t", "log_gdp_t"]).copy()
    counts = sub.groupby("country").size()
    sub = sub[sub["country"].isin(counts[counts >= 2].index)]
    if sub["country"].nunique() < 3 or len(sub) < 10:
        return None

    edu_dm = sub["edu_t"] - sub.groupby("country")["edu_t"].transform("mean")
    gdp_dm = sub["log_gdp_t"] - sub.groupby("country")["log_gdp_t"].transform("mean")

    X = edu_dm.values.reshape(-1, 1)
    y = gdp_dm.values
    ok = ~np.isnan(X.ravel()) & ~np.isnan(y)
    if ok.sum() < 10:
        return None

    reg = LinearRegression(fit_intercept=False).fit(X[ok], y[ok])
    sub["gdp_resid"] = gdp_dm.values - reg.predict(X).ravel()
    edu_gdp_r2 = reg.score(X[ok], y[ok])
    return sub, edu_gdp_r2


def fe_residualize_gdp_twoway(data, time_col="t"):
    """
    Residualize GDP against education with two-way FE (country + time).
    Returns (sub_with_gdp_resid, edu_gdp_r2) or None.
    """
    sub = data.dropna(subset=["edu_t", "log_gdp_t"]).copy()
    counts = sub.groupby("country").size()
    sub = sub[sub["country"].isin(counts[counts >= 2].index)]
    if sub["country"].nunique() < 3 or len(sub) < 10:
        return None

    # Demean by country
    edu_dm = sub["edu_t"] - sub.groupby("country")["edu_t"].transform("mean")
    gdp_dm = sub["log_gdp_t"] - sub.groupby("country")["log_gdp_t"].transform("mean")
    # Demean by time
    sub["_edu_dm"] = edu_dm
    sub["_gdp_dm"] = gdp_dm
    edu_dm2 = sub["_edu_dm"] - sub.groupby(time_col)["_edu_dm"].transform("mean")
    gdp_dm2 = sub["_gdp_dm"] - sub.groupby(time_col)["_gdp_dm"].transform("mean")

    X = edu_dm2.values.reshape(-1, 1)
    y = gdp_dm2.values
    ok = ~np.isnan(X.ravel()) & ~np.isnan(y)
    if ok.sum() < 10:
        return None

    reg = LinearRegression(fit_intercept=False).fit(X[ok], y[ok])
    sub["gdp_resid"] = gdp_dm2.values - reg.predict(X).ravel()
    edu_gdp_r2 = reg.score(X[ok], y[ok])
    return sub, edu_gdp_r2


def run_residualized_sweep(panel, entry_years, outcome_col, ceilings,
                           fe_func=fe_r2, resid_func=fe_residualize_gdp,
                           label="outcome", print_every=10):
    """
    Run the full entry-threshold × ceiling sweep with residualization.
    Returns nested dict: results[ceiling][threshold] = {...}.
    """
    all_results = {}

    for ceiling in ceilings:
        print(f"\n  Ceiling = {ceiling}%")
        print(f"  {'Entry':<8} {'Edu R²':>7} {'GDP R²':>7} {'Resid R²':>9} {'Edu→GDP':>8} {'n':>5} {'Ctry':>5}")
        print(f"  {'-' * 60}")

        ceil_results = {}

        for threshold in range(10, 91):
            if threshold > ceiling:
                break

            cohort = entry_years.get(threshold, {})
            if len(cohort) < 3:
                continue

            sub = filter_panel(panel, cohort, ceiling)
            if len(sub) < 10:
                continue

            r2_e, n_e, c_e = fe_func("edu_t", outcome_col, sub)
            r2_g, n_g, c_g = fe_func("log_gdp_t", outcome_col, sub)

            resid_result = resid_func(sub)
            if resid_result is not None:
                sub_resid, edu_gdp_r2 = resid_result
                r2_resid, _, _ = fe_func("gdp_resid", outcome_col, sub_resid)
            else:
                r2_resid, edu_gdp_r2 = np.nan, np.nan

            if threshold % print_every == 0 or threshold == 10:
                def fmt(v): return f"{v:.3f}" if not np.isnan(v) else "n/a"
                print(f"  >= {threshold}%{'':<2} {fmt(r2_e):>7} {fmt(r2_g):>7} "
                      f"{fmt(r2_resid):>9} {fmt(edu_gdp_r2):>8} {n_e:>5} {c_e:>5}")

            ceil_results[str(threshold)] = {
                "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
                "raw_gdp_r2": round(r2_g, 3) if not np.isnan(r2_g) else None,
                "resid_gdp_r2": round(r2_resid, 3) if not np.isnan(r2_resid) else None,
                "edu_gdp_r2": round(edu_gdp_r2, 3) if not np.isnan(edu_gdp_r2) else None,
                "edu_n": n_e, "edu_countries": c_e,
            }

        all_results[str(ceiling)] = ceil_results

    return all_results


def print_summary(results, ceilings, label="outcome"):
    """Print summary table for entry=10% across ceilings."""
    print(f"\n  SUMMARY: entry=10%")
    efmt = f"Edu→{label}"
    gfmt = f"GDP→{label}"
    rfmt = f"Resid→{label}"
    print(f"  {'Ceiling':<10} {efmt:>9} {gfmt:>9} {rfmt:>10} {'Edu→GDP':>8}")
    print(f"  {'-' * 50}")
    for ceiling in ceilings:
        r = results[str(ceiling)].get("10", {})
        if r and r.get("edu_r2") is not None:
            def fmt(v): return f"{v:.3f}" if v is not None else "n/a"
            print(f"  <= {ceiling}%{'':<4} {fmt(r.get('edu_r2')):>9} {fmt(r.get('raw_gdp_r2')):>9} "
                  f"{fmt(r.get('resid_gdp_r2')):>10} {fmt(r.get('edu_gdp_r2')):>8}")
