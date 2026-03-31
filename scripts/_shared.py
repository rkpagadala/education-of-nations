"""
_shared.py
==========
Shared constants, country mapping, and data loading utilities.

Panel construction and regression utilities live in residualization/_shared.py.
"""

import json
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROC = os.path.join(REPO_ROOT, "wcde", "data", "processed")
DATA = os.path.join(REPO_ROOT, "data")
CHECKIN = os.path.join(REPO_ROOT, "checkin")

# ── Development thresholds (1960 USA benchmarks) ───────────────────
TFR_THRESHOLD = 3.65          # 1960 USA fertility rate
LE_THRESHOLD = 69.8           # 1960 USA life expectancy (years)

# ── Oil states (excluded from floor tests) ─────────────────────────
OIL_STATES = [
    "Qatar", "United Arab Emirates", "Kuwait",
    "Saudi Arabia", "Oman", "Bahrain",
]

WCDE_AGGREGATES = {
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
    "South-Central Asia",
}

# Everything to exclude from country-level analysis (185 countries remain)
REGIONS = WCDE_AGGREGATES

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
    "Libyan Arab Jamahiriya": "libya",
    "Hong Kong Special Administrative Region of China": "hong kong",
    "Macao Special Administrative Region of China": "macao",
    "Côte d'Ivoire": "cote d'ivoire",
    "Democratic People's Republic of Korea": "north korea",
    "Myanmar": "myanmar",
    "Timor-Leste": "timor-leste",
}

# ── Case study countries ───────────────────────────────────────────
# WCDE names → display names for the six case-study countries
CASE_STUDY_NAMES = {
    "Taiwan Province of China": "Taiwan",
    "Republic of Korea":        "South Korea",
    "Cuba":                     "Cuba",
    "Bangladesh":               "Bangladesh",
    "Sri Lanka":                "Sri Lanka",
    "China":                    "China",
}

# Combined development crossing (TFR < 3.65 AND LE > 69.8) — from Table 4
# Taiwan: WCDE 5-year data shows both crossed by 1970 (TFR=3.47, LE=70.0)
CASE_CROSSING_COMBINED = {
    "Taiwan Province of China": 1970,
    "Republic of Korea":        1987,
    "Cuba":                     1974,
    "Bangladesh":               2014,
    "Sri Lanka":                1993,
    "China":                    1994,
}

# TFR-only crossing (TFR < 3.65) — from Table 4
CASE_CROSSING_TFR = {
    "Taiwan Province of China": 1970,
    "Republic of Korea":        1975,
    "Cuba":                     1972,
    "Bangladesh":               1995,
    "Sri Lanka":                1981,
    "China":                    1975,
}

# LE-only crossing (LE > 69.8) — same as combined for these countries
CASE_CROSSING_LE = dict(CASE_CROSSING_COMBINED)

# ── World Bank region names (for excluding from WDI CSVs) ─────────
WB_REGIONS = {
    "africa eastern and southern", "africa western and central", "arab world",
    "caribbean small states", "central europe and the baltics",
    "early-demographic dividend", "east asia & pacific",
    "east asia & pacific (excluding high income)",
    "east asia & pacific (ida & ibrd countries)", "euro area",
    "europe & central asia", "europe & central asia (excluding high income)",
    "europe & central asia (ida & ibrd countries)", "european union",
    "fragile and conflict affected situations",
    "heavily indebted poor countries (hipc)", "high income",
    "ida & ibrd total", "ida blend", "ida only", "ida total", "ibrd only",
    "late-demographic dividend", "latin america & caribbean",
    "latin america & caribbean (excluding high income)",
    "latin america & the caribbean (ida & ibrd countries)",
    "least developed countries: un classification", "low & middle income",
    "low income", "lower middle income", "middle east & north africa",
    "middle east & north africa (excluding high income)",
    "middle east & north africa (ida & ibrd countries)",
    "middle east, north africa, afghanistan & pakistan",
    "middle east, north africa, afghanistan & pakistan (ida & ibrd)",
    "middle east, north africa, afghanistan & pakistan (excluding high income)",
    "middle income", "north america", "not classified", "oecd members",
    "other small states", "pacific island small states",
    "post-demographic dividend", "pre-demographic dividend", "small states",
    "south asia", "south asia (ida & ibrd)", "sub-saharan africa",
    "sub-saharan africa (excluding high income)",
    "sub-saharan africa (ida & ibrd countries)", "upper middle income", "world",
}

# WB country names → WCDE country names (for population lookups)
WB_TO_WCDE = {
    "bahamas, the": "bahamas",
    "cabo verde": "cape verde",
    "congo, dem. rep.": "democratic republic of the congo",
    "congo, rep.": "congo",
    "czechia": "czech republic",
    "egypt, arab rep.": "egypt",
    "gambia, the": "gambia",
    "hong kong sar, china": "hong kong special administrative region of china",
    "iran, islamic rep.": "iran (islamic republic of)",
    "korea, dem. people's rep.": "democratic people's republic of korea",
    "korea, rep.": "republic of korea",
    "kyrgyz republic": "kyrgyzstan",
    "lao pdr": "lao people's democratic republic",
    "libya": "libyan arab jamahiriya",
    "macao sar, china": "macao special administrative region of china",
    "micronesia, fed. sts.": "micronesia (federated states of)",
    "moldova": "republic of moldova",
    "north macedonia": "the former yugoslav republic of macedonia",
    "puerto rico (us)": "puerto rico",
    "slovak republic": "slovakia",
    "somalia, fed. rep.": "somalia",
    "st. kitts and nevis": "saint kitts and nevis",
    "st. lucia": "saint lucia",
    "st. vincent and the grenadines": "saint vincent and the grenadines",
    "turkiye": "turkey",
    "united kingdom": "united kingdom of great britain and northern ireland",
    "united states": "united states of america",
    "venezuela, rb": "venezuela (bolivarian republic of)",
    "virgin islands (u.s.)": "united states virgin islands",
    "west bank and gaza": "state of palestine",
    "yemen, rep.": "yemen",
    "eswatini": "swaziland",
    "cote d'ivoire": "côte d'ivoire",
    "timor-leste": "timor-leste",
}


def load_wb(filename):
    """Load a World Bank WDI CSV (Country × Year wide format)."""
    df = pd.read_csv(os.path.join(DATA, filename))
    df["Country"] = df["Country"].str.lower()
    df = df.set_index("Country")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_wide_indicator(filename):
    """Load wide-format WB CSV, excluding WB region rows.

    Returns DataFrame: index=country (lowercase), columns=year strings.
    """
    df = pd.read_csv(os.path.join(DATA, filename))
    df["country"] = df["Country"].str.lower()
    df = df[~df["country"].isin(WB_REGIONS)]
    return df.set_index("country").drop(columns=["Country"])


def load_population_by_year():
    """Load WCDE population totals by country and year.

    Returns DataFrame: index=WB lowercase country name, columns=WCDE years,
    values=population in thousands.
    """
    pop_path = os.path.join(REPO_ROOT, "wcde", "data", "raw", "pop_both.csv")
    pop = pd.read_csv(pop_path)
    pop = pop[pop["scenario"] == 2]
    pop["country"] = pop["name"].str.lower()
    wcde_regions_lower = {r.lower() for r in REGIONS}
    pop = pop[~pop["country"].isin(wcde_regions_lower)]
    totals = pop.groupby(["country", "year"])["pop"].sum().unstack("year")
    wcde_to_wb = {v: k for k, v in WB_TO_WCDE.items()}
    totals.index = [wcde_to_wb.get(n, n) for n in totals.index]
    return totals


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


# ── WCDE → WDI country name mapping ─────────────────────────────
# Maps WCDE v3 country names to World Bank WDI country names.
# Used by scripts that join education data (WCDE) with WDI indicators.
WCDE_TO_WDI = {
    "Bahamas": "Bahamas, The",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Cape Verde": "Cabo Verde",
    "Congo": "Congo, Rep.",
    "Curaçao": "Curacao",
    "Czech Republic": "Czechia",
    "Democratic People's Republic of Korea": "Korea, Dem. People's Rep.",
    "Democratic Republic of the Congo": "Congo, Dem. Rep.",
    "Egypt": "Egypt, Arab Rep.",
    "Gambia": "Gambia, The",
    "Hong Kong Special Administrative Region of China": "Hong Kong SAR, China",
    "Iran (Islamic Republic of)": "Iran, Islamic Rep.",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Lao People's Democratic Republic": "Lao PDR",
    "Libyan Arab Jamahiriya": "Libya",
    "Macao Special Administrative Region of China": "Macao SAR, China",
    "Micronesia (Federated States of)": "Micronesia, Fed. Sts.",
    "Occupied Palestinian Territory": "West Bank and Gaza",
    "Puerto Rico": "Puerto Rico (US)",
    "Republic of Korea": "Korea, Rep.",
    "Republic of Moldova": "Moldova",
    "Saint Lucia": "St. Lucia",
    "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
    "Slovakia": "Slovak Republic",
    "Swaziland": "Eswatini",
    "The former Yugoslav Republic of Macedonia": "North Macedonia",
    "Turkey": "Turkiye",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "United Republic of Tanzania": "Tanzania",
    "United States of America": "United States",
    "United States Virgin Islands": "Virgin Islands (U.S.)",
    "Venezuela (Bolivarian Republic of)": "Venezuela, RB",
    "Yemen": "Yemen, Rep.",
}


def wcde_to_wdi(name):
    """Map a WCDE country name to its WDI equivalent; identity fallback."""
    return WCDE_TO_WDI.get(name, name)


def interpolate_wide_to_annual(df):
    """Interpolate a wide-format DataFrame from 5-year to annual resolution.

    Input:  index=country, columns=integer years (e.g. 1950, 1955, ...).
    Output: same shape but with every intervening year filled by linear
            interpolation.
    """
    dft = df.T
    dft.index = dft.index.astype(int)
    full_idx = range(dft.index.min(), dft.index.max() + 1)
    dft = dft.reindex(full_idx).interpolate(method="linear")
    return dft.T  # back to country × year


def completion_at_year(df_wide, country, year):
    """Interpolate a single value from a wide-format education CSV.

    Works with 5-year WCDE data: finds the bracketing years and linearly
    interpolates.  Returns np.nan if the country or year is out of range.
    """
    if country not in df_wide.index:
        return np.nan
    row = df_wide.loc[country].dropna()
    years = np.array([int(c) for c in row.index], dtype=float)
    vals = row.values.astype(float)
    if len(years) == 0 or year < years.min() or year > years.max():
        return np.nan
    return float(np.interp(year, years, vals))


def write_checkin(filename, data, script_path=None):
    """Write a checkin JSON file. Adds 'produced' date and 'script' path."""
    if script_path:
        data["script"] = script_path
    data["produced"] = str(pd.Timestamp.now().date())
    os.makedirs(CHECKIN, exist_ok=True)
    path = os.path.join(CHECKIN, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Checkin written to {path}")


# ── Formatting helpers ────────────────────────────────────────────

def fmt_r2(v):
    """Format an R² or float value for display: '0.456' or 'n/a'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.3f}"


# ── Standard data loading ─────────────────────────────────────────

def load_standard_outcomes():
    """Load GDP and the three standard WB outcome datasets.

    Returns dict with keys: gdp, le, tfr, u5mr (all wide-format DataFrames).
    """
    return {
        "gdp": load_wb("gdppercapita_us_inflation_adjusted.csv"),
        "le": load_wb("life_expectancy_years.csv"),
        "tfr": load_wb("children_per_woman_total_fertility.csv"),
        "u5mr": load_wb("child_mortality_u5.csv"),
    }


# ── FE regression with clustered standard errors ─────────────────

def fe_regression(df, x_cols, y_col, cluster_col="country"):
    """Country fixed effects via demeaning, with cluster-robust SEs.

    Parameters
    ----------
    df : DataFrame with columns for x_cols, y_col, and cluster_col.
    x_cols : list of predictor column names.
    y_col : outcome column name.
    cluster_col : column to cluster on (default 'country').

    Returns (statsmodels OLS result, n_obs, n_countries).
    """
    d = df.dropna(subset=x_cols + [y_col]).copy()
    for col in x_cols + [y_col]:
        d[col + "_dm"] = d.groupby(cluster_col)[col].transform(
            lambda x: x - x.mean()
        )
    X = d[[c + "_dm" for c in x_cols]]
    y = d[y_col + "_dm"]
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": d[cluster_col]},
    )
    return model, len(d), d[cluster_col].nunique()

