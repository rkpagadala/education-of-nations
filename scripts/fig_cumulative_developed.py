"""
Cumulative share of world population living in 'developed' countries over time.

Developed = TFR < 3.65 AND LE > 69.8 (1960 USA benchmarks).

For each year from 1960 to 2022, identifies which countries have crossed BOTH
thresholds and sums their population as a share of the world total.

Population source: WCDE v3 scenario 2 (5-year grid). Each year is matched to
the nearest WCDE population year. TFR/LE sources: World Bank WDI wide-format CSVs.

Output: scripts/fig_cumulative_developed.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Paths ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
POP_PATH = os.path.join(REPO_ROOT, "wcde", "data", "raw", "pop_both.csv")
OUT_PATH = os.path.join(SCRIPT_DIR, "fig_cumulative_developed.png")

# ── Constants ────────────────────────────────────────────────────
TFR_THRESHOLD = 3.65          # 1960 USA fertility rate
LE_THRESHOLD = 69.8           # 1960 USA life expectancy (years)
START_YEAR = 1960
END_YEAR = 2022

# World Bank aggregate / region names to exclude.
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

WCDE_REGIONS = {
    "africa", "asia", "europe", "world", "caribbean", "central america",
    "central asia", "eastern africa", "eastern asia", "eastern europe",
    "latin america and the caribbean", "middle africa", "northern africa",
    "northern america", "northern europe", "south america",
    "south-eastern asia", "southern africa", "southern asia",
    "southern europe", "western africa", "western asia", "western europe",
    "less developed regions", "more developed regions",
    "less developed regions, excl. china", "least developed countries",
    "oceania", "melanesia", "micronesia", "polynesia",
}

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


def load_wide_indicator(filename: str) -> pd.DataFrame:
    """Load wide-format WB CSV. Returns DataFrame: index=country (lowercase), columns=year strings."""
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    df["country"] = df["Country"].str.lower()
    df = df[~df["country"].isin(WB_REGIONS)]
    df = df.set_index("country").drop(columns=["Country"])
    return df


def load_population_by_year() -> pd.DataFrame:
    """Load WCDE population totals by country and year.

    Returns DataFrame: index=WB lowercase country name, columns=WCDE years,
    values=population in thousands.
    """
    pop = pd.read_csv(POP_PATH)
    pop = pop[pop["scenario"] == 2]
    pop["country"] = pop["name"].str.lower()
    pop = pop[~pop["country"].isin(WCDE_REGIONS)]
    totals = pop.groupby(["country", "year"])["pop"].sum().unstack("year")

    # Remap WCDE names → WB names
    wcde_to_wb = {v: k for k, v in WB_TO_WCDE.items()}
    totals.index = [wcde_to_wb.get(n, n) for n in totals.index]
    return totals


def main():
    tfr_all = load_wide_indicator("children_per_woman_total_fertility.csv")
    le_all = load_wide_indicator("life_expectancy_years.csv")
    pop_all = load_population_by_year()

    # Available WCDE population years (5-year grid)
    pop_years = sorted(pop_all.columns)

    years = list(range(START_YEAR, END_YEAR + 1))
    pct_developed = []
    n_countries = []

    for yr in years:
        yr_str = str(yr)
        if yr_str not in tfr_all.columns or yr_str not in le_all.columns:
            pct_developed.append(None)
            n_countries.append(0)
            continue

        tfr = tfr_all[yr_str].dropna()
        le = le_all[yr_str].dropna()

        crossed_tfr = set(tfr[tfr < TFR_THRESHOLD].index)
        crossed_le = set(le[le > LE_THRESHOLD].index)
        crossed = crossed_tfr & crossed_le

        # Philippines crossed in 2020 (LE 70.1, TFR 2.08); COVID pulled LE
        # below threshold in 2021-22. Count as crossed from 2020 onward.
        if yr >= 2020 and "philippines" in crossed_tfr:
            crossed = crossed | {"philippines"}

        # Find nearest WCDE population year
        nearest_pop_yr = min(pop_years, key=lambda y: abs(y - yr))
        pop = pop_all[nearest_pop_yr].dropna()

        pop_crossed = sum(pop.get(c, 0) for c in crossed)
        pop_world = pop.sum()

        pct = 100.0 * pop_crossed / pop_world if pop_world > 0 else 0
        pct_developed.append(pct)
        n_countries.append(len(crossed))

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_pct = "#1f4e79"
    ax1.fill_between(years, pct_developed, alpha=0.15, color=color_pct)
    ax1.plot(years, pct_developed, color=color_pct, linewidth=2.5,
             label="% of world population")
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Share of world population (%)", fontsize=12, color=color_pct)
    ax1.tick_params(axis="y", labelcolor=color_pct)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_ylim(0, 100)
    ax1.set_xlim(START_YEAR, END_YEAR)

    # Second axis: number of countries
    color_n = "#c0392b"
    ax2 = ax1.twinx()
    ax2.plot(years, n_countries, color=color_n, linewidth=1.5,
             linestyle="--", alpha=0.7, label="Number of countries")
    ax2.set_ylabel("Number of countries crossed", fontsize=12, color=color_n)
    ax2.tick_params(axis="y", labelcolor=color_n)
    ax2.set_ylim(0, 180)

    # Annotations for key crossings
    annotations = [
        (1972, "Taiwan"),
        (1987, "S. Korea"),
        (1994, "China"),
        (2014, "Bangladesh"),
        (2022, f"{n_countries[-1]} countries"),
    ]
    for yr, label in annotations:
        idx = yr - START_YEAR
        if 0 <= idx < len(pct_developed) and pct_developed[idx] is not None:
            ax1.annotate(
                label, xy=(yr, pct_developed[idx]),
                xytext=(0, 18), textcoords="offset points",
                fontsize=8, ha="center", color="#333333",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
            )

    ax1.set_title(
        "Cumulative share of world population in 'developed' countries\n"
        f"(TFR < {TFR_THRESHOLD}, LE > {LE_THRESHOLD} — 1960 USA benchmarks)",
        fontsize=13, fontweight="bold", pad=15,
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")

    # Print key milestones
    print("\nKey milestones:")
    for target in [10, 25, 50, 75]:
        for i, pct in enumerate(pct_developed):
            if pct is not None and pct >= target:
                print(f"  {target}% of world population: {years[i]}  "
                      f"({n_countries[i]} countries, {pct:.1f}%)")
                break


if __name__ == "__main__":
    main()
