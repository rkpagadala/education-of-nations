# Data Sources

All datasets used in this paper, with provenance and rebuild instructions.

---

## 1. WCDE v3 (Wittgenstein Centre Data Explorer)

**Source:** Wittgenstein Centre for Demography and Global Human Capital (IIASA/VID/WU)

**Citation:** Lutz, W., Lijadi, A.A., Gailey, N. et al. (2021). Wittgenstein Centre Data Explorer Version 3.0. <https://dataexplorer.wiiw.ac.at/vis/3>

**What it contains:**
- Educational attainment proportions by country, year, age group, sex, and education level (no education, incomplete primary, primary, lower secondary, upper secondary, post-secondary)
- Life expectancy at birth (e0) by country and year
- Total fertility rate (TFR) by country and year
- Population by country, year, age, sex, and education level

**Coverage:** 185+ countries, 1950–2100 (historical reconstruction + SSP projections). Historical years (1950–2015) are empirically reconstructed; projection years (2020+) use SSP2 (medium scenario).

**Long-run reconstruction:** The `wcde/scripts/02b_cohort_reconstruction.py` script reconstructs education data back to ~1875 by tracing 20–24 age-group attainment backward (a person aged 60–64 in 1950 was 20–24 in ~1910). This yields 28 countries with data extending to 1900.

**Raw files downloaded by `wcde/scripts/01_download.R`:**
- `wcde/data/raw/prop_both.csv` — education attainment %, both sexes
- `wcde/data/raw/prop_female.csv` — education attainment %, female
- `wcde/data/raw/tfr.csv` — total fertility rate
- `wcde/data/raw/e0_both.csv` — life expectancy at birth, both sexes
- `wcde/data/raw/e0_female.csv` — life expectancy at birth, female
- `wcde/data/raw/pop_both.csv` — population by education level

**Processed files (built by `wcde/scripts/02_process.py`):**
- `wcde/data/processed/completion_both_long.csv` — lower secondary, upper secondary, and college completion rates (20–24 age group), both sexes, long format
- `wcde/data/processed/completion_female_long.csv` — same, female only
- `wcde/data/processed/completion_male_long.csv` — same, male only
- `wcde/data/processed/e0.csv` — life expectancy, wide format
- `wcde/data/processed/lower_sec_both.csv` — lower secondary completion, wide format (country x year)
- Various cohort reconstruction files (`cohort_*.csv`)

**Key variable used in paper:** Lower secondary completion rate for the 20–24 age cohort (reflects completed education, not enrolment).

---

## 2. World Bank World Development Indicators (WDI)

**Source:** World Bank Open Data, <https://databank.worldbank.org/source/world-development-indicators>

**Files in `data/` directory:**

### GDP per capita
- **File:** `data/gdppercapita_us_inflation_adjusted.csv`
- **WDI indicator:** `NY.GDP.PCAP.KD` (GDP per capita, constant 2015 USD)
- **URL:** <https://data.worldbank.org/indicator/NY.GDP.PCAP.KD>
- **Format:** Country (rows) x Year (columns), wide format
- **Coverage:** ~200 countries, 1960–present

### Life expectancy at birth
- **File:** `data/life_expectancy_years.csv`
- **WDI indicator:** `SP.DYN.LE00.IN` (Life expectancy at birth, total years)
- **URL:** <https://data.worldbank.org/indicator/SP.DYN.LE00.IN>
- **Format:** Country (rows) x Year (columns), wide format
- **Coverage:** ~200 countries, 1960–present

### Total fertility rate
- **File:** `data/children_per_woman_total_fertility.csv`
- **WDI indicator:** `SP.DYN.TFRT.IN` (Fertility rate, total births per woman)
- **URL:** <https://data.worldbank.org/indicator/SP.DYN.TFRT.IN>
- **Format:** Country (rows) x Year (columns), wide format
- **Coverage:** ~200 countries, 1960–present

### CO2 emissions
- **File:** `data/co2_emissions_tonnes_per_person.csv`
- **WDI indicator:** `EN.ATM.CO2E.PC` (CO2 emissions, metric tons per capita)
- **URL:** <https://data.worldbank.org/indicator/EN.ATM.CO2E.PC>
- **Note:** Not used in the main paper analysis.

---

## 3. How to Rebuild from Scratch

### Step 1: Download WCDE raw data

```bash
# Requires R with the 'wcde' package installed
Rscript wcde/scripts/01_download.R
```

This downloads raw WCDE v3 data into `wcde/data/raw/`.

### Step 2: Process WCDE data

```bash
python wcde/scripts/02_process.py
```

This converts raw WCDE proportions into completion rates and wide-format files in `wcde/data/processed/`.

### Step 3: Reconstruct long-run cohort data (back to ~1875)

```bash
python wcde/scripts/02b_cohort_reconstruction.py
```

This traces 20–24 age-group attainment backward through older cohorts to extend coverage to ~1875 for 28 countries with sufficient age-group depth.

### Step 4: Rebuild World Bank datasets

```bash
python data/rebuild_datasets.py --download-wdi
```

This re-downloads World Bank WDI bulk CSVs and produces the cleaned wide-format files in `data/rebuilt/`.

Without the `--download-wdi` flag, it rebuilds only WCDE-derived files from existing processed data.

### Step 5: Run analysis scripts

```bash
# Main panel regressions
python scripts/table_1_main.py
python scripts/table_a1_two_way_fe.py
python scripts/education_vs_gdp_by_cutoff.py

# Verification scripts (produce checkin/*.json files)
python scripts/verify_table4_crossings.py
python scripts/verify_country_education.py
python scripts/verify_country_le_tfr.py
python scripts/verify_country_gdp.py
# etc.
```

---

## 4. Notes

- **Country names:** Lowercase throughout in the datasets directory; mixed case in WCDE processed files.
- **Taiwan:** Not present in World Bank WDI (political exclusion). All Taiwan data comes from WCDE v3 only.
- **GDP units:** Constant 2015 USD per capita (World Bank) or constant 2017 USD per capita (noted in paper as US inflation-adjusted).
- **Panel construction:** The main panel uses WCDE 5-year intervals (1975–2015) matched with World Bank annual data at the same 5-year points. 187 countries, 1,683 country-years for education; 148 countries, 1,229 country-years when GDP is included (fewer countries have GDP data).
- **Kerala:** Not a sovereign country in WDI or WCDE. Kerala estimates come from India Sample Registration System and census records.
