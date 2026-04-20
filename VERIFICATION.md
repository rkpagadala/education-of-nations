# Verification System Documentation

**Education of Nations — Full Traceability Pipeline**

Every empirical number in the paper is traceable to a script that produces it from source data. This document describes the verification system: how to run it, what it checks, and how the pieces fit together.

---

## Quick Start for Reviewers

**Prerequisites:** Python 3.9+ and `make`. Nothing else.

```bash
git clone https://github.com/rkpagadala/education-of-nations.git
cd education-of-nations
make setup          # Create venv + install dependencies
make verify         # ~2 seconds: checks all 351 paper claims against source data
make scripts        # Rebuild all checkin JSONs from source data
```

All source data (World Bank CSVs and WCDE processed CSVs) is included in the repository.

A successful run prints:

```
========================================================================
SUMMARY: 351/351 PASS, 0 FAIL, 0 MISSING, 6 REF (manual check)
COVERAGE: 0 lines with unregistered numbers
========================================================================
```

If any number fails, the output shows exactly which claim failed, what the expected value was, what the actual value is, and where in the paper it appears. The detailed report is written to `checkin/VERIFICATION_REPORT.md`.

**What to send back if verification fails:** the console output and the generated `checkin/VERIFICATION_REPORT.md`.

---

## What "Pass" Means — Tolerance Windows

The paper rounds numbers for readability. A "pass" means the script-computed value matches the paper's stated value within a stated tolerance. Tolerances vary by data type:

| Data Type | Tolerance | Reason |
|-----------|-----------|--------|
| Regression R² | ±0.001–0.005 | Full-precision regression output |
| Regression β | ±0.01–0.1 | Wider for baseline-group estimates with fewer observations |
| Education (pp) | ±0.5–2.0 | WCDE reports to 1 decimal; paper rounds further |
| GDP (USD) | ±15–20% | Paper uses approximate values ("~$1,000") |
| Life expectancy (years) | ±0.5–1.5 | Paper rounds to 1 decimal |
| TFR (children/woman) | ±0.1–0.2 | Paper rounds to 1 decimal |
| Counts (N, countries) | ±0 | Exact match required |

Each registered number carries its own tolerance. The master verifier (`verify_nations.py`) reports the tolerance alongside the expected and actual values, so you can judge whether a near-miss is rounding or a real discrepancy.

---

## Verification Independence

The system is designed so that no layer checks itself:

1. **Layer 1 — Source data** (WCDE CSVs, World Bank CSVs). Raw inputs, never modified by scripts.
2. **Layer 2 — Analysis scripts** (40 scripts). Each reads source data independently, runs its analysis, and writes results to a JSON file in `checkin/`. Scripts do not read the paper.
3. **Layer 3 — Master verifier** (`verify_nations.py`). Reads the paper `.tex` file and the checkin JSONs. Compares every registered number in the paper against the independently computed value. Also scans the `.tex` for any numbers that are *not* registered (coverage audit).

The key independence property: Layer 2 scripts produce values from data without knowing what the paper claims. Layer 3 compares those values against the paper. If a script were wrong, the verifier would catch the mismatch; if the paper were wrong, the verifier would catch that too.

The coverage scan (Phase 3) provides a second guarantee: it extracts every number from the paper and flags any that lack a registered source. This prevents numbers from being silently hand-typed.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Sources and Provenance](#2-data-sources-and-provenance)
3. [Claim Trace Table](#3-claim-trace-table)
4. [The Master Verifier (verify_nations.py)](#4-the-master-verifier)
5. [Verification Scripts](#5-verification-scripts)
6. [Analysis Scripts — Main Tables](#6-analysis-scripts--main-tables)
7. [Analysis Scripts — Education vs GDP](#7-analysis-scripts--education-vs-gdp)
8. [Analysis Scripts — Robustness](#8-analysis-scripts--robustness)
9. [Analysis Scripts — Case Studies](#9-analysis-scripts--case-studies)
10. [WCDE Pipeline Scripts](#10-wcde-pipeline-scripts)
11. [Key Methodological Choices](#11-key-methodological-choices)
12. [Data Flow Diagram](#12-data-flow-diagram)

---

## 1. Architecture Overview

The system has three layers:

```
Layer 1: Source data (WCDE CSVs, World Bank WDI CSVs)
    |
Layer 2: Analysis scripts (40 scripts → 46 checkin JSONs)
    |
Layer 3: Master verifier (verify_nations.py → reads paper .tex, checks every number)
```

**Layer 1** — Raw data lives in `data/` (World Bank) and `wcde/data/processed/` (WCDE education). These are never modified by scripts.

**Layer 2** — Each script reads source data, runs its analysis, and writes a JSON to `checkin/`. The scripts are of two kinds:
- **Verify scripts** (prefix `verify_`): Check specific factual claims (country education levels, GDP values, crossing dates).
- **Analysis scripts**: Run regressions, sweeps, or robustness tests and store all computed coefficients.

**Layer 3** — `verify_nations.py` maintains a registry of 351 numbers that appear in the paper. For each one, it looks up the actual value from a checkin JSON (or WCDE/WDI CSV directly), compares it to the expected value within a tolerance, and reports PASS/FAIL. It also scans the `.tex` file for any numbers that are *not* registered.

Two Makefiles orchestrate everything:
- **Top-level Makefile**: `make verify` runs the fast check; `make scripts` rebuilds all JSONs from source data.
- **scripts/Makefile**: Each JSON has a make rule with dependencies on its script and data files, so only changed scripts are re-run.

---

## 2. Data Sources and Provenance

**All data needed to verify the paper is included in the repository.** No downloads are required.

### World Bank WDI (`data/`)

| File | Indicator | WDI Code | Format |
|------|-----------|----------|--------|
| `gdppercapita_us_inflation_adjusted.csv` | GDP per capita | NY.GDP.PCAP.KD | Constant 2017 USD |
| `life_expectancy_years.csv` | Life expectancy at birth | SP.DYN.LE00.IN | Years |
| `children_per_woman_total_fertility.csv` | Total fertility rate | SP.DYN.TFRT.IN | Children per woman |
| `child_mortality_0_5_year_olds_dying_per_1000_born.csv` | Under-5 mortality | SH.DYN.MORT | Deaths per 1,000 |

Format: Country × Year wide CSV. Country names are mixed case in the raw files; scripts lowercase them for matching.

These CSVs are committed to the repository as downloaded from the World Bank WDI data portal. They are static snapshots — the World Bank occasionally revises historical data, so values may differ slightly from current API results.

### Additional Data

| File | Content | Source |
|------|---------|--------|
| `data/p5v2018.xls` | Polity5 regime scores | Center for Systemic Peace |
| `data/kerala_srs.csv` | Kerala Sample Registration System | Indian government SRS reports |
| `data/hlo_raw.csv` | Harmonized Learning Outcomes | World Bank HLO database |

### WCDE v3 (`wcde/`)

Education data comes from the Wittgenstein Centre Data Explorer (WCDE) version 3.

**Processed data** (`wcde/data/processed/`) — 27 CSVs, all committed to the repository:

| File | Content |
|------|---------|
| `lower_sec_both.csv` | Lower secondary completion, both sexes, age 20-24 |
| `lower_sec_female.csv` | Lower secondary completion, female, age 20-24 |
| `completion_both_long.csv` | Same as lower_sec_both, long format (country, year, pct) |
| `cohort_lower_sec_both.csv` | Cohort-reconstructed completion (extends back to 1900) |
| `upper_sec_both.csv` | Upper secondary completion, both sexes |
| `college_both.csv` | Tertiary completion, both sexes |
| `primary_both.csv` | Primary completion, both sexes |

5-year intervals (1950, 1955, ..., 2015). 185 countries after excluding 30+ regional aggregates.

### Tested Dependency Versions

The Makefile creates a virtual environment and installs from `requirements.txt` automatically. The verification was developed and tested with:

| Package | Version |
|---------|---------|
| Python | 3.11.2 |
| pandas | 3.0.2 |
| numpy | 2.4.4 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| scipy | 1.17.1 |
| statsmodels | 0.14.6 |
| xlrd | 2.0.2 |

Regression output (particularly standard errors and p-values) can vary across statsmodels versions. If you see small discrepancies in p-values, check your statsmodels version.

---

## 3. Claim Trace Table

Every number in the paper can be traced from the text to a script and source data. The master verifier (`verify_nations.py`) maintains 351 registered claims. Below is a representative sample showing how to trace key results.

### Core Regressions (Table 1)

| Paper Claim | Script | JSON → Key | Tolerance |
|-------------|--------|------------|-----------|
| N = 1,665 | `tables/table_1_main.py` | `table_1_main.json` → `numbers.panel_obs` | ±0 |
| 185 countries | `tables/table_1_main.py` | `table_1_main.json` → `numbers.panel_countries` | ±0 |
| GDP alone β = 15.4 | `tables/table_1_main.py` | `table_1_main.json` → `numbers.PI-alone-beta` | ±0.5 |
| GDP alone R² = 0.293 | `tables/table_1_main.py` | `table_1_main.json` → `numbers.PI-alone-R2` | ±0.001 |
| Education alone R² = 0.553 | `tables/table_1_main.py` | `table_1_main.json` → `numbers.PI-edu-alone` | ±0.001 |
| GDP β drops 72% conditional on education | derived | `(1 - PI-cond-beta / PI-alone-beta) × 100` | ±5.0 |

### Two-Way FE (Table A1)

| Paper Claim | Script | JSON → Key | Tolerance |
|-------------|--------|------------|-----------|
| β = 0.083 | `robustness/twfe_child_edu.py` | `twfe_child_edu.json` → `numbers.ta1_m1_edu_beta` | ±0.001 |
| R² = 0.009 | `robustness/twfe_child_edu.py` | `twfe_child_edu.json` → `numbers.ta1_m1_r2_within` | ±0.001 |

### Forward Prediction (Table 2)

| Paper Claim | Script | JSON → Key | Tolerance |
|-------------|--------|------------|-----------|
| Education → GDP R² = 0.354 | `wcde/education_outcomes.py` | `education_outcomes.json` → `numbers.T2-GDP-R2` | ±0.001 |
| Education → LE R² = 0.382 | `wcde/education_outcomes.py` | `education_outcomes.json` → `numbers.T2-LE-R2` | ±0.001 |
| Education → TFR β = −0.032 | `wcde/education_outcomes.py` | `education_outcomes.json` → `numbers.T2-TFR-beta` | ±0.001 |
| Panel B: N = 828 | `wcde/education_outcomes.py` | `education_outcomes.json` → `numbers.T2-PB-n` | ±0 |

### Residualized GDP (Table 2b) — "GDP has no independent effect"

| Paper Claim | Script | JSON → Key | Tolerance |
|-------------|--------|------------|-----------|
| LE residualized R² = 0.003 | `tables/regression_tables.py` | `regression_tables.json` → `results.LE.90.GDP (residualized).r2` | ±0.005 |
| LE residualized p = 0.56 | `tables/regression_tables.py` | `regression_tables.json` → `results.LE.90.GDP (residualized).pval` | ±0.05 |
| TFR residualized R² = 0.000 | `tables/regression_tables.py` | `regression_tables.json` → `results.TFR.90.GDP (residualized).r2` | ±0.005 |
| U5MR residualized R² = 0.023 | `tables/regression_tables.py` | `regression_tables.json` → `results.U5MR.90.GDP (residualized).r2` | ±0.005 |

### Education vs GDP Horse Race

| Paper Claim | Script | JSON → Key | Tolerance |
|-------------|--------|------------|-----------|
| At 30% entry: education R² = 0.699 | `residualization/education_vs_gdp.py` | `education_vs_gdp_by_cutoff.json` → `numbers.cutoff_30_edu_r2` | ±0.001 |
| At 30% entry: GDP R² = 0.214 | `residualization/education_vs_gdp.py` | `education_vs_gdp_by_cutoff.json` → `numbers.cutoff_30_gdp_r2` | ±0.001 |
| Education 3.3× more predictive | `residualization/education_vs_gdp.py` | `education_vs_gdp_by_cutoff.json` → `numbers.cutoff_30_ratio` | ±0.001 |

### Figure 1 — Beta vs Baseline

| Paper Claim | Script | JSON → Key | Tolerance |
|-------------|--------|------------|-----------|
| USA β = 1.9 at high baseline | `figures/beta_vs_baseline.py` | `beta_vs_baseline.json` → `numbers.Fig1-USA-beta-high` | ±0.1 |
| Korea β = 6.5 at low baseline | `figures/beta_vs_baseline.py` | `beta_vs_baseline.json` → `numbers.Fig1-Korea-beta-high` | ±0.1 |
| Taiwan β = 5.1 | `figures/beta_vs_baseline.py` | `beta_vs_baseline.json` → `numbers.Fig1-Taiwan-beta` | ±0.1 |

### Figure A1 — Lag Decay

| Paper Claim | Script | JSON → Key | Tolerance |
|-------------|--------|------------|-----------|
| Lag 0: R² = 0.562 | `figures/le_r2_by_lag.py` | `le_r2_by_lag.json` → `numbers.edu_r2_lag0` | ±0.001 |
| Lag 25: R² = 0.364 | `figures/le_r2_by_lag.py` | `le_r2_by_lag.json` → `numbers.edu_r2_lag25` | ±0.001 |
| Lag 50: R² = 0.171 | `figures/le_r2_by_lag.py` | `le_r2_by_lag.json` → `numbers.edu_r2_lag50` | ±0.001 |

### Robustness

| Paper Claim | Script | JSON → Key | Tolerance |
|-------------|--------|------------|-----------|
| Quadratic residualized R² = 0.03 | `robustness/robustness_tests.py` | `robustness_tests.json` → `numbers.Rob-quad-resid-R2` | ±0.01 |
| Bootstrap edu CI lower = 0.33 | `robustness/robustness_tests.py` | `robustness_tests.json` → `numbers.Rob-boot-edu-lo` | ±0.02 |
| Bootstrap GDP CI upper = 0.04 | `robustness/robustness_tests.py` | `robustness_tests.json` → `numbers.Rob-boot-gdp-hi` | ±0.01 |
| 99 former colonies | `robustness/colonial_vs_institutions.py` | `colonial_education_vs_institutions.json` → `n_colonies` | ±0 |
| Colonial edu 1950 → GDP 2015 R² = 0.465 | `robustness/colonial_vs_institutions.py` | `colonial_education_vs_institutions.json` → `r2_education_1950` | ±0.005 |

### Country Education (direct WCDE lookup)

| Paper Claim | Source CSV | Country | Year | Tolerance |
|-------------|-----------|---------|------|-----------|
| Korea 25% in 1950 | `cohort_lower_sec_both.csv` | Korea | 1950 | ±0.5 pp |
| Korea 94% in 1985 | `cohort_lower_sec_both.csv` | Korea | 1985 | ±0.5 pp |
| Taiwan 18% in 1950 | `cohort_lower_sec_both.csv` | Taiwan | 1950 | ±1.0 pp |
| China 31% in 1965 | `cohort_lower_sec_both.csv` | China | 1965 | ±2.0 pp |
| Cambodia 10% in 1975 | `lower_sec_both.csv` | Cambodia | 1975 | ±0.5 pp |
| Cuba 40% in 1960 | `cohort_lower_sec_both.csv` | Cuba | 1960 | ±1.0 pp |

### Country GDP (direct WDI lookup)

| Paper Claim | Country | Year | Tolerance |
|-------------|---------|------|-----------|
| Korea $1,038 | Korea | 1960 | ±$200 |
| Costa Rica $3,609 | Costa Rica | 1960 | ±$500 |
| Bangladesh $1,224 | Bangladesh | 2015 | ±$100 |
| Qatar ~$69,000 | Qatar | 2015 | ±$5,000 |

### LE and TFR Thresholds (direct WDI lookup)

| Paper Claim | Country | Year | Tolerance |
|-------------|---------|------|-----------|
| USA TFR = 3.65 in 1960 | USA | 1960 | ±0.05 |
| USA LE = 69.8 in 1960 | USA | 1960 | ±0.5 |
| Myanmar LE = 44.1 in 1960 | Myanmar | 1960 | ±1.0 |
| Myanmar TFR = 2.3 in 2015 | Myanmar | 2015 | ±0.2 |

### Cases

| Paper Claim | Script | JSON → Key | Tolerance |
|-------------|--------|------------|-----------|
| Indonesia GDP −14.5% (1997–98) | `robustness/asian_financial_crisis.py` | `asian_financial_crisis.json` → `numbers.indonesia_gdp_drop_1997_1998_pct` | ±0.001 |
| Korea expansion 2.13 pp/yr | derived | `(Korea-1985 − Korea-1955) / 30` | ±0.1 |
| Korea 9-fold GDP increase | derived | `GDP-Korea-1990 / GDP-Korea-1960` | ±1.5 |
| Costa Rica 3.5× Korea's GDP in 1960 | derived | `GDP-CostaRica-1960 / GDP-Korea-1960` | ±0.1 |
| 154 countries crossed both thresholds | `cases/development_threshold_count.py` | `development_threshold_count.json` → `numbers.countries_crossing_both` | ±0 |

### Baseline Groups

| Paper Claim | Script | JSON → Key | Tolerance |
|-------------|--------|------------|-----------|
| Low-GDP group β = 1.585 | `robustness/beta_by_baseline_group.py` | `beta_by_baseline_group.json` → `numbers.Grp-low-beta` | ±0.05 |
| Low-GDP group R² = 0.706 | `robustness/beta_by_baseline_group.py` | `beta_by_baseline_group.json` → `numbers.Grp-low-R2` | ±0.02 |
| High-GDP group β = 0.176 | `robustness/beta_by_baseline_group.py` | `beta_by_baseline_group.json` → `numbers.Grp-high-beta` | ±0.05 |

The full registry (351 entries) is in `scripts/verify_nations.py`. Every `reg()` call documents the name, expected value, source, JSON key path, paper section, and tolerance.

---

## 4. The Master Verifier

**File:** `scripts/verify_nations.py` (~1,990 lines)

### Purpose

Maintains a registry of every empirical number in the paper and verifies each one against source data or checkin JSONs. Acts as the single point of truth for whether the paper's numbers are correct.

### Registry Structure

Each number is registered with `reg()`:

```python
reg(name, value, source, detail, section, tol=0.001)
```

- **name**: Unique identifier (e.g., `"T1-obs"`, `"Korea-1950"`, `"PI-alone-beta"`).
- **value**: Expected value as stated in the paper.
- **source**: How to look it up — one of:
  - `"checkin"`: Read from a checkin JSON file. Detail = `(filename, dot.path.to.key)`.
  - `"wcde"`: Look up directly from WCDE CSV. Detail = `(filename, country, year)`.
  - `"wdi"`: Look up directly from World Bank CSV. Detail = `(indicator, country, year)`.
  - `"derived"`: Computed from other registered values (e.g., ratios, differences).
  - `"ref"`: Literature reference — flagged for manual check.
- **section**: Where in the paper this number appears. List of `(section_label, line_offset)` tuples.
- **tol**: Tolerance for comparison. Default 0.001.

### Registry Summary (351 entries)

| Source Type | Count | What It Covers |
|-------------|-------|----------------|
| checkin | 176 | Regression output, test statistics, computed values from JSON files |
| wdi | 82 | GDP, LE, TFR point values looked up from World Bank CSVs |
| derived | 62 | Ratios, rates, percentages computed from other verified values |
| wcde | 31 | Education levels looked up from WCDE CSVs |
| ref | 6 | Literature references (manual check) |

### Verification Phases

**Phase 1 — Data lookup**: For each registry entry, look up the actual value:
- `checkin`: Parse JSON file, navigate dot-path to the value.
- `wcde`: Read CSV, find country-year cell.
- `wdi`: Read CSV via country name mapping, find year column.
- `derived`: Computed after all other lookups complete (e.g., ratios of two verified values).
- `ref`: Flagged for manual check (literature values).

**Phase 2 — Comparison**: For each entry, check `|actual - expected| ≤ tolerance`. Mark PASS, FAIL, or MISSING.

**Phase 3 — Coverage scan**: Parse the `.tex` file, extract all numbers from each section, and report any that are *not* registered. Uses regex to find numbers, then filters out:
- Years (1800–2100)
- Structural numbers (section numbers, LaTeX formatting constants)
- Numbers registered for that section (with 15% relative tolerance for matching)

**Phase 4 — Section consistency**: For each registered number, verify it actually appears in the claimed section of the paper (and at the claimed line offset, if specified). Reports mismatches.

### Example Output

```
========================================================================
PAPER NUMBER VERIFICATION
Paper: paper/education_of_humanity.tex
Registry: 351 entries
========================================================================

  [checkin:table_1_main.json]
    ✓ T1-obs                    exp=1665       act=1665.0000
    ✓ T1-countries              exp=185        act=185.0000
    ✓ PI-alone-beta             exp=15.4       act=15.3690

  [wcde]
    ✓ Korea-1950                exp=24.8       act=24.8000
    ✓ Taiwan-1950               exp=17.75      act=17.7500

  [wdi]
    ✓ GDP-Korea-1960            exp=1038       act=1038.3200
    ✓ TFR-USA-1960              exp=3.65       act=3.6540

  ...

========================================================================
SUMMARY: 351/351 PASS, 0 FAIL, 0 MISSING, 6 REF (manual check)
COVERAGE: 0 lines with unregistered numbers
========================================================================
```

Exit code 0 if all pass, 1 if any fail. Report written to `checkin/VERIFICATION_REPORT.md`.

### Fast Mode (`--fast`)

When run with `--fast`, skips script execution entirely. Only reads existing checkin JSONs and WCDE/WDI CSVs. Takes ~2 seconds. This is what `make verify` uses.

---

## 5. Verification Scripts

These scripts check specific factual claims cited in the paper. Each reads source data and writes a JSON to `checkin/`.

### verify_country_education.py

**Output:** `checkin/country_education.json`

**Data:** `wcde/data/processed/completion_both_long.csv`

**What it does:** Verifies ~20 point claims about country education levels at specific years. For example: "Korea's lower secondary completion was 25% in 1950" — the script looks up the actual WCDE value and compares.

**Claims verified:**
- Korea: 1950, 1955, 1960, ..., 1985 (full trajectory, plus pp/yr expansion rate)
- Taiwan: 1950
- Singapore: 1950, 1995
- Philippines: 1950
- China: 1950, 1965, 1980, 1990
- Cambodia: 1975, 1985, 1995, 2000
- Vietnam: 1960, 2015
- Cuba: 1960
- Bangladesh: 1960
- India: 1950, 2015

**Also computes:** Korea 5-year expansion rates (pp/yr for each period), country expansion rates cited in the paper.

**Tolerances:** ±0.5–2.0 percentage points, reflecting rounding in the paper.

### verify_country_gdp.py

**Output:** `checkin/country_gdp.json`

**Data:** `data/gdppercapita_us_inflation_adjusted.csv`

**What it does:** Verifies GDP per capita claims. Korea 1960 ($1,038), Costa Rica 1960 ($3,609), Korea 1990 ($9,673), etc.

**Tolerances:** ±15% — GDP values are cited approximately in the paper.

### verify_country_le_tfr.py

**Output:** `checkin/country_le_tfr.json`

**Data:** World Bank LE and TFR CSVs, plus WCDE LE/TFR where available.

**What it does:** Verifies 21 life expectancy and TFR claims across 10+ countries (USA, Japan, Sri Lanka, Myanmar, Uganda, India, Cuba, Korea, China, Bangladesh).

**Tolerances:** ±0.5–1.5 for LE (years), ±0.1–0.2 for TFR.

### verify_table4_crossings.py

**Output:** `checkin/table4_crossings.json`

**Data:** World Bank LE and TFR CSVs, WCDE education CSVs.

**What it does:** For each country in Table 4 (the cases), finds the year when:
- TFR first dropped below 3.65 (USA 1960 value)
- LE first exceeded 69.8 (USA 1960 value)
- Education reached 50% lower secondary completion

Computes the lag between education reaching 50% and the LE/TFR crossing. These lags map to "generations" in the paper.

### verify_china_cr.py

**Output:** `checkin/china_cr.json`

**Data:** WCDE cohort data, World Bank LE.

**What it does:** Verifies claims about China's Cultural Revolution era. Computes cohort education gains during 1965–1980 (when the CR disrupted schooling). Also runs the peer comparison: finds countries with similar education levels (25–38%) in 1965, computes their LE and LE gains, and compares to China's trajectory.

### verify_college_le_gradient.py

**Output:** `checkin/college_le_gradient.json`

**Data:** WCDE college completion, World Bank LE.

**What it does:** Among countries with >85% lower secondary completion by 2010, examines the correlation between college completion and life expectancy. Tests whether education continues to matter at higher levels.

**Key result:** r=0.44, 5.5-year LE gap between lowest and highest college quartiles.

### verify_costa_rica_korea.py

**Output:** `checkin/costa_rica_korea.json`

**Data:** WCDE education, World Bank GDP.

**What it does:** Verifies the Korea vs Costa Rica comparison — Costa Rica had 3.5x Korea's GDP in 1960, but Korea invested in education and overtook by 1990 (9-fold GDP increase vs Costa Rica's 1.7-fold).

### verify_figure2_betas.py

**Output:** `checkin/figure2_betas.json`

**Data:** WCDE education (completion_both_long.csv).

**What it does:** Computes the country-specific beta coefficients shown in Figure 2. For each country, runs a sliding-window regression of child education on parent education (25-year lag) and reports the beta at different baseline levels.

### verify_table_a1_cutoffs.py

**Output:** `checkin/table_a1_cutoffs.json`

**Data:** WCDE education, World Bank outcomes.

**What it does:** Verifies Table A1 numbers — the entry-cohort analysis at specific cutoffs. Computes N, countries, R², and betas at the 30% and 50% entry thresholds.

### verify_kerala.py

**Output:** `checkin/kerala.json`

**Data:** `data/kerala_srs.csv` (Kerala Sample Registration System data).

**What it does:** Verifies Kerala's TFR and LE crossing dates from Indian SRS data. Kerala crossed the LE threshold ~3 generations after education investment began, supporting the generational-lag framework.

### edu_vs_gdp_predicts_le.py

**Output:** `checkin/edu_vs_gdp_predicts_le.json`

**Data:** WCDE education, World Bank GDP and LE.

**What it does:** Core test — does education or GDP better predict life expectancy at T+25? Builds a panel with education at T, GDP at T, and LE at T+25. Computes country FE R² for education→LE and GDP→LE at different entry thresholds and ceilings.

**Key result:** At <10% entry, education R²=0.628, GDP R²=0.016. Education is 39x more predictive.

### regression_tables.py

**Output:** `checkin/regression_tables.json`

**Data:** WCDE education, World Bank GDP/LE/TFR/U5MR.

**What it does:** Produces the formal regression output for Table 2b (residualized GDP). For each outcome (LE, TFR, child education, U5MR) at the 90% ceiling:
1. Education R² (education alone predicting outcome)
2. Raw GDP R² (GDP alone predicting outcome)
3. Residualized GDP R² (GDP after removing education's contribution)
4. P-values for each

Uses clustered standard errors (by country).

**Key result:** Residualized GDP R² never exceeds 0.023 (for U5MR). For LE: resid R²=0.003, p=0.56. For TFR: resid R²=0.000, p=0.98.

### u5mr_residual_by_year.py

**Output:** `checkin/u5mr_residual_by_year.json`

**Data:** WCDE education, World Bank GDP, U5MR.

**What it does:** Tests whether the small U5MR signal (residualized GDP R²=0.023) is driven by post-2000 MDG-era health spending. Splits sample at year 2000 and runs residualization separately.

**Key result:** Pre-2000 resid R²=0.008, post-2000 resid R²=0.032. The signal is post-MDG, consistent with targeted health spending (not GDP causing health improvements).

### regime_education_test.py

**Output:** `checkin/regime_education_test.json`

**Data:** WCDE education, Polity5 regime scores (`data/p5v2018.xls`).

**What it does:** Tests whether democracies invest more in education than autocracies. Computes education gain rates (pp/decade) by regime type at 15–20 year lags.

**Key finding:** Both regime types invest, roughly equally (democratic mean ~10.3 pp/decade, autocratic ~8.1). Education investment is not regime-dependent — it's a universal state strategy.

### colonial_education_vs_institutions.py

**Output:** `checkin/colonial_education_vs_institutions.json`

**Data:** WCDE education, Polity5, World Bank GDP.

**What it does:** Tests the Acemoglu-Johnson-Robinson claim that colonial institutions (not education) drive development. Takes 99 former colonies, computes:
- R² of 1950 education levels predicting 2015 GDP
- R² of 1950 Polity scores predicting 2015 GDP
- R² of education + religion predicting GDP

**Key finding:** Colonial-era education (1950) explains 46.5% of 2015 GDP. Adding religion or institutional variables adds nothing. Education is the channel.

### verify_the_case.py

**Output:** `checkin/the_case.json`

**Data:** Same as above + existing checkin JSONs.

**What it does:** Verifies numbers in the companion paper "Education of Humanity" (shorter policy version, `the_case.tex`). Cross-references values from existing checkin JSONs to ensure consistency between the two papers.

### china_mean_yrs_vs_peers.py

**Output:** `checkin/china_mean_yrs_vs_peers.json`

**Data:** WCDE raw proportion data, completion data.

**What it does:** Computes China's mean years of schooling from raw WCDE age-education distribution data, and compares to completion rates. Tests whether China's education claims hold up under different measurement approaches.

---

## 6. Analysis Scripts — Main Tables

### table_1_main.py

**Output:** `checkin/table_1_main.json`

**Data:** WCDE lower secondary (both + female), World Bank GDP.

**What it does:** Produces Table 1 — the core regression of the paper.

**Panel construction:**
- Education measure: WCDE lower secondary completion, both sexes, age 20-24.
- Lag: 25 years (one generational interval).
- Child education at year T is predicted by parent-generation education at T-25.
- Outcome years: 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015.
- Countries: 185 (all WCDE countries minus regional aggregates).
- N = 1,665 country-year observations.

**Models (country fixed effects):**
1. child_edu ~ parent_edu: β=0.483, SE=0.034, R²=0.457
2. child_edu ~ log_gdp: β=15.369, SE=2.354, R²=0.256
3. child_edu ~ parent_edu + log_gdp: β_edu=0.519, β_gdp=5.47, R²=0.556

**Also runs:**
- Female-only specification: β=0.42, R²=0.39 (same sample)
- Parental income collapse test (using statsmodels with clustered SEs):
  - GDP alone: β=15.4, R²=0.293
  - GDP conditional on education: β=4.3, p=0.04 (72% drop)
  - Education alone: R²=0.553; conditional on GDP: R²=0.475 (small drop)

### twfe_child_edu.py

**Output:** `checkin/twfe_child_edu.json`

**Data:** Same as table_1_main.py.

**What it does:** Runs Table A1 — two-way fixed effects (country + time period). This absorbs global time trends, testing whether education's effect survives controlling for worldwide educational expansion.

**Key result:** β drops from 0.483 to 0.083, R² from 0.457 to 0.009. This is expected: when global expansion is absorbed, within-country variation is small but still significant (t=5.5).

**Why this matters:** Confirms the paper's argument that while β is mechanistically real, much of the observed expansion tracks a global wave. The t-statistic confirms the within-country within-period signal is real, just small after absorbing the trend.

### education_vs_gdp_by_cutoff.py

**Output:** `checkin/education_vs_gdp_by_cutoff.json`

**Data:** WCDE education, World Bank GDP and child education.

**What it does:** The core education-vs-GDP horse race. Sweeps over entry thresholds (10–90%) with a 30% ceiling:
- For each threshold: filter to countries that reached it, compute country FE R² for education vs GDP predicting child-generation education.

**Key result at 30% entry:** Education R²=0.699, GDP R²=0.214, ratio=3.3x. Education is 3.3 times more predictive than GDP.

**Also computes:** Results with no cutoff (all observations): education R²=0.533, GDP R²=0.245.

### beta_by_ceiling_cutoff.py

**Output:** `checkin/beta_by_ceiling_cutoff.json`

**Data:** WCDE education.

**What it does:** Computes the education beta coefficient at different ceiling cutoffs (20%, 30%, 50%, 90%, no cutoff). Shows how β changes as you include more educated countries.

**Key result:** β=2.855 at ≤20%, β=1.830 at ≤50%, β=1.236 at ≤90%, β=1.041 at no cutoff. β>1 everywhere except near ceiling — each generation amplifies its parents' education gains.

---

## 7. Analysis Scripts — Education vs GDP

These scripts all test the same question from different angles: does GDP predict development outcomes independently of education?

### edu_vs_gdp_residualized.py

**Output:** `checkin/edu_vs_gdp_residualized.json`

**Data:** WCDE education, World Bank GDP and LE.

**What it does:** Runs the Frisch-Waugh-Lovell residualization for life expectancy.
1. First stage: regress log-GDP on education (country FE) → get GDP residual (the part of GDP not explained by education).
2. Second stage: regress LE on the GDP residual (country FE) → R².

Sweeps over entry thresholds (10–90%) and ceilings (30%, 50%, 60%, 90%).

**Key finding:** At entry=10%, ceiling=90%: education→GDP R²=0.417 (education explains 42% of GDP variation). The remaining 58% of GDP — the part not driven by education — has no predictive power for LE.

### edu_vs_gdp_tfr_residualized.py

**Output:** `checkin/edu_vs_gdp_tfr_residualized.json`

**Data:** WCDE education, World Bank GDP and TFR.

**Same design as above, but outcome = TFR.** Also runs the analysis at the primary education level to test whether fertility responds to basic literacy.

**Key result:** Residualized GDP R²≈0 for TFR. Additionally: primary education R²=0.65 for TFR — fertility responds even to basic education.

### edu_vs_gdp_entry_ceiling.py

**Output:** `checkin/edu_vs_gdp_entry_ceiling.json`

**Data:** WCDE education, World Bank GDP and LE.

**What it does:** Full threshold × ceiling sweep for the entry-cohort design. Tests every combination of entry threshold (10–90%) and ceiling (30–90%).

**Purpose:** Robustness check — confirms the education > GDP result holds across all sample restrictions.

### edu_vs_gdp_entry_threshold.py

**Output:** `checkin/edu_vs_gdp_entry_threshold.json`

**Data:** Same as above.

**What it does:** Sensitivity analysis on the entry threshold alone (holding ceiling at 90%). Shows education dominates GDP at every threshold.

### edu_vs_gdp_by_level.py

**Output:** `checkin/edu_vs_gdp_by_level.json`

**Data:** WCDE primary, lower secondary, upper secondary education; World Bank outcomes.

**What it does:** Repeats the residualization analysis at three education levels. Tests whether the result is specific to lower secondary or holds generally.

### edu_vs_gdp_child_edu_residualized.py

**Output:** `checkin/edu_vs_gdp_child_edu_residualized.json`

**Data:** WCDE education, World Bank GDP.

**What it does:** Residualization where the outcome is child-generation education (not LE or TFR). Tests whether GDP independently predicts educational attainment of the next generation after controlling for parent education.

### female_education_residualized.py

**Output:** `checkin/female_education_residualized.json`

**Data:** WCDE female education, World Bank outcomes.

**What it does:** Runs the full residualization sweep using female education instead of both-sexes. Tests whether the results change when using female education (which theory suggests should be the stronger predictor for fertility and child health).

**Key results:** Female education R² slightly higher for LE (0.531 vs 0.472) and TFR (0.498 vs 0.478).

---

## 8. Analysis Scripts — Robustness

### robustness_tests.py

**Output:** `checkin/robustness_tests.json`

**Data:** WCDE education, World Bank GDP and LE.

**What it does:** Three robustness tests:

1. **Nickell bias test:** Short panels with lagged dependent variables can produce biased FE estimates (Nickell 1981). Uses Anderson-Hsiao IV (2-period lag as instrument) to check whether the standard FE estimate is biased.

2. **Nonlinearity test:** The first-stage residualization assumes a linear education→GDP relationship. Adds a quadratic education term to the first stage and checks whether the residualized GDP R² changes.
   - Result: Quadratic resid R²=0.03 (essentially unchanged). Linearity is fine.

3. **Bootstrap confidence intervals:** 1,000 bootstrap replications of the education R² vs residualized GDP R² comparison.
   - Result: Education 95% CI [0.33, 0.59], GDP 95% CI [0.00, 0.04]. No overlap.

### le_r2_by_lag.py

**Output:** `checkin/le_r2_by_lag.json`

**Data:** WCDE education, World Bank GDP and LE.

**What it does:** Computes education's predictive power at different lag horizons: 0, 10, 15, 25, 50, 75, 100 years. Also computes GDP's predictive power at the same lags for comparison.

**Key result:**
- Education R²: 0.562 (lag 0), 0.364 (lag 25), 0.171 (lag 50), 0.085 (lag 75)
- GDP R²: 0.321 (lag 0), declining similarly
- Education dominates GDP at every lag.
- The 25-year peak corresponds to one generational interval (PTE).

### lag_sensitivity.py

**Output:** `checkin/lag_sensitivity.json`

**Data:** WCDE education, World Bank GDP, LE, TFR.

**What it does:** Tests whether the "zero GDP effect" result is specific to the 25-year lag. Runs the residualization at lags of 10, 15, 20, 25, and 30 years.

**Key result:** Residualized GDP R² < 0.02 for LE and < 0.01 for TFR at all lags tested. The zero-GDP finding is not an artifact of the 25-year lag choice.

### twfe_all_outcomes.py

**Output:** `checkin/twfe_all_outcomes.json`

**Data:** WCDE education, World Bank GDP, LE, TFR, U5MR.

**What it does:** Runs two-way FE (country + time) for all four outcomes. Confirms that the education signal survives even the most conservative specification (absorbing all global time trends).

### residual_by_outcome_year_all.py

**Output:** `checkin/residual_by_outcome_year_all.json`

**Data:** WCDE education, World Bank GDP, all outcomes.

**What it does:** Tests whether the residualized GDP result changes depending on which outcome years are included. Runs the analysis restricting to outcomes before 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020. Checks temporal stability.

### u5mr_by_edu_level.py

**Output:** `checkin/u5mr_by_edu_level.json`

**Data:** WCDE education, World Bank GDP, U5MR.

**What it does:** Investigates the small post-2000 GDP signal for child mortality by education band (0–10%, 10–20%, 20–40%, etc.). Tests whether the signal is concentrated in specific education ranges.

---

## 9. Analysis Scripts — Case Studies

### beta_vs_baseline.py

**Output:** `checkin/beta_vs_baseline.json`

**Data:** WCDE education (completion_both_long.csv).

**What it does:** For 9 countries (USA, Korea, Taiwan, Philippines, Singapore, etc.), computes the education β coefficient at different baseline levels using a sliding-window approach. Each country gets a trajectory of β values as education rises from low to high.

**Key insight:** β starts high (>5 for Korea at low baselines) and falls toward 0 at high baselines. This is the generational amplification coefficient — each educated generation amplifies education in the next, but the effect compresses as you approach the ceiling.

### beta_by_baseline_group.py

**Output:** `checkin/beta_by_baseline_group.json`

**Data:** WCDE education, World Bank GDP.

**What it does:** Splits all 185 countries into three groups by 1975 baseline GDP:
- Low (bottom tercile): β=1.585, R²=0.706
- Medium: β=0.713, R²=0.716
- High (top tercile): β=0.176, R²=0.442

**Key insight:** Education's amplification effect is strongest in the poorest countries. This is where investment has the highest return.

### asian_financial_crisis.py

**Output:** `checkin/asian_financial_crisis.json`

**Data:** WCDE education, World Bank GDP.

**What it does:** Natural experiment — the 1997–98 Asian financial crisis caused GDP to collapse in Indonesia (-14.5%), Thailand (-8.8%), Malaysia (-9.6%), Philippines (-3.0%). Tests whether education growth stalled when income crashed.

**Key finding:** Education continued growing uninterrupted:
- Indonesia: GDP -14.5%, but education +5.4pp (1995–2000)
- Thailand: lower secondary +13.4pp (1995–2000), *faster* than prior period (+10.0pp, 1990–1995)
- Korea: college +4.5pp (1995–2000)

This is the income-removal test: if GDP caused education, removing GDP should stall education. It doesn't. Education trajectories are independent of GDP shocks.

### development_threshold_count.py

**Output:** `checkin/development_threshold_count.json`

**Data:** World Bank LE and TFR.

**What it does:** Counts how many countries have ever crossed both development thresholds (TFR < 3.65 AND LE > 69.8, the USA 1960 values). Used in the abstract and conclusion.

**Result:** 154 countries have crossed both thresholds by 2022.

---

## 10. WCDE Analysis Scripts

### wcde/long_run_generational.py

**Output:** `checkin/long_run_generational.json`

**Data:** Cohort-reconstructed WCDE data.

**What it does:** Tests whether the 25-year generational lag holds over 100+ years (1900–2015). Takes 28 countries with reliable pre-1960 reconstructed data (Japan, Korea, Taiwan, UK, USA, Germany, France, Spain, Australia, NZ, Canada, Argentina, Chile, Cuba, Uruguay, Costa Rica, Sri Lanka, Hong Kong, etc.).

**Methodology:** For each country, regress child-cohort education on parent-cohort education (25-year lag). Compute pooled OLS and country FE.

**Key result:** β=0.90 (pooled), β=0.96 (FE). The generational transmission mechanism holds across a century of data.

### wcde/education_outcomes.py

**Output:** `checkin/education_outcomes.json`

**Data:** WCDE education, World Bank GDP/LE/TFR.

**What it does:** Table 2 — forward prediction. Does education at time T predict GDP, LE, and TFR at T+25?

**Panel:**
- Education measurement years: 1960, 1965, 1970, 1975, 1980, 1985, 1990.
- Outcomes at T+25.
- 185 countries.

**Models:**
- Panel A (education only, country FE):
  - Education → GDP(T+25): β=0.012 (1.2% per pp), R²=0.354
  - Education → LE(T+25): β=0.109 (0.11 years per pp), R²=0.382
  - Education → TFR(T+25): β=-0.032 (−0.032 children per pp), R²=0.362

- Panel B (education + GDP, country FE):
  - Both predictors: β_edu=0.485, β_gdp=3.78, R²=0.500
  - N=828

**Forward R² symmetry test:** GDP(T) → education(T+25): R²=0.259 (weaker reverse direction — education predicts forward better than GDP does).

---

## 11. Key Methodological Choices

### Why Country Fixed Effects?

Country FE absorb all time-invariant country differences (geography, culture, colonial history, ethnic composition). This means the regressions identify effects from *within-country* variation over time — "when Korea's education rose from 25% to 94%, what happened to Korea's outcomes?" rather than "do more educated countries have better outcomes?"

### Why 25-Year Lag?

The biological argument: humans have ~18 years of juvenile dependency, followed by ~7 years before household formation. An educated 20-year-old becomes a decision-making parent around age 25. So education at time T produces development outcomes at T+25.

Empirically: tested at lags 0, 10, 15, 20, 25, 30, 50, 75, 100 years. The pattern is informative at all lags, but 25 years is the natural generational timescale. The "zero GDP" result holds at all lags tested (10–30 years).

### Why Lower Secondary Completion?

Completion (not enrollment) measures actual education received. Lower secondary is the level that reliably produces the behavioral changes (fertility decisions, health practices, economic participation) central to the argument. Primary is necessary but not sufficient; upper secondary and college are not yet universal in enough countries for robust analysis.

Both sexes (not female-only) is the default because the theory argues for household-level effects. Female-only specifications are run as robustness checks and show slightly stronger effects for fertility and child health.

### Why Entry-Cohort Design?

Including all country-years biases toward high-education countries (they have more data points and less noise). The entry-cohort design tracks countries from the moment they reach a threshold (e.g., 10% or 30% lower secondary completion), following their trajectory forward. This focuses on the dynamic: *what happens when education expands from low to high?*

### Why Ceiling Constraint?

Countries near 100% completion have ceiling compression — education can't rise much further, so β→0 mechanically. The ceiling constraint (e.g., ≤30%, ≤50%, ≤90%) keeps the sample in the dynamic range where education is actively expanding. Results are shown at multiple ceilings for transparency.

### Why Residualization (Frisch-Waugh-Lovell)?

GDP is on the causal path: education → GDP → outcomes. Controlling for current GDP blocks this mediated pathway (bad control problem). Instead, the residualization isolates GDP variation that is *orthogonal to education*. If GDP has independent causal power, this orthogonal component should predict outcomes. It doesn't (R² < 0.025, p > 0.1 for all outcomes except U5MR post-2000).

### Country Exclusions

The 30+ WCDE regional aggregates (Africa, Asia, World, etc.) are excluded because they are weighted averages, not independent observations. Including them would violate independence assumptions and double-count data. After exclusion: 185 countries.

---

## 12. Data Flow Diagram

```
wcde/data/processed/*.csv  (185 countries, 1950–2015, 5-year intervals)
      │
      │         World Bank WDI CSVs (data/*.csv)
      │              │
      ▼              ▼
  ┌──────────────────────────────────┐
  │     40 Analysis Scripts          │
  │  (scripts/Makefile orchestrates) │
  │                                  │
  │  cases/                          │
  │    country_education.py          │
  │    country_gdp.py                │
  │    country_le_tfr.py             │
  │    threshold_crossings.py  ...   │
  │                                  │
  │  tables/                         │
  │    table_1_main.py               │
  │    twfe_child_edu.py        │
  │    regression_tables.py          │
  │                                  │
  │  residualization/                │
  │    education_vs_gdp.py           │
  │    education_vs_tfr.py           │
  │    by_entry_ceiling.py     ...   │
  │                                  │
  │  robustness/                     │
  │    robustness_tests.py           │
  │    lag_sensitivity.py            │
  │    asian_financial_crisis.py ... │
  │                                  │
  │  figures/                        │
  │    le_r2_by_lag.py           │
  │    beta_vs_baseline.py       │
  └──────────────────────────────────┘
      │
      ▼
  checkin/*.json  (42 JSON files with all computed values)
      │
      │         paper/education_of_humanity.tex
      │              │
      ▼              ▼
  ┌──────────────────────────────────┐
  │  verify_nations.py               │
  │  (351 registered claims)         │
  │                                  │
  │  Phase 1: Look up actual values  │
  │  Phase 2: Compare exp vs actual  │
  │  Phase 3: Scan .tex for          │
  │           unregistered numbers   │
  │  Phase 4: Section consistency    │
  └──────────────────────────────────┘
      │
      ▼
  checkin/.verified         (timestamp = all passed)
  checkin/VERIFICATION_REPORT.md  (human-readable report)
```
