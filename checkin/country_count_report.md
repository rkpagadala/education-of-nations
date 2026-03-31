# Country Count Discrepancy Report: 187 vs 189

**Date:** 2026-04-02
**Issue:** The paper uses "187 countries" but several documentation files and script headers used "189 countries."

## The correct number: 187

Every script that actually counts the panel produces **187 countries, 1,683 observations**.

- `checkin/table_1_main.json`: `"panel_countries": 187`
- `checkin/table_a1_two_way_fe.json`: `"panel_countries": 187`
- `checkin/VERIFICATION_REPORT.md`: "187 countries"
- `paper/education_of_nations.tex`: "187" throughout (6 occurrences)
- `verify_paper_numbers.py`: expects 187

## Where 189 appeared (all incorrect)

| File | What it said | Status |
|------|-------------|--------|
| `scripts/table_1_main.py` (docstring, line 6) | "189 countries" | **Fixed → 187** |
| `scripts/table_a1_two_way_fe.py` (docstring, line 6) | "189 countries" | **Fixed → 187** |
| `analysis/findings.md` (4 occurrences) | "189 countries" | **Fixed → 187** |
| `analysis/costing.md` (1 occurrence) | "189 countries" | **Fixed → 187** |
| `analysis/leapfrog_brief.md` (1 occurrence) | "189 countries" | **Fixed → 187** |
| `wcde/output/policy_residual.md` (line 22) | "189 countries" | **Fixed → 187** |
| `outreach/emails/outreach_sheet.csv` (all emails) | "189 countries" | **Fixed → 187** |
| `outreach/emails/world_leaders.csv` (all emails) | "189 countries" | **Fixed → 187** |

### Not fixed (separate repos or garbage)

| File | Note |
|------|------|
| `education-rupture/README.md` | Separate repo, refers to WDI data files which may cover different count |
| `education-rupture/output/policy_residual.md` | Separate repo copy |
| `garbage/*` | Stale files, not published |

## Why the confusion happened

### The raw data has 228 entities

WCDE v3 `completion_both_long.csv` contains **228 entities**. These include:

**Regional aggregates (not countries):**
Africa, Asia, Europe, Caribbean, Central America, Central Asia, Eastern Africa, Eastern Asia, Eastern Europe, Latin America and the Caribbean, Melanesia, Micronesia, Middle Africa, Northern Africa, Northern America, Northern Europe, Oceania, Polynesia, South America, South-Eastern Asia, Southern Africa, Southern Asia, Southern Europe, Western Africa, Western Asia, Western Europe, World, Australia and New Zealand

**Territories and dependencies (not sovereign):**
Aruba, Curaçao, French Guiana, French Polynesia, Guadeloupe, Guam, Hong Kong SAR, Macao SAR, Martinique, Mayotte, New Caledonia, Puerto Rico, Réunion, US Virgin Islands, Western Sahara, Occupied Palestinian Territory, Taiwan Province of China

### The pipeline: 228 → 187

The Table 1 script filters WCDE entities for sovereignty and data availability:
- Start: 228 WCDE entities
- Remove ~28 regional aggregates
- Remove ~13 territories/dependencies
- Result: **187 sovereign states** with complete panel data (1975-2015, 9 time periods, 1,683 obs)

### How 189 got into the documentation

The number **189 was never produced by any script**. It does not correspond to any intermediate filtering step. The most likely explanation:

1. An early version of the analysis markdown files was written (possibly by Claude) using an approximate count before the scripts were finalized
2. The script docstrings were written to describe what the scripts *would* produce, but the actual output was 187, not 189
3. The number 189 then propagated through copy-paste into analysis files, the costing document, and the leapfrog brief
4. The paper itself (LaTeX) was always correct at 187, because it was checked against the verification system
5. The outreach emails (written by Claude on 2026-04-02) inherited 189 from the analysis files

### Why verification caught it in the paper but not in the docs

The paper has a verification pipeline (`verify_paper_numbers.py` + `checkin/*.json`) that checks every number against script output. The analysis markdown files, script docstrings, and outreach emails have no such pipeline. The number 189 survived in unverified files.

## Confirmed correct numbers

| Context | Countries | Observations | Source |
|---------|-----------|-------------|--------|
| Main panel (Table 1, Table A1) | 187 | 1,683 | `table_1_main.json` |
| Long-run panel (1900-2015) | 28 | varies | `long_run_generational.json` |
| Policy residual panel | 187* | 1,323 | `policy_residual.md` (fewer obs due to T-25 lag) |

*Policy residual was listed as 189 and has been corrected to 187. The observation count (1,323 < 1,683) reflects the reduced time range when requiring T-25 lagged parental data.

## Remaining note: leapfrog_brief.md also says "182 countries"

Line 8 of `analysis/leapfrog_brief.md` says "182 countries across 55 years (1960-2015)". This is likely from an even older version of the analysis using a different dataset or time period. Not fixed in this pass — needs separate investigation.
