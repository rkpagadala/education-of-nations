# Script Importance Guide

**Which scripts matter most, and why.**

The paper makes one central claim — education causes development, GDP doesn't independently — and builds it in layers. The scripts below are ranked by how load-bearing they are for the argument.

---

## Tier 1 — The argument collapses without these

| Script | What it produces | Why it matters |
|--------|-----------------|----------------|
| `scripts/tables/panel_full_fe.py` | Full-panel one-way FE diagnostic (β=0.483, N=1665, 185 countries). Parental income collapse test (GDP β drops 72% conditional on education). Cited in year-FE discussion and footnotes. | If this breaks, there's no paper. |
| `scripts/residualization/by_gdp_cutoff.py` | Headline Table 1 (β=1.376, N=629, 105 countries, <30% active-expansion subsample). Also R²-vs-cutoff curve 10%–90%. | This is what the paper's Table 1 actually shows. |
| `scripts/tables/regression_tables.py` | Table 2b: residualized GDP R² never exceeds 0.023. LE resid R²=0.003 (p=0.56), TFR resid R²=0.000 (p=0.98). | The "GDP has no independent effect" result. The single most provocative claim. |
| `scripts/wcde/education_outcomes.py` | Table 2: forward prediction. Education at T predicts GDP, LE, TFR at T+25. | Eliminates reverse causality by design. |

**If a reviewer has 10 minutes, they should verify these three.**

---

## Tier 2 — The argument is weakened without these

| Script | What it produces | Why it matters |
|--------|-----------------|----------------|
| `scripts/residualization/education_vs_gdp.py` | Frisch-Waugh-Lovell residualization for LE across entry thresholds and ceilings. | The full sweep showing orthogonal GDP predicts nothing. |
| `scripts/residualization/education_vs_tfr.py` | Same residualization for TFR. Primary education R²=0.65 for TFR. | Extends the zero-GDP result to fertility. Shows even basic education matters. |
| `scripts/figures/beta_vs_baseline.py` | Figure 1: β trajectories for 9 countries. Korea β=6.5 at low baseline, USA β=1.9. | The visual heart of the paper. Shows generational amplification (β>1) compressing toward ceiling. |
| `scripts/figures/le_r2_by_lag.py` | Figure 3: education vs. income predictive power across lag lengths 0–100 years. | Education R² persists across four generations; GDP collapses within one. Visual proof of multi-generational persistence. |
| `scripts/robustness/robustness_tests.py` | Bootstrap CIs: education [0.33, 0.59], GDP [0.00, 0.04]. Nickell bias check. Nonlinearity check. | No overlap in confidence intervals. A reviewer will go straight here. |
| `scripts/wcde/long_run_generational.py` | 28 countries, 1900–2015. β=0.96 (FE) over a century. | Without this, the 25-year lag choice looks arbitrary. |

---

## Tier 3 — Defends against specific objections

| Script | Objection it blocks | Key result |
|--------|---------------------|------------|
| `scripts/robustness/asian_financial_crisis.py` | "GDP causes education" | GDP crashed (Indonesia −14.5%), education kept growing (+5.4pp). Income-removal natural experiment. |
| `scripts/robustness/colonial_vs_institutions.py` | "Institutions cause development" (Acemoglu) | 1950 education explains 46.5% of 2015 GDP in 99 colonies. Adding institutions adds nothing. |
| `scripts/robustness/regime_education.py` | "It's just democracy" | Democracies (~10.3 pp/decade) and autocracies (~8.1 pp/decade) invest roughly equally. |
| `scripts/robustness/twfe_child_edu.py` | "It's just a global trend" | β shrinks to 0.083 under two-way FE. Signal survives absorbing global time trends. |
| `scripts/robustness/goodman_bacon_decomposition.py` | "Two-way FE kills the result" | Goodman-Bacon (2021) decomposition: clean comparisons give β̄=11.3, contaminated give β̄=1.1. The estimator, not the evidence, is broken. |
| `scripts/robustness/callaway_santanna.py` | "Two-way FE kills the result" | Callaway–Sant'Anna (2021) heterogeneity-robust ATT=7.9 (95% CI: 4.4–11.0). Event study: +1.3pp at onset → +21.4pp at 35yr, all significant. |
| `scripts/robustness/lag_sensitivity.py` | "You cherry-picked 25 years" | Residualized GDP R² < 0.02 at all lags tested (10–30 years). |
| `scripts/robustness/u5mr_residual_by_year.py` | "GDP helps child mortality" | The small U5MR signal (R²=0.023) is post-2000 only — MDG health spending, not GDP causing health. |
| `scripts/robustness/threshold_robustness.py` | "Your threshold is arbitrary" | Every country crosses under loose (TFR<4.5, LE>65), main, and strict (TFR<2.5, LE>72.6) specs. Ordering never changes. |
| `scripts/robustness/grandparent_effect.py` | "It's just the current generation" | Adding grandparent education raises child-edu R² by 5.2pp and LE R² by 3.6pp. Grandfather and grandmother contribute equally in low-education settings. |

---

## Tier 4 — Case studies and supporting evidence

| Script | What it supports |
|--------|-----------------|
| `scripts/cases/country_education.py` | Korea 25%→94%, Cambodia collapse, China stall. The narrative backbone of §5. |
| `scripts/cases/threshold_crossings.py` | Table 4: generational lags for each case country. Connects data to the PT mechanism. |
| `scripts/cases/china_cultural_revolution.py` | China CR peer comparison. Shows disruption effect and recovery. |
| `scripts/cases/costa_rica_korea.py` | Costa Rica had 3.5× Korea's GDP in 1960. Korea invested in education, overtook by 1990. |
| `scripts/robustness/beta_by_baseline_group.py` | Low-GDP β=1.585 vs High-GDP β=0.176. Policy punchline: invest where baselines are lowest. |
| `scripts/robustness/beta_by_ceiling_cutoff.py` | β>1 at every ceiling except near 100%. Amplification is universal. |
| `scripts/cases/development_threshold_count.py` | 154 countries crossed both thresholds. Used in abstract and conclusion. |
| `scripts/cases/country_gdp.py` | GDP point values cited in case studies. |
| `scripts/cases/country_le_tfr.py` | LE and TFR point values cited in case studies. |
| `scripts/cases/kerala.py` | Kerala SRS data. Sub-national case supporting generational-lag framework. |

---

## Reviewer's shortcut

If a reviewer has 30 minutes, run Tier 1 (3 scripts) plus `robustness_tests.py`. Everything else strengthens the case but the argument stands on those four:

```bash
make setup
.venv/bin/python scripts/tables/panel_full_fe.py
.venv/bin/python scripts/tables/regression_tables.py
.venv/bin/python scripts/wcde/education_outcomes.py
.venv/bin/python scripts/robustness/robustness_tests.py
```
