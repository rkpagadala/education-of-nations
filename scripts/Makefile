# scripts/Makefile — build checkin JSONs from analysis scripts
#
# Each script reads source data and writes a JSON to ../checkin/.
# Only reruns if the script or its data dependencies changed.
#
# Usage:
#   make          — build all checkin JSONs
#   make clean    — remove all checkin JSONs
#   make <name>   — build one, e.g. make regime

PYTHON = python3
CHECKIN = ../checkin
DATA = ../data
WCDE = ../wcde/data/processed
RESID_SHARED = residualization/_shared.py

WB_DATA = $(wildcard $(DATA)/*.csv)
WCDE_DATA = $(wildcard $(WCDE)/*.csv)
ALL_DATA = $(WB_DATA) $(WCDE_DATA)

.PHONY: all clean

all: verify analysis

# ── Verify scripts (produce checkin JSONs) ────────────────────────────

verify: \
	$(CHECKIN)/country_education.json \
	$(CHECKIN)/country_gdp.json \
	$(CHECKIN)/country_le_tfr.json \
	$(CHECKIN)/table4_crossings.json \
	$(CHECKIN)/china_cr.json \
	$(CHECKIN)/college_le_gradient.json \
	$(CHECKIN)/costa_rica_korea.json \
	$(CHECKIN)/figure2_betas.json \
	$(CHECKIN)/table_a1_cutoffs.json \
	$(CHECKIN)/kerala.json \
	$(CHECKIN)/edu_vs_gdp_predicts_le.json \
	$(CHECKIN)/regression_tables.json \
	$(CHECKIN)/u5mr_residual_by_year.json \
	$(CHECKIN)/regime_education_test.json \
	$(CHECKIN)/colonial_education_vs_institutions.json \
	$(CHECKIN)/iv_2sls_colonial.json \
	$(CHECKIN)/china_mean_yrs_vs_peers.json

# ── Analysis scripts (produce checkin JSONs) ──────────────────────────

analysis: \
	$(CHECKIN)/fig_beta_vs_baseline.json \
	$(CHECKIN)/beta_by_baseline_group.json \
	$(CHECKIN)/robustness_tests.json \
	$(CHECKIN)/education_outcomes.json \
	$(CHECKIN)/long_run_generational.json \
	$(CHECKIN)/table_1_main.json \
	$(CHECKIN)/table_a1_two_way_fe.json \
	$(CHECKIN)/education_vs_gdp_by_cutoff.json \
	$(CHECKIN)/beta_by_ceiling_cutoff.json \
	$(CHECKIN)/development_threshold_count.json \
	$(CHECKIN)/asian_financial_crisis.json \
	$(CHECKIN)/fig_a1_lag_decay.json \
	$(CHECKIN)/edu_vs_gdp_residualized.json \
	$(CHECKIN)/edu_vs_gdp_tfr_residualized.json \
	$(CHECKIN)/edu_vs_gdp_entry_ceiling.json \
	$(CHECKIN)/edu_vs_gdp_entry_threshold.json \
	$(CHECKIN)/edu_vs_gdp_by_level.json \
	$(CHECKIN)/edu_vs_gdp_child_edu_residualized.json \
	$(CHECKIN)/female_education_residualized.json \
	$(CHECKIN)/twoway_fe_all_outcomes.json \
	$(CHECKIN)/residual_by_outcome_year_all.json \
	$(CHECKIN)/u5mr_by_edu_level.json \
	$(CHECKIN)/lag_sensitivity.json \
	$(CHECKIN)/completion_vs_test_scores.json \
	$(CHECKIN)/threshold_robustness.json \
	$(CHECKIN)/happiness_education.json

# ── Rules ─────────────────────────────────────────────────────────────

$(CHECKIN)/country_education.json: cases/country_education.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/country_gdp.json: cases/country_gdp.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/country_le_tfr.json: cases/country_le_tfr.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/table4_crossings.json: cases/threshold_crossings.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/china_cr.json: cases/china_cultural_revolution.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/college_le_gradient.json: cases/college_le_gradient.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/costa_rica_korea.json: cases/costa_rica_korea.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/figure2_betas.json: cases/figure2_betas.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/table_a1_cutoffs.json: cases/threshold_cutoffs.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/kerala.json: cases/kerala.py $(DATA)/kerala_srs.csv
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_predicts_le.json: residualization/education_predicts_le.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/regression_tables.json: tables/regression_tables.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/u5mr_residual_by_year.json: robustness/u5mr_residual_by_year.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/regime_education_test.json: robustness/regime_education.py $(ALL_DATA) $(DATA)/p5v2018.xls
	$(PYTHON) $<

$(CHECKIN)/colonial_education_vs_institutions.json: robustness/colonial_vs_institutions.py $(ALL_DATA) $(DATA)/p5v2018.xls
	$(PYTHON) $<

$(CHECKIN)/iv_2sls_colonial.json: robustness/iv_2sls_colonial.py robustness/colonial_vs_institutions.py $(ALL_DATA) $(DATA)/p5v2018.xls
	$(PYTHON) $<

	$(PYTHON) $<

$(CHECKIN)/china_mean_yrs_vs_peers.json: cases/china_mean_years.py _shared.py $(ALL_DATA) ../wcde/data/raw/prop_both.csv
	$(PYTHON) $<

$(CHECKIN)/table_1_main.json: tables/table_1_main.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/table_a1_two_way_fe.json: tables/table_a1_two_way_fe.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/education_vs_gdp_by_cutoff.json: residualization/by_gdp_cutoff.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/beta_by_ceiling_cutoff.json: robustness/beta_by_ceiling_cutoff.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/development_threshold_count.json: cases/development_threshold_count.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/asian_financial_crisis.json: robustness/asian_financial_crisis.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/fig_a1_lag_decay.json: figures/fig_a1_lag_decay.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_residualized.json: residualization/education_vs_gdp.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_tfr_residualized.json: residualization/education_vs_tfr.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_entry_ceiling.json: residualization/by_entry_ceiling.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_entry_threshold.json: residualization/by_entry_threshold.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_by_level.json: residualization/by_education_level.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_child_edu_residualized.json: residualization/child_education.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/female_education_residualized.json: residualization/female_education.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/twoway_fe_all_outcomes.json: robustness/twoway_fe_all_outcomes.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/residual_by_outcome_year_all.json: robustness/residual_by_outcome_year.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/u5mr_by_edu_level.json: robustness/u5mr_by_edu_level.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/lag_sensitivity.json: robustness/lag_sensitivity.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/completion_vs_test_scores.json: robustness/completion_vs_test_scores.py _shared.py $(RESID_SHARED) $(ALL_DATA) $(DATA)/hlo_raw.csv
	$(PYTHON) $<

$(CHECKIN)/threshold_robustness.json: robustness/threshold_robustness.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/happiness_education.json: happiness_education.py $(DATA)/world_happiness_report.csv $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/fig_beta_vs_baseline.json: figures/fig_beta_vs_baseline.py $(WCDE_DATA)
	$(PYTHON) $<

$(CHECKIN)/beta_by_baseline_group.json: robustness/beta_by_baseline_group.py $(WCDE_DATA)
	$(PYTHON) $<

$(CHECKIN)/robustness_tests.json: robustness/robustness_tests.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/education_outcomes.json: wcde/education_outcomes.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/long_run_generational.json: wcde/long_run_generational.py _shared.py $(WCDE_DATA)
	$(PYTHON) $<

# ── Clean ─────────────────────────────────────────────────────────────

clean:
	rm -f $(CHECKIN)/*.json
