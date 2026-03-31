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
	$(CHECKIN)/colonial_education_vs_institutions.json

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
	$(CHECKIN)/lag_sensitivity.json

# ── Rules ─────────────────────────────────────────────────────────────

$(CHECKIN)/country_education.json: verify_country_education.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/country_gdp.json: verify_country_gdp.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/country_le_tfr.json: verify_country_le_tfr.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/table4_crossings.json: verify_table4_crossings.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/china_cr.json: verify_china_cr.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/college_le_gradient.json: verify_college_le_gradient.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/costa_rica_korea.json: verify_costa_rica_korea.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/figure2_betas.json: verify_figure2_betas.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/table_a1_cutoffs.json: verify_table_a1_cutoffs.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/kerala.json: verify_kerala.py $(DATA)/kerala_srs.csv
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_predicts_le.json: edu_vs_gdp_predicts_le.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/regression_tables.json: regression_tables.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/u5mr_residual_by_year.json: u5mr_residual_by_year.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/regime_education_test.json: regime_education_test.py $(ALL_DATA) $(DATA)/p5v2018.xls
	$(PYTHON) $<

$(CHECKIN)/colonial_education_vs_institutions.json: colonial_education_vs_institutions.py $(ALL_DATA) $(DATA)/p5v2018.xls
	$(PYTHON) $<

$(CHECKIN)/table_1_main.json: table_1_main.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/table_a1_two_way_fe.json: table_a1_two_way_fe.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/education_vs_gdp_by_cutoff.json: education_vs_gdp_by_cutoff.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/beta_by_ceiling_cutoff.json: beta_by_ceiling_cutoff.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/development_threshold_count.json: development_threshold_count.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/asian_financial_crisis.json: asian_financial_crisis.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/fig_a1_lag_decay.json: fig_a1_lag_decay.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_residualized.json: edu_vs_gdp_residualized.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_tfr_residualized.json: edu_vs_gdp_tfr_residualized.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_entry_ceiling.json: edu_vs_gdp_entry_ceiling.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_entry_threshold.json: edu_vs_gdp_entry_threshold.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_by_level.json: edu_vs_gdp_by_level.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_child_edu_residualized.json: edu_vs_gdp_child_edu_residualized.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/female_education_residualized.json: female_education_residualized.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/twoway_fe_all_outcomes.json: twoway_fe_all_outcomes.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/residual_by_outcome_year_all.json: residual_by_outcome_year_all.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/u5mr_by_edu_level.json: u5mr_by_edu_level.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/lag_sensitivity.json: lag_sensitivity.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/fig_beta_vs_baseline.json: fig_beta_vs_baseline.py $(WCDE_DATA)
	$(PYTHON) $<

$(CHECKIN)/beta_by_baseline_group.json: beta_by_baseline_group.py $(WCDE_DATA)
	$(PYTHON) $<

$(CHECKIN)/robustness_tests.json: robustness_tests.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/education_outcomes.json: ../wcde/scripts/07_education_outcomes.py $(ALL_DATA)
	$(PYTHON) ../wcde/scripts/07_education_outcomes.py

$(CHECKIN)/long_run_generational.json: ../wcde/scripts/04b_long_run_generational.py $(WCDE_DATA)
	$(PYTHON) ../wcde/scripts/04b_long_run_generational.py

# ── Clean ─────────────────────────────────────────────────────────────

clean:
	rm -f $(CHECKIN)/*.json
