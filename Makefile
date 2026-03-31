# Makefile — top level
#
# make          — fast verify + build paper
# make verify   — check paper numbers against existing JSONs (~2 sec)
# make scripts  — rebuild all checkin JSONs (only if script/data changed)
# make paper    — build PDFs
# make full     — rebuild JSONs + verify + paper (the full pipeline)
# make clean    — remove generated files
# make sync-cause — push to public repo

PYTHON = python3
PAPER_TEX = paper/education_of_nations.tex
VERIFY_STAMP = checkin/.verified
CAUSE_REPO = /tmp/education-of-nations

.PHONY: all verify scripts paper full clean sync-cause

all: verify paper

# Fast verify: check paper numbers against existing JSONs (~2 sec)
verify: $(VERIFY_STAMP)

$(VERIFY_STAMP): checkin/*.json scripts/verify_paper_numbers.py $(PAPER_TEX)
	$(PYTHON) scripts/verify_paper_numbers.py --fast
	@touch $@

# Build all checkin JSONs (only reruns changed scripts)
scripts:
	cd scripts && $(MAKE)

# Build paper PDFs
paper:
	cd paper && $(MAKE)

# Full pipeline: rebuild JSONs + verify + paper
full:
	cd scripts && $(MAKE)
	rm -f $(VERIFY_STAMP)
	$(MAKE) verify paper

