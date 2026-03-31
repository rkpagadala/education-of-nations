FIGS = fig_lag_comparison.png fig_beta_vs_baseline.png \
       fig_cumulative_developed.png fig_residualization.png

all: education_of_humanity.pdf education_of_nations.pdf cies_handout.pdf

education_of_humanity.pdf: education_of_humanity.tex $(FIGS)
	xelatex -interaction=nonstopmode $<
	xelatex -interaction=nonstopmode $<
	rm -f education_of_humanity.aux education_of_humanity.log education_of_humanity.out

education_of_nations.pdf: education_of_nations.tex $(FIGS)
	xelatex -interaction=nonstopmode $<
	xelatex -interaction=nonstopmode $<
	rm -f *.aux *.log *.out *.toc

cies_handout.pdf: cies_handout.tex
	xelatex cies_handout.tex
	rm -f cies_handout.aux cies_handout.log cies_handout.out

clean:
	rm -f *.pdf *.aux *.log *.out *.toc
