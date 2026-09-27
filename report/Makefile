REPORT=27284808RW741assignment3
.DEFAULT_GOAL := all

.PHONY: all check clean figures

figures:
	../.venv/bin/python build_figures.py

all:
	mkdir -p ../output/pdf
	latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=$(REPORT) main.tex
	cp $(REPORT).pdf ../output/pdf/$(REPORT).pdf

check:
	python3 check_writing_rules.py main.tex
	chktex -q -n 1 -n 8 -n 13 -n 24 -n 38 main.tex

clean:
	latexmk -C -jobname=$(REPORT) main.tex
