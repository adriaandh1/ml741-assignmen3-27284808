PYTHON := .venv/bin/python
SMOKE_OUTPUT ?= results-smoke
REPORT := 27284808RW741assignment3.pdf

.PHONY: all setup test lint smoke data preflight final-runs analysis sensitivity figures report clean

all: test report

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.lock.txt
	$(PYTHON) -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src scripts tests report/build_figures.py

smoke:
	$(PYTHON) -m ml741_assignment3.cli smoke --config configs/smoke.yaml --output $(SMOKE_OUTPUT)

data:
	$(PYTHON) -c "from pathlib import Path; from ml741_assignment3.datasets import prepare_all; prepare_all(Path.cwd())"

preflight:
	$(PYTHON) scripts/run_final.py

final-runs:
	$(PYTHON) scripts/run_final.py --execute

analysis:
	$(PYTHON) scripts/collect_final.py
	$(PYTHON) scripts/analyse_final.py

sensitivity:
	$(PYTHON) scripts/run_controller_sensitivity.py

figures:
	$(MAKE) -C report figures

report:
	$(MAKE) -C report check
	$(MAKE) -C report
	cp output/pdf/$(REPORT) $(REPORT)

clean:
	$(MAKE) -C report clean
