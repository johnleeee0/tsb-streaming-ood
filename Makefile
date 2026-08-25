# THESIS_FINAL — convenience targets.
# On Windows without `make`, run the underlying commands directly (see README).
PY ?= .venv/Scripts/python

.PHONY: install smoke test reproduce figures clean

install:
	$(PY) -m pip install -r requirements.txt

smoke:
	$(PY) -m pytest tests/test_smoke.py -q

test:
	$(PY) -m pytest tests -q

reproduce:
	$(PY) runners/reproduce_all.py

figures:
	$(PY) runners/evaluate.py
	$(PY) runners/appendix_figures.py

clean:
	rm -rf results/tsb results/benchmark.csv results/class_d_*.csv results/findings.md results/figures
