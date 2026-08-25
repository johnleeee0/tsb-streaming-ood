# Reproducibility

Every number and figure in the thesis is regenerated from `results/` by the runners — none are
hand-entered.

## Determinism
- Global seed **42** (`--seed`, default in `core/seed.py` / `runners/pipeline.py`).
- CPU execution; single ResNet backbone trained per dataset (CE on temporal pseudo-classes).
- Source-boundary split, window-level OOD labelling, seed-fixed stratified sampling.
- Verified: `tests/test_determinism.py` (same seed → identical scores) and a bit-identical
  migration parity check (old vs new codebase, Δ = 0 across all 24 detectors).

## Environment
- Python 3.14, dependencies pinned in `requirements.txt` (CPU `torch==2.12.1+cpu`).
- No GPU required. The full sweep is CPU-bound (multi-hour to multi-day depending on coverage).

## Commands
```bash
# quick sanity (no data)
python -m pytest tests -q

# main headline benchmark (17 detectors)
python runners/run.py --scale full --split U
python runners/aggregate.py
python runners/evaluate.py

# appendix study (7 class-D detectors, separate protocols)
python runners/run.py --group class_d --scale full --split U

# everything
python runners/reproduce_all.py
```
Runs are **resumable**: a completed `(dataset, method)` is skipped, so the sweep can span sessions;
raise `--n-per-cell` to expand coverage.

## Hardware note
The prior full-budget 183-dataset run was GPU; this artifact reproduces on CPU (slower). Record your
machine + wall-clock here when you run it.
