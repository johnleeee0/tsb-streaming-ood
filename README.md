# When Confidence Lies — OOD Detection in Streaming Time Series

**Deep Feature Modeling versus Post-hoc Softmax Scores for Out-of-Distribution Detection in Streaming Time Series**

Stylianos Giannoulis · Aristotle University of Thessaloniki (AUTH), MSc in Data and Web Science · Supervisor: John Paparrizos · 2026

[![tests](https://img.shields.io/badge/tests-pytest-blue)](tests/) [![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Abstract
A reproducible study of out-of-distribution (OOD) detection for streaming time series. Every OOD
detector is first **audited for faithfulness** to its source paper/code; the faithful set is then
evaluated on the **TSB-StreamingAD** benchmark under a source-boundary, window-level protocol. The
central finding: **distance- and density-based detectors dominate post-hoc softmax detectors**, which
are frequently below chance under distribution shift — independently confirming, and extending to the
streaming setting, the central claim of the reference paper (Gungor et al., AAAI 2025).

This artifact ships **17 faithful production detectors** (the headline benchmark) plus **7 appendix
("class-D") detectors** that cannot be faithfully evaluated on this protocol and are reported
separately. The **per-method fidelity audit** in [`audit/`](audit/) is the supplementary material.

## Repository map
```
core/            base detector, registry, metrics, seed, losses, augmentations
data/            TSB/UCR loaders, aux-outlier corpus, download scripts (raw data not committed)
models/
  backbones/     ResNet / LSTM / Transformer
  detectors/     17 main detectors (one self-registering file each)
    class_d/     7 appendix detectors
runners/         run.py (entrypoint), pipeline.py, class_d.py, aggregate.py, evaluate.py, reproduce_all.py
configs/         YAML configs
viz/             plotting
tests/           pytest: smoke (all 24) + unit + determinism
docs/            method_cards/ (one per detector), reproducibility.md, datasets.md, protocol.md
audit/           per-method fidelity verdicts + fix plans (paper supplementary)
paper/           LaTeX thesis
results/         generated outputs (gitignored; tables committed as artifacts)
```

## Install
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
# torch is the CPU build; if needed:
# pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
```

## Data
Raw TSB-StreamingAD (U/M) and UCR/UEA archives are **not committed**. Fetch via `data/download/`,
then point the loaders at them with `--data-root <dir>` or `TSB_DATA_ROOT` (default
`C:\THESIS\benchmark1\datasets`). See [`docs/datasets.md`](docs/datasets.md).

## Quickstart (no data needed)
```bash
.venv/Scripts/python -m pytest tests -q        # 41 tests: all 24 detectors run + unit checks
```

## Reproduce
The 17-detector headline benchmark (seed 42, CPU):
```bash
.venv/Scripts/python runners/run.py --scale full --split U      # full ~183-dataset sweep
.venv/Scripts/python runners/aggregate.py                       # rebuild results/benchmark.csv
.venv/Scripts/python runners/evaluate.py                        # rankings, Friedman, tables, figures
```
Smaller runs:
```bash
.venv/Scripts/python runners/run.py --scale few --n-per-cell 6 --split U   # quick stratified sample
.venv/Scripts/python runners/run.py --scale one --dataset OOD_009          # a single dataset
```
The 7-detector appendix study (separate protocols):
```bash
.venv/Scripts/python runners/run.py --group class_d --scale full --split U
```
Everything at once:
```bash
.venv/Scripts/python runners/reproduce_all.py       # main + class_d -> aggregate -> evaluate -> tables/figures
```
All figures/tables in the paper regenerate from `results/`. See [`docs/reproducibility.md`](docs/reproducibility.md).

## Results
Headline rankings, the Friedman test, per-category/per-normalization dichotomy tables, and figures are
generated into `results/` by `runners/evaluate.py` (`results/findings.md`, `results/tables/*.tex`,
`results/figures/*.pdf`). They are regenerated from scratch by the reproduce commands above.

## Detector fidelity (the audit)
Each detector carries a verdict — **FAITHFUL / CORRECTED / ADAPTATION / NOT-THE-METHOD** — in
[`docs/method_cards/`](docs/method_cards/) (distilled) and [`audit/verification/`](audit/verification/)
(full). Auditing baselines for source-faithfulness is a deliberate contribution: most benchmark papers
do not.

## Testing
`pytest tests/` runs a smoke test (all 24 detectors instantiate + fit + score), unit tests for the
metrics and loaders, and a determinism check (same seed → identical scores). CI runs the smoke test on
push (`.github/workflows/ci.yml`).

## Cite
See [`CITATION.cff`](CITATION.cff).

## License
MIT — see [`LICENSE`](LICENSE).
