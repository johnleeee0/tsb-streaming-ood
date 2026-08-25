# Production Test Set & Test Plan (B)

**Author:** Stylianos Giannoulis · AUTH MSc Data and Web Science · Supervisor: John Paparrizos
**Date:** 2026-06-25

Defines exactly which detectors enter the production benchmark and where/how the test runs, based on
the full original-code verification (`registry/METHOD_VERIFICATION_FULL.md`).

## Where the test runs
- **Runner:** `experiments/tsb_benchmark.py` (resumable, stratified, per-method incremental writes, dataset-level error isolation).
- **Target datasets:** TSB-StreamingAD-U and TSB-StreamingAD-M (`benchmark1/datasets/TSB-StreamingAD-{U,M}`), stratified across DRIFT / OOD / STABLE; coverage set by `TSB_N_PER_CELL`.
- **Protocol:** source-boundary split, window-level OOD, ResNet backbone (CE on temporal pseudo-classes), 50/50 val/test, **seed 42**, **CPU**, 40 epochs.
- **Outputs:** `results/tsb_benchmark.csv` → `experiments/aggregate_tsb.py` (rebuild from disk) → `results/evaluate_tsb.py` (rankings, per-category/normalization, Friedman, figures) → `results/tsb_findings.md`.

## Tier A — PROD-CORE (faithful or corrected; the reportable benchmark)
These are original-code-aligned and are what conclusions will be drawn from:

`msp`, `odin`, `energy` (EBO), `mahalanobis`, `dfm_pca`, `srs`,
`react_enh`, `dice_enh`, `scale_enh`, `gradnorm_enh`, `dimmad_enh`,
`driftlens` (reported as a PCA-Mahalanobis / feature-drift variant).

Note: `energy` (EBO) is run as a clean energy detector (the `outlier_exposure` energy path), labelled
**Energy**, not "Outlier Exposure".

## Tier B — PROD-EXTENDED (adaptations; reported separately, labelled "-lite/adaptation")
`codit`, `diversify`, `invad`, `m2n2`, `deedee`, `tdivdm`, `catsight`, `ae_adwin_lstm`.
Included for breadth but explicitly described as adaptations, not faithful reproductions.

## Excluded from the headline benchmark (structural issues)
`outlier_exposure` (no OE training → folded into Tier A as **Energy**), `divoe`, `diversemix`
(orientation regime-dependent), `diffad` (input-independent reverse). Reported only as a "not
faithfully reproducible on this protocol" note.

## What "start testing" will do
On your go, I will:
1. Set `tsb_benchmark.py`'s method set to **Tier A (+ Tier B, labelled)**, drop the excluded ones.
2. Choose coverage (`TSB_N_PER_CELL`) per your instruction (e.g. 6 ≈ 36 datasets fast; 20 ≈ 120; 50 ≈ near-full per split).
3. Run resumably on TSB-U and TSB-M, then `aggregate_tsb.py` + `evaluate_tsb.py`, and report rankings/findings for the **verified** set.
