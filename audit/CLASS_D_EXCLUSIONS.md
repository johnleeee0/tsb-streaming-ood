# Class-D Exclusions — detectors not faithfully reproducible on this protocol

**Date:** 2026-08-21 · MSc Thesis, S. Giannoulis (AUTH), supervisor J. Paparrizos

The Pass-1 fidelity audit (`methods/<id>/VERIFICATION.md`) found 7 detectors that **cannot be reproduced
faithfully** under the benchmark's protocol — shuffled per-window evaluation, a frozen ResNet backbone,
and no auxiliary outlier corpus. Rather than ship an unfaithful implementation under the original name,
these are **excluded from the production benchmark** (`experiments/tsb_benchmark.py`) and reported as a
"not faithfully reproducible on this protocol" note. This is the honest scientific choice: forcing them
to run would report numbers that are not the methods the papers describe.

| # | method | audit verdict | why it cannot be faithful here | disposition |
|---|---|---|---|---|
| 1 | outlier_exposure | NOT-THE-METHOD | OE *trains* the classifier with cross-entropy-to-uniform on an **auxiliary outlier dataset**; the protocol has no such corpus and the backbone is frozen | folded into the clean **Energy (EBO)** row |
| 2 | divoe | NOT-THE-METHOD | needs auxiliary outliers **+ PGD synthesis + training**; none available | excluded (redundant with Energy) |
| 3 | diversemix | NOT-THE-METHOD | needs a real auxiliary outlier set + end-to-end training; at chance in either orientation on real data | excluded; reported as a negative result |
| 4 | driftlens | ADAPTATION | official score is **window/batch-level** Fréchet (Wasserstein-2) drift; per-window shuffled eval makes it a per-sample PCA-Mahalanobis (ρ≈0.999 with mahalanobis) | excluded to avoid a duplicate top-rank tie |
| 5 | tdivdm | ADAPTATION | multi-scale time-division / variable-density pillars need ordered multi-scale windows; no public code; our version is generic KDE density | excluded (would be "KDE-density", not TD-IVDM) |
| 6 | ae_adwin_lstm | NOT-THE-METHOD | LSTM prediction + ADWIN change detection require an **ordered stream**; the eval shuffles windows, so the temporal components are inert | excluded |
| 7 | diversify | ADAPTATION | the original is adversarial **representation learning** that retrains the feature extractor and **defines no OOD score**; incompatible with a frozen backbone | excluded |

## What replaced them
- **Energy (EBO)** is retained as a clean, faithful baseline (the `outlier_exposure` energy path, labelled
  `energy`) — this is the legitimate, training-free detector those three augmented-training methods collapse to.

## Effect on the benchmark
Excluding these 7 removes the artificial Mahalanobis/DriftLens tie at the top and drops three redundant
energy rows and two protocol-incompatible temporal methods. The production set is the 17 detectors in
`experiments/tsb_benchmark.py::method_set` — each faithful or a corrected/faithful variant, per its
`methods/<id>/VERIFICATION.md`.
