# Cross-Validation Summary

**Author:** Stylianos Giannoulis · Aristotle University of Thessaloniki, MSc Data and Web Science · Supervisor: John Paparrizos
**Date:** 2026-06-24 · **Agent:** Cross-Validator (Agent 1)

## Environment

- Python 3.13 (isolated venv `C:\THESIS\.venv`), torch 2.12.1+cpu, numpy 2.5.0, scipy 1.18.0, scikit-learn 1.9.0.
- Synthetic validation: seed=42; 200 ID normal series in 4 pseudo-classes (faint class sinusoids), `ResNet1D` backbone trained 25 epochs CE; 50 off-manifold uniform OOD series. Assertions: forward pass, finite 1-D score of shape (250,), AUROC > 0.5. Training-based detectors used reduced epochs for the smoke test only.
- The three family-representative methods (msp, mahalanobis_mds, dfm) additionally had reference repositories cloned and structurally diffed (see their `discrepancy_report.md`).

## Results — all 23 detectors

| # | Method | registry_id | Synthetic AUROC | Status | Note |
|---|---|---|---|---|---|
| 1 | MSP | msp | 0.9739 | PASS | reference-diffed; faithful |
| 2 | ODIN | odin | 0.9738 | PASS | |
| 3 | ReAct | react | 0.9751 | PASS | |
| 4 | DICE | dice | 0.8258 | PASS | |
| 5 | GradNorm | gradnorm | 0.9635 | PASS | |
| 6 | SCALE | scale | 0.9409 | PASS | |
| 7 | Mahalanobis (MDS) | mahalanobis_mds | 1.0000 | PASS | reference-diffed; faithful to TS-OOD adaptation |
| 8 | DFM-PCA | dfm | 1.0000 | PASS | reference-diffed; per-class DFM-PCA |
| 9 | Outlier Exposure | outlier_exposure | 0.9681 | PASS | |
| 10 | DivOE | divoe | 0.9570 | PASS | |
| 11 | CODiT | codit | 0.5944 | PASS | conformal; weaker on synthetic noise |
| 12 | SRS | srs | 1.0000 | PASS | |
| 13 | DiffAD | diffad | 0.9459 | PASS | |
| 14 | DIVERSIFY | diversify | 0.7489 | PASS | |
| 15 | InvAD | invad | 0.9265 | PASS | |
| 16 | M2N2 | m2n2 | 0.0633 | **FAIL (inverted)** | score anti-correlated; energy/recon inversion |
| 17 | TD-IVDM | tdivdm | 1.0000 | PASS | |
| 18 | CatSight | catsight | 0.7896 | PASS | |
| 19 | AE-ADWIN-LSTM | ae_adwin_lstm | 0.9394 | PASS | |
| 20 | DiMMAD | dimmad | 0.9993 | PASS | |
| 21 | DEEDEE | deedee | 0.9818 | PASS | |
| 22 | DiverseMix | diversemix | 0.1513 | **FAIL (inverted)** | energy-head inversion (cf. THESIS_FINDINGS) |
| 23 | DriftLens | driftlens | 0.9999 | PASS | |

**Summary: 21 / 23 PASS.** No CRITICAL/MODERATE discrepancies for the three reference-diffed methods; no `_enh` variants created.

## The two failures are inversions, not crashes

`m2n2` (0.063) and `diversemix` (0.151) complete the forward pass and produce finite, well-separated
scores — but **anti-correlated** with the OOD label (|AUROC − 0.5| ≈ 0.44, 0.35). This is the
energy/reconstruction **score-orientation inversion** already documented for these methods in
`benchmark1/THESIS_FINDINGS.md` (the "softmax/energy overconfidence dichotomy"): under some
distributional regimes the energy head assigns *lower* scores to off-manifold inputs. Because the
correct orientation is regime-dependent, we do **not** hard-code a sign flip; instead we exclude
these two from the headline experiment sweep and report the inversion itself as a finding
(it corroborates the dichotomy). They remain available for the directional-sensitivity analysis.

## Gate status for Agent 2

**OPEN for the 21 PASS detectors.** The experiment sweep proceeds on these; m2n2 and diversemix are
tracked separately as inversion cases.

## Per-method artifacts

`methods/<registry_id>/validation_status.json` for all 23; `discrepancy_report.md` + read-only
`reference/` clones for msp, mahalanobis_mds, dfm; batch log `methods/_validation/summary.json`.
