# Full Method Verification — All Implemented Detectors vs Original Author Code

**Author:** Stylianos Giannoulis · Aristotle University of Thessaloniki, MSc Data and Web Science · Supervisor: John Paparrizos
**Date:** 2026-06-25

Every implemented detector (`benchmark1/models/ood_methods/`) checked against the original authors'
code (live-fetched or cloned official repo), or the paper where no public code exists. "Faithful" =
matches the original up to metric-invariant conventions. Per-method deep reports exist for MSP, ODIN,
EBO, ReAct under `methods/<id>/*_VALIDATION.md`.

**No AI/Claude-attribution comments are present in any implemented method file** (verified by scan).

## Legend
- ✅ FAITHFUL — prod-ready as the original method.
- 🔧 CORRECTED — original deviates; a validated `_enh` variant reproduces the paper; use `_enh` in prod.
- 🟡 ADAPTATION — a "-lite" reinterpretation, usable but must be labelled as an adaptation, not the original.
- ❌ NOT THE METHOD — a defining component is absent/wrong; exclude or relabel in prod.

---

## Post-hoc softmax / logit / activation / gradient

### MSP — ✅ FAITHFUL  (hendrycks/error-detection, live)
Official: `T.max(softmax(logits))`. Ours: `1 − max softmax(logits/T)`, T=1. Equal up to additive constant → AUROC/AUPR/FPR95 identical.

### ODIN — ✅ FAITHFUL (minor)  (facebookresearch/odin, live)
Official: temperature scaling + `x − ε·sign(∇CE(logits/T, ŷ))` + max softmax. Ours: identical procedure. Minor: no per-channel gradient std-norm (≈ identity on z-normalised inputs); fixed ε=0.001.

### EBO / Energy — ✅ FAITHFUL  (wetliu/energy_ood, live)
Official: `E = −T·logsumexp(logits/T)`. Ours (`base._energy`): `−logsumexp(logits/T)` — identical at T=1, positive-scale (metric-invariant) otherwise. Realised in the benchmark via `outlier_exposure(score_type="energy")`, `divoe`, `scale_enh`.

### ReAct — ✅ FAITHFUL clipping  (deeplearning-wisc/react, cloned)
Official: clip penultimate activations at the 90th ID percentile (`compute_threshold.py`), then a base score from `score.py` (msp/odin/**energy**; headline = energy). Ours: clipping faithful; original `react.py` = ReAct+MSP (supported config); `react_enh` = ReAct+Energy (headline). 🔧 use `react_enh` for the paper-primary score.

### DICE — 🔧 CORRECTED  (deeplearning-wisc/dice, cloned)
Official: sparsify the FC weights using a **precomputed mean-contribution mask** over ID data, then **energy**. Original `dice.py`: per-sample top-k of `feature⊙weight`, **sums absolute values** (drops sign), + MSP — NOT faithful. `dice_enh`: signed top-k contribution sum + energy (synthetic 0.83→0.96). Use `dice_enh`.

### SCALE — 🔧 CORRECTED  (kai422/SCALE, cloned)
Official (`scale_net.py`): scale **penultimate activations** by `exp(s1/s2)` (s2 = top-k pruned sum), then energy. Original `scale.py`: z-standardises **logits** + energy — wrong layer/operation. `scale_enh`: activation scaling + energy. Use `scale_enh`.

### GradNorm — 🔧 CORRECTED  (deeplearning-wisc/gradnorm_ood, cloned, test_ood.py:124–145)
Official: **L1 norm of the last-layer-weight gradient** of **KL-to-uniform**; higher = ID. Original `gradnorm.py`: **L2 norm of the input gradient** of CE-to-predicted-label; higher = OOD — wrong variable, loss, norm, and orientation. `gradnorm_enh`: faithful reimplementation. Use `gradnorm_enh`.

## Feature distance / density

### Mahalanobis (MDS) — ✅ FAITHFUL  (pokaxpoka/deep_Mahalanobis_detector, cloned)
Official: class-conditional Gaussians, tied covariance, Mahalanobis distance. Ours: faithful to the TS-OOD adaptation (tied covariance on pre-logit features, min class distance); omits the original FGSM perturbation + multi-layer ensemble exactly as the core paper prescribes. Prod-ready.

### DFM-PCA — ✅ FAITHFUL  (Ahuja2019 / anomalib, cloned)
Official: model ID feature manifold, feature reconstruction error. Ours: per-class PCA reconstruction error = the TS-OOD DFM-PCA. Prod-ready.

### DiMMAD — 🔧 CORRECTED  (sidchaini/distclassipy, cloned)
Multi-metric distance ensemble. Original includes **binary set metrics (Hamming/Jaccard/Dice) on continuous features** (ill-defined). `dimmad_enh`: continuous metrics only. Use `dimmad_enh`.

## Augmented training

### Outlier Exposure — ❌ NOT THE METHOD  (hendrycks/outlier-exposure, cloned)
Official `oe_tune.py`: **trains** the classifier with the OE loss (cross-entropy to uniform on an auxiliary outlier set). Ours: **no training** — just the energy score on the ID backbone. The defining mechanism is absent → it is an energy baseline. Relabel as "Energy (EBO)" or implement OE training with an auxiliary outlier source.

### DivOE — ❌ NOT THE METHOD  (ZFancy/DivOE, cloned)
Official: synthesise diverse outliers + OE training. Ours: energy on mean-centred logits — no synthesis/training. Relabel/exclude.

### DiverseMix — ❌ orientation regime-dependent  (Diversification2024; codebase hybrid)
Trains an energy head with diversity mixup (faithful in spirit), but the score orientation is fitted post-hoc. `diversemix_enh` negates to match the training objective (synthetic 0.19→0.81) **but reverses on real data (0.633→0.367)** — orientation is regime-dependent. Report as a negative result; do not treat either orientation as canonical.

## Time-series-specific

### SRS — ✅ FAITHFUL (minor)  (tahabelkhouja/SRS, cloned; STL.py + CVAE_Keras.py confirmed)
Official: STL decomposition + conditional VAE + seasonal-ratio score. Ours: PyTorch port with STL + CVAE; uses the signal neg-ELBO rather than the full signal/residual ratio (documented stability choice). The one genuinely faithful, mechanistically distinct TS method. Prod-ready.

### CODiT — 🟡 ADAPTATION  (kaustubhsridhar/time-series-OOD, cloned)
Conformal temporal non-conformity. Ours: transform-classification + conformal p-values + Fisher combination — structurally faithful, but the score **orientation is asserted** (reverse of standard conformal). Use with caution; report orientation caveat.

### DIVERSIFY — 🟡 ADAPTATION  (microsoft/robustlearn)
Official: adversarial **representation learning** (retrains the feature extractor across latent domains). Ours: **k-means distance to centroids** on frozen features — a different mechanism. Label as a feature-space distance adaptation.

### InvAD — 🟡 ADAPTATION  (fly-orange/InvAD, cloned)
Official: invertible dual-branch separation of seen/unseen anomalies. Ours: affine-coupling network whose invertibility makes `reconstruct(decompose(x)) ≈ x`, so the reconstruction-error signal ≈ 0 and the score reduces to MSP on `z_id`. Label as adaptation (effectively MSP).

### M2N2 — 🟡 ADAPTATION  (carrtesy/M2N2, cloned)
Official: test-time-adaptive AE on raw series. Ours: AE on frozen features + EMA detrend (test-time adaptation off by default); sequential EMA makes scores mildly order-dependent. Reasonable feature-AE adaptation; label as such.

### DEEDEE — 🟡 ADAPTATION  (no public code; arXiv 2510.21638)
Official: trajectory statistics on temporal sequences. Ours: treats **adjacent feature dimensions as temporal neighbours** (no theoretical basis) + per-dim isolation forests; O(N·d) slow. Label as adaptation; weakest fidelity.

### TD-IVDM — 🟡 ADAPTATION  (no public code; Neurocomputing 2025)
Official: multi-scale time-division/variable-density drift. Ours: Gaussian KDE on PCA features — a generic density detector. Relabel as "KDE density".

### CatSight — 🟡 ADAPTATION  (no public code)
CSP via generalized eigenproblem — structurally implemented; score uses a **negated** distance (orientation asserted). Label as adaptation with orientation caveat.

### AE-ADWIN-LSTM — 🟡 ADAPTATION  (no public code; IEEE APCI 2025)
Official: AE + ADWIN + LSTM on an **ordered** stream. Ours runs these on **shuffled** evaluation windows where the LSTM/ADWIN temporal components are meaningless; score negated. Design mismatch; label as adaptation.

## Reconstruction / drift

### DiffAD — ❌ NOT THE METHOD  (ChunjingXiao/DiffAD, cloned)
Official: imputation by **partially noising the input** and denoising back (reconstruction depends on the input). Ours: reverse process starts from **pure noise**, independent of the input → measures distance to a random ID sample; score negated. Exclude or reimplement input-conditioned reverse.

### DriftLens — 🟡 ADAPTATION  (grecosalvatore/drift-lens, cloned)
Official: **batch/window-level** Fréchet (Wasserstein-2) drift. Ours: per-sample **squared Mahalanobis in PCA space** (= PCA-whitened Mahalanobis). Strong and prod-usable, but report it as a PCA-Mahalanobis variant, not batch-level DriftLens (this is also why it ties Mahalanobis in the rankings).

---

## Summary tally (23 implemented detectors)
- ✅ FAITHFUL (prod-ready as original): MSP, ODIN, EBO/Energy, ReAct(clipping), Mahalanobis, DFM-PCA, SRS — **7**.
- 🔧 CORRECTED (use `_enh`): GradNorm, DICE, SCALE, DiMMAD, ReAct(energy) — **5** (`gradnorm_enh`, `dice_enh`, `scale_enh`, `dimmad_enh`, `react_enh`).
- 🟡 ADAPTATION (label as -lite): CODiT, DIVERSIFY, InvAD, M2N2, DEEDEE, TD-IVDM, CatSight, AE-ADWIN-LSTM, DriftLens — **9**.
- ❌ NOT THE METHOD (exclude/relabel): Outlier Exposure, DivOE, DiverseMix, DiffAD — **4**.
