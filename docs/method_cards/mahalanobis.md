# Mahalanobis Distance Detector / MDS (`mahalanobis`)

- **Category:** main
- **Paper:** A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks — Lee et al., 2018
- **Official code:** https://github.com/pokaxpoka/deep_Mahalanobis_detector
- **Fidelity verdict:** CORRECTED (fix applied 2026-08-20; now FAITHFUL) — within-class scatter covariance restored
- **Core idea:** Fit class-conditional Gaussians on pre-logit features with a single shared (tied) covariance, and score a sample by the minimum Mahalanobis distance to any ID class mean. Larger distance indicates OOD. Post-hoc, supervised (requires ID class labels).
- **Key parameters:** Single pre-logit feature layer; tied within-class covariance with `1e-6·I` ridge; score `min_c √((x−μ_c)ᵀ Σ_W⁻¹ (x−μ_c))`, higher = OOD.
- **Divergences from original / caveats:**
  - Original defect (now fixed): the tied covariance was computed as the **total** covariance `Σ_T = Σ_W + Σ_B` (raw features pooled, centered on the global mean) instead of the within-class scatter `Σ_W`. This is not rank-preserving and systematically degrades the score. The 3-line fix now class-centers features before pooling, matching Lee et al. Eq. (1), the official `sample_estimator`, and TS-OOD's explicit "tied covariance".
  - Documented simplifications, confirmed consistent with the TS-OOD target: single pre-logit layer (best per TS-OOD), no FGSM input perturbation, no multi-layer logistic-regression ensemble.
  - Fix changes every saved MDS score; `experiments/*/*/mahalanobis/` results must be regenerated (verified only via smoke test, not full benchmark). Note `driftlens` ties MDS at ρ≈0.999 — regenerate together.
  - Real-data magnitude of the old defect was not measurable (no cached features); synthetic tests show a small, consistently negative AUROC penalty growing with class separation.
- **Where it runs:** `models/detectors/mahalanobis.py`
