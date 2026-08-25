# DFM-PCA (`dfm_pca`)

- **Category:** main
- **Paper:** Probabilistic Modeling of Deep Features for Out-of-Distribution and Adversarial Detection — Ahuja et al., 2019
- **Official code:** https://github.com/MehrtashHarandi/DFM (tracker URL 404s; project page https://intellabs.github.io/dfm)
- **Fidelity verdict:** FAITHFUL — to the TS-OOD DFM-PCA target; an ADAPTATION relative to the cited Ahuja et al. 2019 original
- **Core idea:** Fit one PCA subspace per ID class on pre-logit features. Score a sample by the minimum feature reconstruction error (FRE) across class models — project into each class subspace, inverse-project, measure the residual. Large residual indicates OOD. Post-hoc, supervised.
- **Key parameters:** `n_components = 32`, capped at `min(32, feat_dim, n_class_samples − 1)`; per-class PCA on pre-logit features; score `min_c ‖x − x̂_c‖` (L2), higher = OOD.
- **Divergences from original / caveats:**
  - Faithful to TS-OOD's named "DFM-PCA" variant (per-class PCA, FRE, min over classes, pre-logit). But an adaptation vs Ahuja 2019, whose method is density-based (Gaussian/GMM likelihood, PCA only as preprocessing). FRE actually originates in later Intel Labs work (Ndiour et al., ICIP 2022 / BMVC 2023) — the thesis should cite both and not call this a reproduction of the 2019 paper.
  - `n_components=32` is an undocumented policy deviation: references pin explained *variance* (0.995 Ahuja, 0.97 anomalib), not a fixed count, so effective capacity drifts with feature dimension. In practice the per-class sample cap (~19 with 4 pseudo-classes × ~20 windows) binds, so 32 is rarely the operative value.
  - Latent min-over-classes bias: undersampled classes get fewer components and larger FRE, but the loader uses equal temporal bins so class sizes differ by ≤1 — bias is latent, not active.
  - L2 vs squared-L2 residual is √-monotone (AUROC-invariant). Real-data impact of the component policy not measurable (no cached features).
- **Where it runs:** `models/detectors/dfm_pca.py`
