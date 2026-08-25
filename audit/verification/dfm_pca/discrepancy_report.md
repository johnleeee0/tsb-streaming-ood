# Discrepancy Report — `dfm` (Deep Feature Modeling, DFM-PCA)

**Author:** Stylianos Giannoulis · Aristotle University of Thessaloniki, MSc Data and Web Science · Supervisor: John Paparrizos
**Method id:** `dfm` · **Citation:** Ahuja et al., 2019 (arXiv:1909.11786) — as adapted (DFM-PCA) by Gungor et al., 2025 (TS-OOD)
**Local implementation:** `benchmark1/models/ood_methods/dfm_pca.py` (`DFMPcaDetector`)
**Reference:** `methods/dfm/reference/src/anomalib/models/image/dfm/torch_model.py` (`DFMModel`, `SingleClassGaussian`)

> Note on target. The thesis reproduces the **DFM-PCA** variant as defined in TS-OOD
> (Gungor2025), which models the in-distribution feature manifold **per ID class** and scores
> by feature reconstruction error (FRE). The anomalib reference implements a closely related
> but **global** (single-model) DFM with both FRE and NLL scoring. Severities are graded
> against the TS-OOD target.

## Component comparison

| Component | Finding | Severity | Recommendation |
|---|---|---|---|
| Feature source | Both score on deep features from a pre-trained/ID-trained backbone (local: pre-logit layer; reference: a named `timm` layer with avg-pooling). TS-OOD specifies pre-logit features. Matches the target. | — | None |
| Modeling granularity | Local fits **one PCA per ID class** and takes the **minimum** reconstruction error across classes. Reference fits a **single global PCA** over all ID features. TS-OOD's DFM-PCA is explicitly **per ID class** ("project the features of each ID class … via PCA"). Local matches the target; reference is the global variant. | MINOR | Keep per-class; note the global variant as an ablation option. |
| Scoring function (FRE) | Both use feature reconstruction error: project to PCA subspace, inverse-transform, measure residual. Reference uses squared L2 (`Σ(x−x̂)²`); local uses the L2 norm (`‖x−x̂‖`). Monotone-equivalent → AUROC/AUPR-invariant. | MINOR | Keep; optionally square to match reference exactly. |
| NLL scoring branch | Reference also offers a Gaussian NLL score (`SingleClassGaussian`, SVD-based). Local implements the FRE branch only (the DFM-PCA reported by the core paper). | MINOR | Keep FRE; NLL is a documented alternative for future work. |
| Number of components | Reference selects components by explained-variance ratio (`n_comps=0.97`). Local uses a fixed count (`n_components=32`), capped at `min(32, feat_dim, n_class_samples−1)`. Both are valid component-selection policies; this is a hyperparameter, not a procedural error. | MINOR | Keep fixed count for comparability; report the value. |
| Per-class sample sufficiency | Local guards classes with `<2` samples (PCA needs ≥2). Sensible for small TS training sets. | — | None |
| Label requirement | Local requires `y_id` (per-class modeling). Consistent with the per-class DFM-PCA definition. | — | None |
| Evaluation protocol | scikit-learn AUROC/AUPR; threshold-free. Matches. | — | None |

## Summary

The local `DFMPcaDetector` is a faithful implementation of the per-class DFM-PCA detector as
defined by the TS-OOD evaluation it extends: it fits an independent principal-component model
to the in-distribution features of each class and scores a test sample by its smallest feature
reconstruction error across those models. The anomalib reference realises a closely related but
deliberately different design — a single global feature model offering both reconstruction-error
and Gaussian negative-log-likelihood scores — so the principal structural difference
(per-class versus global modeling) reflects a documented variant choice rather than an
implementation defect, and the local choice is the one prescribed by the core paper. The
residual differences (L2 norm versus squared L2, fixed component count versus variance-ratio
selection) are either monotone-invariant under threshold-free metrics or standard
hyperparameter settings. No corrective `_enh` variant is required; the component-count policy
and the per-class design are recorded for the thesis appendix.
