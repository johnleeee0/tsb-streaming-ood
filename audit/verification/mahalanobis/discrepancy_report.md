# Discrepancy Report — `mahalanobis_mds` (Mahalanobis Distance / MDS)

**Author:** Stylianos Giannoulis · Aristotle University of Thessaloniki, MSc Data and Web Science · Supervisor: John Paparrizos
**Method id:** `mahalanobis_mds` · **Citation:** Lee et al., 2018 (NeurIPS) — as adapted by Gungor et al., 2025 (TS-OOD)
**Local implementation:** `benchmark1/models/ood_methods/mahalanobis.py` (`MahalanobisDetector`)
**Reference:** `methods/mahalanobis_mds/reference/` (pokaxpoka/deep_Mahalanobis_detector); routine `lib_generation.py::sample_estimator`, `get_Mahalanobis_score`

> Note on target. The thesis reproduces the **TS-OOD adaptation** of MDS (Gungor2025), which
> states explicitly that for multivariate time series — lacking large pretrained backbones —
> the backbone is trained on ID classes only and a single pre-logit feature layer is used.
> Severities below are graded against **two** references: the original Lee et al. (2018) and
> the TS-OOD adaptation that this thesis actually targets.

## Component comparison

| Component | Finding | Severity (vs Lee2018 / vs Gungor2025) | Recommendation |
|---|---|---|---|
| Class-conditional means | Both compute per-class feature means. Matches. | — / — | None |
| Tied covariance | Reference fits a single pooled `EmpiricalCovariance` over class-centered features and uses `precision_`. Local fits `EmpiricalCovariance` over pooled features, adds `1e-6·I` ridge, then inverts (`pinv` fallback). Same estimator; local adds light regularisation for invertibility. | MINOR / MINOR | Keep ridge; note it aids stability on small TS training sets. |
| Scoring formula | Reference computes the Gaussian confidence `−½(x−μ)ᵀΣ⁻¹(x−μ)` and takes the **max** over classes (higher = more ID). Local computes the Mahalanobis **distance** `√((x−μ)ᵀΣ⁻¹(x−μ))` and takes the **min** over classes (higher = more OOD). These are monotone-equivalent in ranking (min distance ↔ max negative half-squared distance); the local orientation (higher = OOD) is the project convention. AUROC/AUPR-invariant. | MINOR / MINOR | Keep; document the √ vs squared and sign conventions. |
| Input pre-processing (FGSM perturbation) | Reference adds an adversarial-style input perturbation of magnitude `m` before scoring. Local applies **no** input perturbation. | MODERATE / — | Lee2018's full method includes perturbation; TS-OOD deliberately omits it. Omission is correct for the thesis target; mention as a documented simplification. |
| Multi-layer feature ensemble + logistic-regression detector | Reference extracts features at several depths and fits a logistic-regression weighting; the magnitude/layer are tuned. Local uses a **single pre-logit layer**, unweighted. | MODERATE / — | TS-OOD uses pre-logit features only ("obtained on average superior results using the pre-logit layer"). Local matches the target; note divergence from the original. |
| Label requirement | Local raises if `y_id` is absent (per-class means require labels). Consistent with the supervised class-conditional construction in both references. | — / — | None |
| Evaluation protocol | scikit-learn AUROC/AUPR; threshold-free. Matches. | — / — | None |

## Summary

The local `MahalanobisDetector` faithfully implements the Mahalanobis out-of-distribution
score in the form adopted by the TS-OOD evaluation it extends: class-conditional Gaussians
with a tied, pooled covariance estimated on in-distribution pre-logit features, scored by the
minimum Mahalanobis distance to the class means. Relative to the original method of Lee et al.
(2018) it omits two elements — the FGSM-style input perturbation and the multi-layer feature
ensemble with a learned logistic-regression combiner — but these omissions are precisely the
simplifications prescribed by the reference paper for the multivariate time-series setting,
where no large pretrained backbone is available and a single pre-logit layer is empirically
preferable. The remaining differences (a small ridge term for covariance invertibility and a
monotone change of the score's sign and square root) do not affect threshold-free detection
metrics. The implementation is therefore a faithful reproduction of the intended procedure and
no corrective `_enh` variant is required; the two documented simplifications relative to the
original are reported for the thesis appendix.
