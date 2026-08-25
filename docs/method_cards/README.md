# Method Cards

One-page fidelity cards for every OOD/drift detector in the benchmark, distilled from the
per-method audits in `audit/verification/<id>/VERIFICATION.md`. Each card records the paper,
official code, the fidelity verdict (FAITHFUL / CORRECTED / ADAPTATION / NOT-THE-METHOD), core
idea, production parameters, the honest divergence list, and where the detector runs.

**Fidelity verdict** = the first-line heading of the method's `VERIFICATION.md`. **Main** methods run
via `models/detectors/<id>.py`; **class-D appendix** methods run via `models/detectors/class_d/<id>.py`
with `--group class_d` and never enter the 17-method frozen-backbone leaderboard.

## Main

| id | name | category | verdict | one-line |
|---|---|---|---|---|
| `msp` | Maximum Softmax Probability | main | FAITHFUL | Max softmax probability as confidence; low = OOD. |
| `odin` | ODIN | main | FAITHFUL | Temperature scaling + input perturbation to sharpen ID/OOD softmax separation. |
| `energy` | Energy-based OOD Detection (EBO) | main | FAITHFUL | Free energy `−logsumexp` over logits; higher = OOD. |
| `react` | ReAct: Rectified Activations | main | CORRECTED | Clip penultimate activations at a percentile, then energy score (`react_enh`). |
| `mahalanobis` | Mahalanobis Distance (MDS) | main | CORRECTED | Min Mahalanobis distance to class Gaussians with tied within-class covariance. |
| `dfm_pca` | DFM-PCA | main | FAITHFUL | Per-class PCA; min feature reconstruction error across classes. |
| `srs` | Seasonal Ratio Scoring (SRS) | main | FAITHFUL | STL split + twin CVAEs; ratio of signal to residual neg-ELBO. |
| `dice` | DICE: Directed Sparsification | main | FAITHFUL | Static ID-mean weight mask sparsifies the head, then energy (`dice_enh`). |
| `scale` | SCALE: Activation Scaling | main | CORRECTED | Rescale penultimate activations by `exp(s1/s2)`, then energy (`scale_enh`). |
| `gradnorm` | GradNorm | main | CORRECTED | L1 norm of the KL-to-uniform gradient w.r.t. head weights (`gradnorm_enh`). |
| `dimmad` | DiMMAD: Distance Multi-Metric Ensemble | main | ADAPTATION | Ensemble of distance metrics to class centroids; min-over-classes, median-over-metrics. |
| `catsight` | CatSight | main | ADAPTATION | CSP spatial filters on frozen features, distance to ID centroid (orientation fixed). |
| `codit` | CODiT: Conformal OOD Detection | main | FAITHFUL | Multi-draw conformal p-values over random transforms, combined via Fisher. |
| `invad` | InvAD: Invertible Anomaly Detection | main | ADAPTATION | Invertible decomposition with lossy (constant-residual) reconstruction error + confidence. |
| `m2n2` | M2N2: Test-time Adaptation | main | ADAPTATION | Autoencoder recon error with an EMA detrender adapting to new normals. |
| `deedee` | DEEDEE: OOD Dynamics Detection | main | FAITHFUL | Two trajectory statistics over time + isolation forest (`deedee_fix`). |
| `diffad` | DiffAD: Diffusion Anomaly Detection | main | FAITHFUL | Imputation-based partial-noise-and-denoise reconstruction error (`diffad_fix`). |

## Class-D appendix

| id | name | category | verdict | one-line |
|---|---|---|---|---|
| `outlier_exposure` | Outlier Exposure | class-D appendix | NOT-THE-METHOD | Production is a bare Energy baseline; faithful OE fine-tuning only in the class-D build. |
| `divoe` | DivOE: Diversified Outlier Exposure | class-D appendix | NOT-THE-METHOD | Production is energy on mean-centred logits; PGD outlier synthesis + OE only in the class-D build. |
| `diversemix` | DiverseMix | class-D appendix | NOT-THE-METHOD | No auxiliary outliers to diversify (ID interpolants); at chance; faithful build uses a real aux corpus. |
| `driftlens` | DriftLens | class-D appendix | ADAPTATION | Production is per-sample PCA-Mahalanobis; faithful batch-level Fréchet build in the appendix. |
| `ae_adwin_lstm` | AE-ADWIN-LSTM | class-D appendix | NOT-THE-METHOD | Temporal terms dead on shuffled windows + inverted; faithful ordered-stream build in the appendix. |
| `tdivdm` | TD-IVDM | class-D appendix | ADAPTATION | Gaussian KDE density on PCA features (relabel "KDE density"); multi-scale build in the appendix. |
| `diversify` | DIVERSIFY | class-D appendix | ADAPTATION | Frozen-feature diversity-regularised k-means; GRL adversarial build in the appendix. |
