# DIVERSIFY (`diversify`)

- **Category:** class-D appendix
- **Paper:** Out-of-Distribution Representation Learning for Time Series Classification — Lu et al., ICLR 2023 (arXiv:2209.07027); TPAMI extension doi:10.1109/TPAMI.2024.3355212
- **Official code:** https://github.com/microsoft/robustlearn (`diversify/`)
- **Fidelity verdict:** ADAPTATION — honestly labelled "DIVERSIFY-Lite" in source, but the gap from the original is total. A faithful adversarial class-D build reproduces the mechanism in the appendix.
- **Core idea:** Official DIVERSIFY adversarially **retrains the feature extractor** through a gradient-reversal layer across three update pathways (seven networks), obtaining worst-case latent domains and matching their distributions. It is a representation-learning / domain-generalisation method and **defines no OOD score**.
- **Key parameters:** production `diversify.py` trains **only K centroid vectors** on a permanently frozen backbone (diversity-regularised k-means: `cluster_loss + α·Σ 1/dᵢⱼ`); score = min Euclidean distance to a centroid, higher = OOD. Class-D build: from-scratch 1-D CNN featurizer trained via a Gradient Reversal Layer with cosine-distance latent-domain assignment on L2-normalised features; invented score = energy `−logsumexp` (primary) or cosine-centroid distance (secondary).
- **Divergences from original / caveats:**
  - Production omits the adversarial retraining entirely (frozen backbone, centroids only), assigns domains by Euclidean on unnormalised features (official uses cosine on normalised), and **invents** an OOD score the paper never defines. The "adversarial" label in the source is a misnomer — recommend "diversity-regularised k-means".
  - Third-strongest verified detector (0.699 over 37; 0.660 TSB-U). Covers only 18 univariate datasets — three small YAHOO series silently produced no output (`N < latent_domain_num`), which is the likely origin of the recurring "18 univariate datasets" figure across the audit.
  - The class-D build reproduces the GRL adversarial extractor and fixes the `N < latent_domain_num` crash but BREAKS the frozen-backbone fair comparison (learns its own representation) → appendix-only, exploratory (no ground-truth OOD number to validate against).
- **Where it runs:** `models/detectors/class_d/diversify.py` (`--group class_d`)
