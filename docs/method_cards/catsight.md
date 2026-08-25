# CatSight (`catsight`)

- **Category:** main
- **Paper:** CatSight, a direct path to proper multi-variate time series change detection: perceiving a concept drift through common spatial pattern — Flórez, Rodríguez-Moreno, Artetxe et al., Int. J. Machine Learning and Cybernetics, 2023 (doi:10.1007/s13042-023-01810-z). (The registry title is wrong.)
- **Official code:** none public (no repository found; paper paywalled)
- **Fidelity verdict:** ADAPTATION — honestly labelled "CatSight-Lite"; the score orientation was flipped by the 2026-08-21 fix
- **Core idea:** Learn Common Spatial Pattern (CSP) filters via a generalised eigenproblem on ID pseudo-class covariances, project frozen features into CSP space, and score by the normalised distance to the ID centroid. Higher distance = more OOD (post-fix).
- **Key parameters:** CSP via `eigh(C1, C1+C2)` with `1e-4·I` regularisation; 3 top + 3 bottom of 6 components; ID mean/std reference in CSP space; score `+‖normalized_diff‖` (post-fix, un-negated); higher = OOD.
- **Divergences from original / caveats:**
  - Orientation fix is load-bearing: as-implemented (negated) mean AUROC was 0.250 and returned exactly 0.000 on all five extreme-feature-magnitude datasets; flipped it is 0.750 (would make CatSight the second-strongest detector). Re-run and re-report.
  - The CSP machinery is verified sound: the eigenproblem is provably equivalent to the documented form (identical eigenvectors/order), and the confusing double-indexing picks the intended component set (correct but fragile).
  - The paper's step (ii) — a **trained classifier** on CSP features — is replaced by a centroid distance (the substantive adaptation). Only the first two of four ID pseudo-classes are used for CSP.
  - Low verification confidence: paper paywalled and no public code, so the CSP was checked only for internal mathematical correctness, not against the paper's equations. Covers 18 univariate datasets (three small YAHOO series silently dropped, same three as DIVERSIFY).
- **Where it runs:** `models/detectors/catsight.py`
