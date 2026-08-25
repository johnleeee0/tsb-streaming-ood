# DiMMAD: Distance Multi-Metric Ensemble (`dimmad`)

- **Category:** main
- **Paper:** In Search of the Unknown Unknowns: A Multi-Metric Distance Ensemble for Out of Distribution Anomaly Detection in Astronomical Surveys — Chaini, Bianco & Mahabal, NeurIPS ML4PS 2025 (arXiv:2510.23702). (The registry title/link are wrong.)
- **Official code:** https://github.com/sidchaini/distclassipy (metric library) + https://github.com/sidchaini/dimmad (scoring notebooks). The registry URL `DiMMAD/DistClassiPy` 404s and the local clone is empty.
- **Fidelity verdict:** ADAPTATION — honestly labelled "DiMMAD-Lite"; the 2026-08-21 fix restored continuous Jaccard and the median centroid statistic but the ensemble is still a subset
- **Core idea:** Compute the distance of each test sample to every ID class centroid under a diverse ensemble of distance metrics, aggregate min-over-classes then median-over-metrics, and rank highest as most anomalous. Higher = OOD.
- **Key parameters:** `class_agg='min'`, `metric_agg='median'` (verbatim the paper's prescription); 11-metric continuous ensemble (post-fix; incl. Ružička/continuous Jaccard); class centroids via **median** central statistic (post-fix, matching DistClassiPy default); higher = OOD.
- **Divergences from original / caveats:**
  - The two-stage aggregation is exact; the **metric ensemble** — the method's actual contribution — overlaps the paper's 16 metrics on only 8/16, with three non-paper members retained (`mahalanobis`, `minkowski(p=3)`, `standardized_euclidean`) and eight paper metrics still absent (Clark, Hellinger, Kulczynski, Lorentzian, Meehl, Motyka, Soergel, Wave-Hedges).
  - Fix restored continuous Jaccard (the prior fix wrongly deleted it) and switched centroid statistic mean→median. Correctly removed Hamming/Dice (not in the paper's 16).
  - Domain shift from astronomical light-curve features to frozen deep features (disclosed). Silent Euclidean fallback on metric failure is a correctness hazard (in the untouched base file).
  - Second-strongest verified detector (TSB-U 0.790, below chance on only 2/21). The correction did not improve real-data performance (Δ −0.0074 on n=4, with a metric-set confound).
- **Where it runs:** `models/detectors/dimmad.py`
