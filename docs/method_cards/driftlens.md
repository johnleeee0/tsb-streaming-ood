# DriftLens (`driftlens`)

- **Category:** class-D appendix
- **Paper:** Unsupervised Concept Drift Detection from Deep Learning Representations in Real-time — Greco, Vacchetti, Apiletti & Cerquitelli, 2024 (arXiv:2406.17813; TKDE version also cited by the repo). (Both the registry and the code title are wrong.)
- **Official code:** https://github.com/grecosalvatore/drift-lens
- **Fidelity verdict:** ADAPTATION — the per-sample production variant is effectively a PCA-Mahalanobis detector and should be relabelled as such. A faithful batch-level class-D build restores the native granularity.
- **Core idea:** Official DriftLens computes one **distribution-to-distribution** distance (Fréchet, or KL/Bhattacharyya/JS/distribution-Mahalanobis) per window of embeddings, with a per-label decomposition. Higher distance = more drift.
- **Key parameters:** production `driftlens.py` scores per-sample squared Mahalanobis to a single global PCA-space baseline (`n_components=150`, capped); ignores labels; higher = OOD. Class-D build: offline PCA baseline `(μ_b, Σ_b)`, per monitoring batch the Fréchet (Wasserstein-2) distance; `B=32 (U)/16 (M)`; batch-level AUROC.
- **Divergences from original / caveats:**
  - The production variant substitutes a per-sample score the official code never defines; it ties the `mahalanobis` detector (median Spearman 0.999, 24/40 at ρ>0.99), and the two occupy the top two leaderboard places — reporting both as independent overstates diversity. Relabel as "PCA-Mahalanobis".
  - Dead code in production: `frechet_distance()`, `score_batch()`, and the entire threshold-estimation path are computed but never used. Per-label decomposition (the paper's emphasis) is absent.
  - Per-sample mean AUROC 0.844 (0.819 TSB-U). The class-D build restores the faithful batch-level Fréchet granularity but is a protocol change: files whose ordered stream is almost all OOD cannot form two mixed batches (batch AUROC undefined → NaN) — a genuine protocol incompatibility, not a bug.
- **Where it runs:** `models/detectors/class_d/driftlens.py` (`--group class_d`)
