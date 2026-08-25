# TD-IVDM (`tdivdm`)

- **Category:** class-D appendix
- **Paper:** TD-IVDM: A multi-scale concept drift detection method for time series forecasting tasks — Wang, Zhu, Qin, Han & Yan, Neurocomputing, 2025 (doi:10.1016/j.neucom.2025.131120). TD-IVDM = "Time Dependency – Inter Variable Dependency" (both the registry and code acronym renderings are wrong).
- **Official code:** none public (no repository found; paper paywalled)
- **Fidelity verdict:** ADAPTATION — effectively a generic KDE density detector and should be relabelled "KDE density". A multi-scale-inspired class-D build exists but makes no fidelity claim.
- **Core idea:** Estimate a Gaussian kernel density over PCA-reduced frozen features and score by negative log-density. The paper proper pairs multi-dimensional KDE (inter-variable dependency) with an improved TS2Vec representation network (time dependency) at multiple scales. Higher `−log density` = OOD.
- **Key parameters:** `scipy.stats.gaussian_kde` on 20-component whitened PCA features; `bw_method='scott'`; score `−log_density`, higher = OOD. Class-D build measures density in several PCA subspaces (`scales=[5,10,20]`), standardised per scale and aggregated, on an ordered stream.
- **Divergences from original / caveats:**
  - Multi-dimensional KDE is genuinely one of the two pillars, so the KDE choice is not arbitrary — but the TS2Vec time-dependency branch, the multi-scale treatment of time frames/variable subsets, the three-stage workflow, and the streaming drift task are all absent. Docstring line claiming "temporal and inter-variable dependencies at multiple scales" is false and should be deleted.
  - Strongest verified detector on the full 40-set (0.838; 0.815 TSB-U) but 16/40 datasets score exactly 1.000, largely the extreme-feature-magnitude family where any density method separates trivially — the headline is partly a benchmark artefact.
  - Lowest verification confidence of any method: no code, paper paywalled (403 on ScienceDirect and ResearchGate), no equation-level comparison possible. Class-D build is captioned "TD-IVDM-inspired (unverifiable)".
- **Where it runs:** `models/detectors/class_d/tdivdm.py` (`--group class_d`)
