# ReAct: Rectified Activations (`react`)

- **Category:** main
- **Paper:** ReAct: Out-of-distribution Detection With Rectified Activations — Sun et al., 2021
- **Official code:** https://github.com/deeplearning-wisc/react
- **Fidelity verdict:** CORRECTED — the benchmark ran `react_enh`, a faithful reproduction of the paper's headline (ReAct+energy) configuration
- **Core idea:** Clip (rectify) the penultimate activations at a threshold `c` set from ID statistics, `min(x, c)`, to limit the effect of noisy over-activations on far-off-manifold inputs, then compute the energy score on the resulting logits. Higher energy indicates OOD. Post-hoc; `fit()` only estimates the threshold.
- **Key parameters:** `percentile = 90` (threshold = 90th percentile of ID activations, computed globally over samples and channels); base score = energy `−logsumexp`, higher = OOD; temperature `T = 1.0` (exposed, inert at default).
- **Divergences from original / caveats:**
  - The benchmark variant `react_enh` is faithful: clipping op/location, global (not per-channel) threshold, percentile 90, and energy base score all match `compute_threshold.py` / `score.py` and the paper.
  - Only divergences: a metric-invariant sign flip (which actually matches the paper's higher=OOD convention), the inert exposed temperature, and the absent 2000-sample cap on threshold estimation (immaterial with ~82 training windows).
  - The base `react.py` variant (ReAct+MSP) is also correctly implemented but is a supported non-headline configuration; `CHANGES.md` wrongly calls it an "inconsistency" — the paper explicitly permits MSP.
  - The base variant ran on only 4 of 40 datasets, so the claimed "decisive Phase 2" score-family ablation is n=4 evidence only — directionally favourable but too thin.
  - Below-chance on the benchmark (mean 0.284 over 40; 0.250 TSB-U): clipping bounds activations but does not prevent inflated backbone logits (extreme value −1.056e6) from dominating the energy score.
- **Where it runs:** `models/detectors/react.py`
