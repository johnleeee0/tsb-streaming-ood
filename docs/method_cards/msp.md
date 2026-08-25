# Maximum Softmax Probability (`msp`)

- **Category:** main
- **Paper:** A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks — Hendrycks & Gimpel, 2016/2017
- **Official code:** https://github.com/hendrycks/error-detection
- **Fidelity verdict:** FAITHFUL — minimal, faithful reproduction of the max-softmax-probability baseline
- **Core idea:** Run the pre-trained classifier and take the maximum softmax probability over classes as a confidence score. In-distribution inputs tend to be classified more confidently, so low max-softmax indicates OOD. Purely post-hoc, no retraining.
- **Key parameters:** Temperature `T = 1.0` (exposed but default). Score reported as `1 − max softmax(logits/T)` so higher = OOD.
- **Divergences from original / caveats:**
  - Sign convention flipped (`1 − c` vs raw max-softmax) and temperature exposed; both are metric-invariant at defaults, so AUROC/AUPR/FPR@95 are identical to the official code.
  - Framework port Theano/Lasagne → PyTorch; `fit()` is a no-op (post-hoc only).
  - Below-chance AUROC on the streaming benchmark (mean 0.385 over 40 datasets) is genuine softmax overconfidence on far-off-manifold windows, confirmed from saved scores — not an orientation bug. Observed score range `[0, 0.7427] ≤ 1 − 1/K` with K=4 pseudo-classes.
  - Five degenerate datasets have constant scores (saturated softmax) giving AUROC 0.5 by tie-breaking.
  - Unresolved: 18-vs-21 univariate dataset-count discrepancy between prior notes and on-disk artifacts (conclusion unchanged).
- **Where it runs:** `models/detectors/msp.py`
