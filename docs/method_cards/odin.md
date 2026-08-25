# ODIN (`odin`)

- **Category:** main
- **Paper:** Enhancing the Reliability of Out-of-Distribution Image Detection in Neural Networks — Liang et al., 2018
- **Official code:** https://github.com/facebookresearch/odin
- **Fidelity verdict:** FAITHFUL — defining ODIN mechanism reproduced correctly; one hyperparameter-magnitude deviation
- **Core idea:** Combine temperature scaling with a confidence-increasing input perturbation. Perturb the input by `x − ε·sign(∇ₓ CE(logits/T, ŷ))`, re-run the classifier, and take the max softmax at temperature T. The perturbation raises ID confidence more than OOD, sharpening separation. Post-hoc.
- **Key parameters:** Temperature `T = 1000`, perturbation `ε = 0.001` (fixed, untuned). Score `1 − max softmax(logits_pert/T)`, higher = OOD.
- **Divergences from original / caveats:**
  - Per-channel gradient std-normalisation (official divides sign by CIFAR raw-pixel stds) is omitted. This reparameterises ε into normalised units rather than being an identity: the effective perturbation is ~5.7× weaker than the official default (≈ raw ε 0.00025, near the bottom of the paper's 0–0.004 search grid). A hyperparameter deviation, not a mechanism change; ε sensitivity untested.
  - Sign convention flip is metric-invariant; batched gradient is provably bitwise-identical to the official per-sample computation.
  - At T=1000 the score concentrates in a ~1e-2-wide band near `1 − 1/K = 0.75` (94% of windows); ranking is unaffected and float32 headroom is ample. 18 windows saturate to exactly 0 from pathologically large backbone logits (backbone property, not ODIN).
  - Below-chance on the benchmark (mean 0.311 over 40; 0.261 TSB-U) — expected amplification of the MSP overconfidence inversion.
- **Where it runs:** `models/detectors/odin.py`
