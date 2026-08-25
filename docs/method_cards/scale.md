# SCALE: Activation Scaling (`scale`)

- **Category:** main
- **Paper:** Scaling for Training Time and Post-hoc Out-of-distribution Detection Enhancement — Xu, Lian, Liu, Jiang et al., ICLR 2024 (arXiv:2310.00227)
- **Official code:** https://github.com/kai422/SCALE
- **Fidelity verdict:** CORRECTED — the benchmark variant `scale_enh` is a faithful reproduction of SCALE's post-hoc scaling; the percentile default was fixed 65→85 (2026-08-21)
- **Core idea:** Rescale the penultimate activations by `exp(s1/s2)`, where `s1` is the sum of all activations and `s2` the sum of the top-(100−p) percentile — pruning is used only to compute `s2`, and the **original unpruned** activations are scaled. The scaled features pass through the head and are scored with energy. Higher = OOD.
- **Key parameters:** `percentile = 85` (post-fix, matching the paper's validated p=0.85 and official config); scaling factor `exp(s1/s2)`; base score energy `−logsumexp`; `torch.relu(feats)` applied before scaling (backbone adaptation); no fitting stage.
- **Divergences from original / caveats:**
  - Scaling operation is **numerically identical** to the official `scale()` (max abs diff 0 to 5.7e-6, float32 noise) — the strongest positive fidelity result in the set.
  - Percentile fixed 65→85; at p=65 the mean sharpening was ≈2.94 vs ≈5.55 at p=85 (about half the paper's), and this is not rank-preserving — SCALE numbers must be regenerated at p=85.
  - ReLU on features is a required, disclosed adaptation (the `s1/s2 ≥ 1` invariant needs non-negative activations); a `clamp(max=50)` guard is verified never to bind at p≤85.
  - The base `scale.py` is not SCALE — it z-standardises the **logits** (wrong layer, wrong operation, needs a fitting stage); ran on 4 datasets only.
  - No automatic percentile search (official uses APS). Below-chance pre-fix (mean 0.289 over 40; 0.260 TSB-U) — the logit-inversion signature.
- **Where it runs:** `models/detectors/scale.py`
