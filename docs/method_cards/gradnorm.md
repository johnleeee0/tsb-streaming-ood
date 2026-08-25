# GradNorm (`gradnorm`)

- **Category:** main
- **Paper:** On the Importance of Gradients for Detecting Distributional Shifts in the Wild — Huang, Geng & Li, NeurIPS 2021 (arXiv:2110.00218)
- **Official code:** https://github.com/deeplearning-wisc/gradnorm_ood
- **Fidelity verdict:** CORRECTED — the benchmark variant `gradnorm_enh` is a faithful reproduction; the base variant was wrong on all four axes
- **Core idea:** Backpropagate the KL divergence between the softmax output and a uniform distribution, take the gradient with respect to the last classification-layer weights, and use its L1 norm as the score. Gradient magnitude is higher for ID than OOD, so the norm is negated for the higher-is-OOD convention.
- **Key parameters:** loss = KL-to-uniform (implemented as a ones-target cross-entropy, differing from `D_KL` only by the positive factor `K`); gradient w.r.t. last-FC weights; **L1** norm; temperature `T = 1.0`; per-sample scoring; score `−L1`, higher = OOD.
- **Divergences from original / caveats:**
  - `gradnorm_enh` is faithful: two non-obvious equivalences verified bitwise-exactly — (a) computing features under `no_grad` gives identical head-weight gradients to the official full-graph backward; (b) the ones-target loss is affine in KL-to-uniform, so the L1 ranking matches the paper's Eq. (4).
  - The base `gradnorm.py` is wrong in variable (input not weights), loss (CE-to-argmax not KL), norm (L2 not L1) and orientation (uninverted), and misattributes its orientation to the paper — quarantine or delete.
  - **The one `_enh` correction that clearly pays off on real data:** +0.1217 mean AUROC over the base across the 4 shared datasets (vs ReAct +0.019, DICE +0.0004, SCALE −0.025) — use as the worked example for the corrected-variant narrative. Still n=4.
  - Per-sample Python loop (O(N) backward passes) is unavoidable and faithful, just slow. Below-chance overall (mean 0.289; 0.247 TSB-U) — the logit-space family inverts.
- **Where it runs:** `models/detectors/gradnorm.py`
