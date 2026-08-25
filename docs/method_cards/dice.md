# DICE: Directed Sparsification (`dice`)

- **Category:** main
- **Paper:** DICE: Leveraging Sparsification for Out-of-Distribution Detection — Sun & Li, ECCV 2022 (arXiv:2111.09805)
- **Official code:** https://github.com/deeplearning-wisc/dice
- **Fidelity verdict:** FAITHFUL — the benchmark variant `dice_enh` was rewritten (fix 2026-08-21) to implement the static ID-mean weight mask; before the fix both variants were NOT-THE-METHOD
- **Core idea:** Rank the classifier's FC weights by their mean contribution over ID data (ID-mean feature vector × weights), threshold at a percentile to build a **static** binary mask, sparsify the weights once, and score every input by the energy over the sparsified logits. Higher = OOD.
- **Key parameters:** sparsity by global percentile `p = 90` over the full `(C, D)` contribution matrix (one-sided, discards negatives → unequal units per class); signed ranking (`np.abs` disabled, matching official); base score energy `−logsumexp`; `fit()` builds the ID-mean mask. The runner still passes `top_k=20` but the faithful path ignores it.
- **Divergences from original / caveats:**
  - `dice_enh` now reproduces all four decisive official properties (precomputed ID-mean mask, signed ranking, global percentile, one-sided selection); verdict FAITHFUL within the frozen-backbone/linear-head protocol. Re-run needed to regenerate genuine DICE numbers.
  - The base `dice.py` variant remains NOT-THE-METHOD (per-sample abs-topk of MSP, no `fit()`), ran on only 4 datasets.
  - Pre-fix, the old per-sample abs-topk correlated only ρ≈0.43 with the official static mask and its claimed advantage over the base was +0.0004 across 4 datasets — the ablation claim in `CHANGES.md` is unsupported.
  - Below-chance pre-fix (mean 0.286 over 40; 0.258 TSB-U, 16/21 below chance) — energy inherits the inflated-backbone-logit inversion (extreme −1.04e6).
- **Where it runs:** `models/detectors/dice.py`
