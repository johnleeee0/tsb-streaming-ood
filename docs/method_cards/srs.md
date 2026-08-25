# Seasonal Ratio Scoring / SRS (`srs`)

- **Category:** main
- **Paper:** Out-of-distribution Detection in Time-series Domain: A Novel Seasonal Ratio Scoring Approach — Belkhouja, Yan & Doppa, 2022/2023 (arXiv:2207.04306)
- **Official code:** https://github.com/tahabelkhouja/SRS
- **Fidelity verdict:** FAITHFUL — the seasonal ratio was restored by the 2026-08-21 fix (before the fix it was an ADAPTATION that returned the signal neg-ELBO alone)
- **Core idea:** Per-class STL decomposition splits each window into a seasonal signal and a residual; two class-conditional VAEs are trained on each. The score is the per-sample **ratio** of signal to residual neg-ELBO — OOD samples fit no ID class pattern, giving a high signal neg-ELBO and thus a high ratio. Higher = OOD.
- **Key parameters:** score `neg_elbo_sig / neg_elbo_res` with denominator clamped at `eps=1e-8`; per-class STL (`period=T//4`), single-pass xcorr circular alignment; CVAE latent 32; Adam lr 1e-3, ~30 epochs, `mc_samples=10`; higher = OOD.
- **Divergences from original / caveats:**
  - Titular seasonal ratio now formed (one-line fix at `score()`); before the fix only the signal neg-ELBO was returned, discarding the trained residual CVAE — that ranking was non-monotone vs the ratio, so pre-fix numbers must be regenerated.
  - The CVAE differs from the official generative model: Gaussian/MSE + linear decoder vs official Bernoulli/BCE + sigmoid on min-max data; decoder-only conditioning; neg-ELBO includes a KL term the official reconstruction likelihood omits; STL/alignment done per-sample rather than per-class-concatenated with iterative DTW. Deliberate adaptations to the frozen-backbone, per-window protocol.
  - The per-sample neg-ELBO deliberately **improves** on the official code, whose batch-likelihood reduction makes every sample in a batch share one constant score.
  - Official AUROC is orientation-agnostic (`max(auc, 1−auc)`) over a two-sided variance test, so the paper's reported SRS numbers are not directly comparable to this project's standard `roc_auc_score`.
  - Empirically the strongest verified detector (TSB-U mean 0.841 pre-fix); covers only 29 of 40 datasets (20 univariate) — the missing 11 are unexplained.
- **Where it runs:** `models/detectors/srs.py`
