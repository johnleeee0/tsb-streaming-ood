# Outlier Exposure (`outlier_exposure`)

- **Category:** class-D appendix
- **Paper:** Deep Anomaly Detection with Outlier Exposure — Hendrycks, Mazeika & Dietterich, ICLR 2019 (arXiv:1812.04606)
- **Official code:** https://github.com/hendrycks/outlier-exposure
- **Fidelity verdict:** NOT-THE-METHOD — the production detector is an Energy baseline; no training of any kind occurs; eval protocol: off-protocol (breaks the frozen-backbone fair comparison in the class-D build, so appendix-only; production runs additionally double-count the EBO/Energy row).
- **Core idea:** OE fine-tunes the classifier against an auxiliary outlier corpus with a 0.5-weighted cross-entropy-to-uniform penalty, so the model produces low-confidence (uniform) predictions on outliers. At test time an MSP or energy score then separates ID from OOD.
- **Key parameters (class-D build):** faithful OE objective `L = CE(id) + 0.5·CE_to_uniform(aux)` (λ=0.5); 10 epochs; lr=1e-3; batch size 64; energy scorer `score = −logsumexp(head(model(x))/T)`, higher = OOD; auxiliary corpus = real channel-matched hold-out TSB windows (persisted `aux_manifest.json`, no leakage). Two arms: `head_only` (ResNet frozen) and `full_net` (paper-faithful).
- **Divergences from original / caveats:**
  - Production `outlier_exposure.py` implements no `fit`, no optimizer, no backward pass, no auxiliary data, and no docstring — it returns `−logsumexp(logits/T)` on the shared ID-trained backbone (verified: 100% negative scores, ceiling at −log 4). It is exactly the Energy (EBO) baseline, and the same 40 score files back both the "Outlier Exposure" and "EBO" rows (double-counting).
  - Production mean AUROC 0.2949 (all 40) / 0.2770 (TSB-U, 15/21 below chance) — below chance, the logit-space inversion signature.
  - The class-D build closes the gap but BREAKS the frozen-backbone fair comparison (it updates weights the 17 leaderboard methods do not have), so it is appendix-only and never a leaderboard row; the clean Energy row stays in the main table.
  - Tiny ID/test sets (down to 2 windows on some STABLE files) make per-file AUROCs high-variance; on the smallest files fine-tuning can overfit (mitigated by early stopping and few epochs). Report aggregated over the full eval partition.
- **Where it runs:** `models/detectors/class_d/outlier_exposure.py` (`--group class_d`)
