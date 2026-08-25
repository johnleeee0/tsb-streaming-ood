# DivOE: Diversified Outlier Exposure (`divoe`)

- **Category:** class-D appendix
- **Paper:** Diversified Outlier Exposure for Out-of-Distribution Detection via Informative Extrapolation — Zhu, Lin, Zhou, Yang, Yang, Liu et al., NeurIPS 2023
- **Official code:** https://github.com/ZFancy/DivOE
- **Fidelity verdict:** NOT-THE-METHOD — the production detector is energy on mean-centred logits; no synthesis, no auxiliary data, no training. A faithful class-D build closes the gap in the appendix.
- **Core idea:** DivOE synthesises diversified outliers by multi-step projected-gradient ("informative extrapolation") on the Outlier Exposure objective in input space, mixes them into a real auxiliary outlier batch, and fine-tunes the classifier with the OE loss. At test time an energy/MSP score separates ID from OOD.
- **Key parameters (class-D build):** input-space PGD synthesis (`extrapolation_ratio=0.5`, `num_steps=5`, ε=0.1 in normalised-window units); fine-tune with `CE(id) + 0.5·CE_to_uniform` over synthesised ∪ real aux outliers; energy scorer `−logsumexp`, higher = OOD; two arms (`head_only`, `full_net`).
- **Divergences from original / caveats:**
  - Production `divoe.py` implements none of the method: no `extrapolate`/`backward`/optimizer/aux data/CE loss and no docstring. `fit()` only stores a logit mean; `score()` returns energy on mean-centred logits — an invented step that is near-inert (ρ=0.9999 vs plain energy) and slightly **hurts** (−0.019 AUROC).
  - It duplicates the Energy/EBO result: EBO, Outlier Exposure and DivOE are three rows backed by one energy detector (two literally the same files). Below chance (mean 0.276; 0.230 TSB-U).
  - The class-D build is a faithful OE+synthesis reproduction but BREAKS the frozen-backbone fair comparison (training + input-space PGD), so it is appendix-only and never a leaderboard row. On the tiny ID sets DivOE tracks plain OE closely; report the OE-vs-DivOE delta honestly.
- **Where it runs:** `models/detectors/class_d/divoe.py` (`--group class_d`)
