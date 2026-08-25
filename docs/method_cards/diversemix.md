# DiverseMix (`diversemix`)

- **Category:** class-D appendix
- **Paper:** Out-Of-Distribution Detection with Diversification (Provably) — Yao, Han, Fu, Peng, Hu & Zhang, NeurIPS 2024 (arXiv:2411.14049). (The registry title is wrong.)
- **Official code:** https://github.com/HaiyunYao/diverseMix
- **Fidelity verdict:** NOT-THE-METHOD — the production detector has no auxiliary outlier set to diversify and is at chance on real data in either orientation. A faithful class-D build closes the gap in the appendix.
- **Core idea:** DiverseMix enhances the diversity of a **collected** auxiliary outlier corpus via score-adaptive mixup, training an energy head to assign high energy (low ID-ness) to outliers. Score = `−logsumexp`, higher = OOD.
- **Key parameters (class-D build):** energy head with `relu(logsumexp+1)` auxiliary penalty; score-adaptive Beta mixup `λ ~ Beta(ŝ_aux·α, ŝ_id·α)` pairing REAL aux outliers with ID samples; score `−logsumexp` (official orientation); two arms (`head_only`, `full_net`).
- **Divergences from original / caveats:**
  - Production `diversemix.py` has **no auxiliary outlier data** — it fabricates pseudo-outliers by convex-combining ID features from two classes, which land 100% inside the ID distribution (closer to the centre than real ID data), so the auxiliary loss trains the head backwards. The defining mechanism is absent.
  - Orientation resolves cleanly (not "regime-dependent"): the official `eval_ood_detection.py` treats `+logsumexp` as an ID-ness score, so OOD = `−logsumexp` (the `_enh` choice); the base `+logsumexp` contradicts both the official convention and its own objective. Moot in practice — the two variants are exact negations (`base+enh=1.0`) and the base sits at 0.520 over 40 datasets (0.502 TSB-U). Report as a negative result.
  - The class-D build uses a REAL channel-matched hold-out aux corpus and the correct orientation but BREAKS the frozen-backbone fair comparison → appendix-only; result quality is gated by aux-corpus quality.
- **Where it runs:** `models/detectors/class_d/diversemix.py` (`--group class_d`)
