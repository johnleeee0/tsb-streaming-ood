# InvAD: Invertible Anomaly Detection (`invad`)

- **Category:** main
- **Paper:** Detecting Both Seen and Unseen Anomalies in Time Series (per the implementation docstring), ACM TKDD 2025 (doi:10.1145/3717071). (The registry title differs; ACM DL was unreachable, so the title is unresolved.)
- **Official code:** https://github.com/fly-orange/InvAD
- **Fidelity verdict:** ADAPTATION (faithful mechanism) — restored by the 2026-08-21 fix; before the fix it was NOT-THE-METHOD (the reconstruction term was identically zero and the score collapsed to a scaled MSP)
- **Core idea:** An invertible network decomposes frozen features into a primary branch `z_id` and a residual `z_ood`. The residual is replaced by a constant before inverting, making the reconstruction **deliberately lossy** and its error informative. The score combines that reconstruction error with classifier confidence. Higher = OOD.
- **Key parameters:** inverse pass on `cat([z_id, full_like(z_ood, res_const)])` with `res_const=0.0`; recon score `MSE(x_recon, x) + MSE(z_ood, const)`; fixed seeded half-permutation between coupling layers; score `0.6·recon + 0.4·(1 − max_softmax)`; higher = OOD.
- **Divergences from original / caveats:**
  - Two provable pre-fix defects fixed: (1) the old `cat([z_id, z_ood])` reassembled the forward output bit-for-bit so reconstruction error was float round-off (~1e-14); (2) coupling layers left the first half untouched with no permutation, so `z_id` was bit-identical to the raw feature slice. Both confirmed gone (recon MSE ~78.7; `z_id` differs by up to 71.2). Network remains exactly invertible; informativeness comes solely from discarding `z_ood`, as in the official design.
  - Disclosed domain adaptations remain (InvAD-Lite): pooled frozen features not raw series, MLP subnets not attention/TCN/RNN, `1 − max_softmax` head not the learned `sigmoid(score_net)`, no SoftDTW pseudo-label alignment.
  - Re-run needed — the old row tracked MSP (TSB-U 0.356); the new signal is genuinely distinct with no guarantee of improvement.
- **Where it runs:** `models/detectors/invad.py`
