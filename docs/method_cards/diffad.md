# DiffAD: Diffusion Anomaly Detection (`diffad`)

- **Category:** main
- **Paper:** Imputation-based Time-Series Anomaly Detection with Conditional Weight-Incremental Diffusion Models — Xiao et al., KDD 2023 (doi:10.1145/3580305.3599391)
- **Official code:** https://github.com/ChunjingXiao/DiffAD
- **Fidelity verdict:** FAITHFUL — production now wires the faithful `diffad_fix` variant (2026-08-21); the base `diffad` was NOT-THE-METHOD
- **Core idea:** Imputation-based: partially noise the **input** to an intermediate diffusion step and denoise it back, so the reconstruction is conditioned on the specific sample. Large (non-negated) reconstruction error = OOD.
- **Key parameters:** `diffad_fix` noises the input to `t0 = n_steps // 2` then denoises back; reports the non-negated reconstruction error; DDPM ε-prediction MLP denoiser, linear β schedule 1e-4→0.02; `n_steps=20`, `recon_samples=2`; higher = OOD.
- **Divergences from original / caveats:**
  - The base `diffad` ran the reverse process from **pure noise** independent of the input (`torch.randn_like`) and negated the score — provably reducing to an inverted distance-to-ID-mean detector (ρ=0.971 against that quantity). Mean AUROC 0.287 (0.264 TSB-U, 15/21 below chance), exactly 0.000 on all five extreme-magnitude datasets. The 0.287 must not be reported as DiffAD.
  - `diffad_fix` restores input-conditioned partial-noising and removes the negation; verified end-to-end in the integration smoke test.
  - Evidence limitation: the official reverse-loop initialisation (`networks.define_G()`) was not read, so the exact partial-noising level `t0` is a reasonable default, not a verified match — but the official test path is confirmed input-conditioned (`super_resolution(self.data['SR'], …)`, and "imputation" implies conditioning). Operand is frozen features (disclosed adaptation).
- **Where it runs:** `models/detectors/diffad.py`
