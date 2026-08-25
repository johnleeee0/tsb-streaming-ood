# Energy-based OOD Detection / EBO (`energy`)

- **Category:** main
- **Paper:** Energy-based Out-of-distribution Detection — Liu et al., 2020
- **Official code:** https://github.com/wetliu/energy_ood
- **Fidelity verdict:** FAITHFUL — faithful reproduction of the EBO energy score
- **Core idea:** Score each input by the free energy over its classifier logits, `E(x) = −T·logsumexp(logits/T)`. Higher energy (unnormalised logit magnitude is low) indicates OOD. Parameter-free and post-hoc; production runs the clean energy (EBO) path, not the Outlier Exposure training variant.
- **Key parameters:** Temperature `T = 1.0`. Implemented statistic `−logsumexp(logits/T)`, higher = OOD.
- **Divergences from original / caveats:**
  - The outer `× T` factor is omitted; the implemented score is `E/T`. Exactly identical to official at `T = 1` (the default), and a positive rescaling otherwise (rank/AUROC preserved). Would only matter under a fixed absolute threshold, moot at the default.
  - Reference clone `methods/energy_ebo/reference/` is absent; verification relied on the live official repo plus a byte-identical copy in the local DivOE reference. Structural checks on saved scores confirm `−logsumexp` with K=4.
  - Below-chance on the benchmark (mean 0.295 over 40; 0.277 TSB-U) and inverts more strongly than MSP — energy reads unnormalised logit magnitude, so it is more exposed to inflated logits (order 10⁵–10⁶ on some datasets).
  - Only the `outlier_exposure` energy path is unmodified EBO; other consumers (scale, divoe, diversemix) apply their own transforms.
  - Unresolved shared 18-vs-21 univariate dataset-count discrepancy.
- **Where it runs:** `models/detectors/energy.py`
