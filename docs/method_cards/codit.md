# CODiT: Conformal OOD Detection (`codit`)

- **Category:** main
- **Paper:** CODiT: Conformal Out-of-Distribution Detection in Time-series Data for Cyber-Physical Systems — Kaur, Sridhar, Jha, Roy, Sokolsky & Lee, ICCPS 2022 (arXiv:2207.11769)
- **Official code:** https://github.com/kaustubhsridhar/time-series-OOD
- **Fidelity verdict:** FAITHFUL (on-protocol) — restored by the 2026-08-21 fix; before the fix it was NOT-THE-METHOD
- **Core idea:** Train a transform classifier on ID windows, then for each test window draw multiple random temporal transformations, compute a conformal p-value per draw from a held-out ID calibration set, combine them into a Fisher statistic, and score by the (negated) Fisher value. Higher = OOD.
- **Key parameters:** five transforms (`speed, shuffle, reverse, periodic, identity`); `eval_n = 20` random draws combined; conformal p-value `(#{test ≤ cal} + 1)/(n_cal + 1)`; Fisher term count matched to `eval_n`; seeded calibration split (seed 42); score `−Fisher` (official convention: higher Fisher = ID).
- **Divergences from original / caveats:**
  - All four pre-fix defects corrected: multi-draw random-transform nonconformity (was identity-only), `eval_n`-way p-value combination (was a single p-value), correct Fisher term count (a hardcoded 20 terms on one p-value had saturated every score to within 3e-11 of 1.0 and left 19/40 datasets numerically constant), and the official orientation (was inverted). Re-run needed — the ~0.386 pre-fix mean must not be reported.
  - Two disclosed, protocol-mandated adaptations remain (not fidelity defects here): transform classifier is a linear head on frozen features rather than the end-to-end `r3d_regressor`; the official run-length detection over an ordered trace has no analogue under per-window shuffled evaluation.
- **Where it runs:** `models/detectors/codit.py`
