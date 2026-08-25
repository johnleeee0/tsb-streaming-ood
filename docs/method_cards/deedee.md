# DEEDEE: OOD Dynamics Detection (`deedee`)

- **Category:** main
- **Paper:** DEEDEE: Fast and Scalable Out-of-Distribution Dynamics Detection — Aljaafari, Kanade, Torr & Schroeder de Witt, 2025 (arXiv:2510.21638). (The registry title is wrong.)
- **Official code:** none public (no repository found; paper reachable only via secondary summary)
- **Fidelity verdict:** FAITHFUL — production now wires the faithful `deedee_fix` variant (2026-08-21); the base `deedee` was NOT-THE-METHOD and is retired
- **Core idea:** Summarise each window by two trajectory statistics — an episodewise mean over the time axis and an RBF self-similarity over consecutive timesteps — and fit an isolation forest on the ID trajectory statistics. Higher anomaly score (`−score_samples`) = OOD.
- **Key parameters:** `deedee_fix` computes the mean as `x.mean(axis=2)` over the real time axis and the RBF over consecutive-timestep differences; **one** vectorised isolation forest; score `−score_samples`, higher = OOD.
- **Divergences from original / caveats:**
  - The retired base `deedee` was NOT-THE-METHOD on every axis: its "episodewise mean" was a single raw feature value (no averaging), its RBF ran over unordered embedding **dimensions** not time, it fit `feat_dim` separate forests, and it made one isolation-forest call per (sample, dimension) — inverting DEEDEE's headline 600× compute reduction into one of the slowest detectors. It had results on only 4 ablation datasets.
  - `deedee_fix` fixes the axis, the mean and the cost, and is verified end-to-end in the integration smoke test. Its RBF term is a consecutive-timestep **self**-similarity rather than similarity to a stored training summary — the one remaining gap from the paper.
  - Low verification confidence: no public code and only an abstract-level secondary summary of the paper, so no equation-level comparison was possible.
- **Where it runs:** `models/detectors/deedee.py`
