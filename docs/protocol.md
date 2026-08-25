# Evaluation protocol

## Main benchmark (the 17 faithful detectors)
- **Split:** source-boundary — train on Source-1 normal windows only; evaluate on later windows.
- **Windowing:** size 64 / stride 32 (U), 128 / 64 (M).
- **Backbone:** a single ResNet trained per dataset (cross-entropy on temporal pseudo-classes),
  **shared frozen** across all post-hoc detectors so the comparison is fair.
- **Labels:** window-level OOD; metrics are **per-sample AUROC / AUPR / FPR95** + detection accuracy.
- **Normalization:** per-series for Medical/HumanActivity domains, global otherwise.
- Seed 42, CPU, resumable.

## Why 7 detectors are a separate appendix (class-D)
They cannot be faithfully evaluated under the protocol above, so mixing them into the leaderboard
would be misleading. Each runs under its own protocol via `--group class_d`:
- **outlier_exposure, divoe, diversemix** — require an auxiliary-outlier corpus + fine-tuning
  (backbone is retrained on a deep copy; both head-only and full-net arms reported).
- **driftlens, tdivdm, ae_adwin_lstm** — require ordered/window-level evaluation (the main path
  shuffles windows). DriftLens is scored at **batch level**; the other two on the **ordered stream**.
- **diversify** — adversarial representation learning that trains its own extractor and defines no
  OOD score (an energy/cosine score is added, disclosed as such).

Full rationale: [`../audit/CLASS_D_EXCLUSIONS.md`](../audit/CLASS_D_EXCLUSIONS.md). The shared frozen
backbone is **never mutated** by any class-D fine-tuning (deep-copied first; enforced in tests).
