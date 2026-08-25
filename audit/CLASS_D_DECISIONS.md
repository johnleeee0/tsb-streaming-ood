# Class-D Build — Author Decisions (locked 2026-08-21)

These are the binding choices for building the 7 excluded detectors as a separate appendix study.
All builds must follow these; anything not listed uses the BUILD_PLAN_CLASS_D.md recommended default.

## Cross-cutting
1. **Auxiliary outlier corpus:** HOLD-OUT TSB files (primary). Partition the 600-file corpus into
   eval vs aux with a **persisted manifest** so no aux file is ever an eval file (no leakage).
   Synthetic outliers = ablation only.
2. **Multivariate aux matching:** exact channel-count match; synthetic fallback if none.
3. **Fine-tuning scope (OE / DivOE / DiverseMix):** run BOTH arms — (a) head-only (least-unfair vs
   the frozen-backbone 17) and (b) full-net (paper-faithful) — and report the pair.
4. **Where Class-D runs live:** a NEW `experiments/run_class_d.py` + separate appendix result files
   (`results/class_d_*.csv`). The production harness (`tsb_benchmark.py`, `run_experiments.py`) and its
   resumable results stay UNTOUCHED. Deep-copy the shared backbone before any fine-tuning so it is never
   mutated.
5. **Ordered-eval flag:** add `ordered_eval=False` to `load_tsb` (backward-compatible; default preserves
   current shuffled behaviour).
6. **Scope:** BUILD ALL 7 FULLY. TD-IVDM and DIVERSIFY are captioned honestly as
   "inspired / unverifiable (paper paywalled, no code)" and "score added; paper defines none".

## Per-method defaults (accepted)
- **OE:** λ=0.5, 10 epochs, lr=1e-3, oe_batch_size=64, early-stop on val.
- **DivOE:** input-space PGD on the fine-tuned copy, ε in normalised-window units, num_steps=5, ratio=0.5.
- **DiverseMix:** −logsumexp (official orientation); both fine-tune arms as in (3).
- **DriftLens:** batch-level Fréchet, B=32 (U) / 16 (M) consecutive non-overlapping windows, τ=0.5;
  batch-level AUROC ONLY. Do NOT report a per-sample proxy (it duplicates mahalanobis, ρ≈0.999).
- **TD-IVDM:** keep "KDE-density" in the main study; appendix reconstruction labelled
  "TD-IVDM-inspired (unverifiable)".
- **AE-ADWIN-LSTM:** real ADWIN (river or a correct custom impl), enable incremental update, flip
  orientation; headline per-window AUROC on the ordered stream + drift-delay as a secondary metric.
- **DIVERSIFY:** invented score = energy (−logsumexp) primary, cosine-centroid secondary; minimal
  from-scratch GRL adapted to 1-D windows; fix the N < latent_domain_num crash.

## Reporting frame
All 7 are SEPARATE appendix studies, never rows in the main 17-method leaderboard. Redundant /
unverifiable / at-chance cases are reported as explicit negative or protocol-incompatible results,
consistent with `CLASS_D_EXCLUSIONS.md`.
