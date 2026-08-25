# CHANGES vs the prior write-up (for reconciliation with `main.tex`)

**Date:** 2026-08 (combined U+M finalization) · Author: S. Giannoulis (AUTH) · Supervisor: J. Paparrizos

The draft sections `paper/results_draft.tex` and `paper/discussion_draft.tex` are
regenerated from the authoritative, **finalized combined TSB-StreamingAD-U +
TSB-StreamingAD-M** results in `results/findings.md`, the `results/tables/*.tex`,
`audit/METHOD_VERIFICATION_FULL.md`, and `audit/CLASS_D_EXCLUSIONS.md`. They differ
from the current `main.tex` Results/Discussion (and from the earlier U-only draft)
as follows. Every number below traces to `results/findings.md` or the generated
tables. `main.tex` itself was **not** edited per instruction; these deltas record
what must change there to reconcile with the finalized drafts.

## 1. Scale of the benchmark (combined U+M)
- **Detectors:** prior `main.tex` says **22** evaluated (23 candidates). Now: **17**
  evaluated, **24** audited candidates, **7** excluded as protocol-incompatible.
- **Datasets / runs:** prior says **36 datasets, 772 runs**. Now: **527 datasets
  (260 univariate + 267 multivariate), 8668 runs**. (The earlier interim U-only
  draft said 260 datasets / 4400 runs / TSB-M pending; TSB-M is now run.)
- **Friedman test:** prior reports chi2 = **61.15**, p = 8.5e-6, k=22, N=22. Now:
  chi2 = **1674.97**, p < 1e-16, k=**17**, N=**250**. Caveat: N=250 is
  complete-case — SRS is univariate-seasonal and its **TSB-M run is pending**, so the
  complete-case pool is effectively univariate-complete. State this honestly.

## 2. Combined leaderboard (all mean AUROC over 527 datasets)
| method | prior main.tex | finalized combined U+M |
|---|---|---|
| mahalanobis | 0.874 | **0.826** (sole top) |
| dfm-pca | 0.821 | **0.812** |
| dimmad | 0.782 | **0.747** |
| invad | 0.432 | **0.740** |
| catsight | 0.221 (inverted!) | **0.735** |
| m2n2 | 0.784 | **0.733** |
| deedee | (not listed) | **0.700** |
| codit | 0.384 | **0.684** |
| diffad | 0.262 | **0.531** (just above chance) |
| msp | 0.386 | **0.370** (below chance) |
| srs | 0.790 (top TS method) | **0.352** (below chance; U-only, M pending) |
| dice | 0.261 | **0.313** |
| scale | 0.270 | **0.308** |
| odin | 0.294 | **0.303** |
| react | 0.267 | **0.303** |
| energy | — | **0.301** |
| gradnorm | 0.267 | **0.294** |

**Top tier = the eight distance/density/feature-manifold detectors** (mahalanobis …
codit, 0.684–0.826). **Every** post-hoc softmax/logit/activation/gradient detector
(msp, dice, scale, odin, react, energy, gradnorm) and the seasonal SRS fall **below
the 0.5 chance line**, several inverted.

## 3. Per-split (both families now real)
- The `tab:tsb_split` table now carries **both** U and M columns (prior U-only draft
  had TSB-M pending). U vs M highlights:
  - Leaders stable across both; U slightly easier for the feature family
    (mahalanobis 0.865 U / 0.788 M; dfm-pca 0.852 / 0.774).
  - **DEEDEE drops most on M** (0.810 U → 0.592 M): adjacent-dim heuristic transfers
    poorly to true multivariate channels.
  - Post-hoc **less inverted on M but still failing** (msp 0.305 U → 0.433 M).
  - **SRS M pending** (univariate-seasonal); shown as `pend.` in the table.
- This **reverses the direction** of the old `main.tex` "TSB-M is easier" claim: with
  finalized data the univariate family is modestly easier for the feature leaders.

## 4. Detectors that LEFT the headline ranking (the 7 class-D exclusions)
`driftlens`, `tdivdm`, `diversify`, `outlier_exposure`, `divoe`, `diversemix`,
`ae_adwin_lstm`. Prior `main.tex` featured **DriftLens (0.857)** and **TD-IVDM
(0.852)** as co-leaders and cited DiverseMix/OE/DivOE throughout. These must be
removed from the abstract, contributions, Results, Discussion, and Conclusion and
pointed instead to the class-D exclusion table (`tab:classd`), the class-D appendix
results (`tab:class_d_appendix`), and `CLASS_D_EXCLUSIONS.md`. Class-D appendix
outcome (U-only, favourable arms): all cluster **near/below chance (0.398–0.662)**,
**no winner**; M arms pending.

## 5. Narrative reversals (the honest corrections)
1. **Mahalanobis is the SOLE top method.** The prior Mahalanobis/DriftLens tie was a
   relabelling artifact (per-sample DriftLens = PCA-Mahalanobis, ρ≈0.999); DriftLens
   excluded and the within-class-scatter/tied-covariance estimation fixed.
2. **SRS now falls BELOW chance (0.352), reversing the "faithful top TS method"
   claim.** Restored to STL + conditional-VAE seasonal-ratio form, it does not
   survive the abrupt two-source streaming boundary.
3. **CatSight / InvAD / DEEDEE rank high only after fixes.** Previously inverted or
   failing (catsight 0.221, invad 0.432); now top-tier (0.735 / 0.740 / 0.700) after
   orientation (CatSight negated distance), reduction (InvAD → feature-space
   statistic on the identity branch), and variant (DEEDEE adjacent-dim + isolation
   forest) corrections. Report as adaptations per `app:fidelity`.
4. **Faithful DICE (energy) fails with the post-hoc family (0.313).** Repairing the
   post-hoc code does not rescue it — the failure is real, not a bug.

## 6. Positive framing added per coordinator (equal/greater weight)
The drafts now foreground the **constructive** result: a clear winner and
recommendation (Mahalanobis 0.826 / DFM-PCA 0.812), a positive mechanism
(direct ID-manifold geometry/density → robust to overconfidence; no auxiliary data,
no retraining, cheap at ~1–4 ms; deedee ~1.0 ms, mahalanobis ~1.9 ms), category-level
deployment guidance (strongest on STABLE/DRIFT), the fidelity audit + fixes as a
reusable methodology, and the streaming extension of Gungor2025 across 527 datasets.
A "Key findings and takeaways" bullet list (7 items) opens the Discussion. The
below-chance analysis is retained but balanced by the affirmative findings.

## 7. `main.tex` claims now STALE (must be edited to reconcile)
- Abstract: "twenty-two OOD detectors", "36 datasets and 772 runs", "Mahalanobis
  (0.874) and DriftLens (0.857) lead", "chi2 = 61.15, p = 8.5e-6" → 17 detectors /
  527 datasets (260 U + 267 M) / 8668 runs / Mahalanobis (0.826) & DFM-PCA (0.812) /
  chi2 = 1674.97, p < 1e-16, N=250 complete-case.
- Intro contribution bullets: drop DriftLens/TD-IVDM as co-leaders; reframe audit
  tally as 24 audited → 17 evaluated + 7 excluded; Pareto default = Mahalanobis +
  DFM-PCA.
- §Results / §Discussion / §Conclusion / Appendix (audit table, ablation top-5,
  reproducibility counts) all carry old numbers/methods and need the same edits.

## 8. Items intentionally kept
The overconfidence mechanism + normalisation dichotomy, the single-winning-principle
(feature-space distance/density) framing, the verify-before-experiment gate as a
first-class contribution, and the Gungor2025 extension claim — all still supported,
now more strongly, by the finalized combined numbers.

## 9. Asset notes
- Tables `\input` by the drafts (paper/tables/, now regenerated to combined U+M):
  `tsb_main.tex` (`tab:tsb_main`, 17 detectors, adds N column),
  `tsb_by_split.tex` (`tab:tsb_split`, **both** U and M columns),
  `class_d_appendix.tex` (`tab:class_d_appendix`, class-D arm results). The
  7-exclusion table `tab:classd` is written inline in `results_draft.tex`.
- Figures referenced (all exist as PNG/PDF in `paper/figures/`):
  `tsb_heatmap_category.png`, `tsb_split_bar.png`, `tsb_norm_dichotomy.png`,
  `tsb_efficiency.png`. **Caveat:** captions were rewritten to the finalized combined
  numbers; verify the figure images themselves were regenerated from the combined
  run (they are auto-generated assets) — if any still show interim values, regenerate
  before final submission.
- `ablation_deltas.tex` / `ablation_delta.png` (referenced by the old
  `main.tex`/`appendix.tex`) are **not** required by these drafts; if the
  original-vs-corrected ablation is retained, regenerate those assets from the
  combined run.
