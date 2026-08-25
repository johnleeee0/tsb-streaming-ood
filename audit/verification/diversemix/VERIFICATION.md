# DiverseMix Faithfulness Verification — NOT-THE-METHOD (no auxiliary outlier set to diversify; at chance on real data in either orientation)

**Method id:** `diversemix` (+ `diversemix_enh`) · **Paper:** Yao, Han, Fu, Peng, Hu & Zhang,
*Out-Of-Distribution Detection with Diversification (Provably)*, NeurIPS 2024 (arXiv:2411.14049)
**Official code:** `https://github.com/HaiyunYao/diverseMix`
**Implementation:** `benchmark1/models/ood_methods/diversemix.py` (`DiverseMixDetector`)
**Corrected variant:** `methods/diversemix/diversemix_enh/diversemix_enh.py`
**Verified:** 2026-08-20

---

## Verdict

**NOT-THE-METHOD.** DiverseMix's contribution is *"enhanc[ing] the diversity of the **auxiliary outlier
set**"* — it takes a real auxiliary outlier corpus (the official repo uses **ImageNet64x64**) and mixes
within it. **This implementation has no auxiliary outlier data at all.** It fabricates pseudo-outliers
by convex-combining ID features from two different classes (`:204-212`), so there is nothing to
diversify.

**The fabricated "outliers" are not outliers — they are more central than real ID samples.** Measured on
matched synthetic data: median Mahalanobis distance to the ID mean is **4.229** for the pseudo-outliers
versus **5.637** for genuine ID samples, and **100%** of them fall inside the ID 95th-percentile shell
(§3B). Mixing two ID classes lands you inside the ID convex hull, near its centre. The auxiliary loss
`relu(logsumexp + 1)` (`:155`) therefore trains the energy head to assign *low* energy to the **interior
of the ID distribution** — the opposite of its intent.

**Four corrections to the tracker's framing:**

1. **There is a canonical repo.** The tracker says *"no single canonical repo (codebase hybrid)"*.
   `https://github.com/HaiyunYao/diverseMix` is the official NeurIPS 2024 release
   (`train_diverseMix.py`, `eval_ood_detection.py`, built on ATOM/OpenOOD).
2. **The official orientation is `diversemix_enh`'s, not "neither".** `eval_ood_detection.py` computes
   `torch.logsumexp(outputs, dim=1)` as an **ID-ness** score — *"higher energy scores indicate
   in-distribution samples, while lower scores suggest out-of-distribution"*. So the OOD score is
   **−logsumexp**, exactly what `diversemix_enh:34` returns. The base variant's `+logsumexp` (`:306`)
   contradicts the official convention **and** its own training objective.
3. **It is not "regime-dependent orientation" — it is a pure sign flip.** On all four shared datasets
   `base + enh = 1.0000` exactly (§3A), because `enh` only negates the identical scores. There is one
   score with two readings, not two methods.
4. **The cited real-data reversal (0.633 → 0.367) is the n=4 ablation subset, not the sweep.** On the
   full 40 datasets the base scores **0.5200** (TSB-U **0.5018**, 9/21 below chance) and the flipped
   orientation therefore **0.4800**. **The detector is at chance in either direction** — so the
   orientation question is moot, and this is a negative result about the detector, not about the sign.

**The paper title in the tracker is wrong; the docstring is right** (`:4-5`). Ninth title error in this
audit.

---

## 1. Source accessibility

| Source | Status |
|---|---|
| `methods/diversemix/reference/` | **ABSENT — directory does not exist.** The folder holds `validation_status.json` and the `diversemix_enh/` subdirectory. |
| `github.com/HaiyunYao/diverseMix` | **Fetched live.** Official NeurIPS 2024 code; main files `train_diverseMix.py`, `eval_ood_detection.py`, `compute_metrics.py`; auxiliary data *"Downsampled ImageNet Datasets … ImageNet64x64"*; evaluation invoked with `--method energy`; acknowledges ATOM and OpenOOD. |
| `raw.githubusercontent.com/.../eval_ood_detection.py` | **Fetched live** — the energy scoring line and its stated orientation (quoted below). |
| `train_diverseMix.py` | **Not fetched.** The loss formulation was not read from the official source, so the training comparison below rests on the paper's abstract-level description plus the local code. Stated rather than implied. |
| Paper (arXiv:2411.14049) | Title, authors and abstract-level method description obtained via search results (secondary). Full text not read. |

### What the official sources establish

- **Purpose:** DiverseMix *"enhances the diversity of auxiliary outlier set for training in an efficient
  way"*, motivated by the observation that *"existing methods struggle to generalize … due to limited
  diversity of auxiliary outliers **collected**"*. The auxiliary outliers are **collected data**, not
  synthesised.
- **Auxiliary data:** ImageNet64x64 (downsampled ImageNet) — a real outlier corpus.
- **Test-time score** (`eval_ood_detection.py`):
  `return torch.logsumexp(outputs,dim=1).float().detach().cpu().numpy()`, with higher = **in**-distribution.

## 2. Divergence table

| Component | Official | Base `diversemix.py:line` | `diversemix_enh.py:line` |
|---|---|---|---|
| **Auxiliary outlier set** | **real corpus** (ImageNet64x64) | **fabricated** by cross-class ID mixing (`:172-219`) — 100% inside the ID distribution (§3B) | inherited (unchanged) |
| **What mixup diversifies** | the collected auxiliary outliers | ID interpolants | inherited |
| Score-adaptive mixup `λ ~ Beta(ŝᵢα, ŝⱼα)` | the paper's mechanism | **implemented** (`:221-273`, formula at `:226`) | inherited |
| Energy head + CE on ID | trained network | `EnergyHead` on frozen features (`:39-58`, `:154`) | inherited |
| Auxiliary loss | pushes outlier energy down | `relu(logsumexp + 1)`, ω=0.5 (`:155-156`) | inherited |
| Backbone | trained end-to-end | **frozen**; only the head trains (`:100-102`, `:110`) | inherited |
| **Test score orientation** | `−logsumexp` (higher = OOD) | **`+logsumexp`** (`:306`) — **contradicts official and own objective** | **`−logsumexp`** (`:34`) — **matches official** |
| Which variant the benchmark ran | — | **40 datasets** | 4 datasets |

## 3. Empirical findings

**(A) The two variants are exact negations, and both are at chance on the sweep.**

| Dataset (shared) | base | enh | sum |
|---|---|---|---|
| TSB-M-DRIFT003 | 0.625 | 0.375 | **1.000** |
| TSB-U-DRIFT024 | 0.585 | 0.415 | **1.000** |
| TSB-U-OOD009 | 0.750 | 0.250 | **1.000** |
| TSB-U-STABLE001 | 0.571 | 0.429 | **1.000** |
| mean (n=4) | 0.6328 | 0.3672 | 1.0000 |

`mean(base + enh) = 1.0000` confirms a pure sign flip on identical magnitudes.

On the full sweep the base variant is **uninformative**:

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| base — All | 40 | **0.5200** | — |
| base — TSB-U | 21 | **0.5018** | 9/21 |
| flipped (= enh orientation) — All | 40 | **0.4800** | — |

So the 4-dataset ablation subset (0.633) is not representative: across 40 datasets the detector sits on
the chance line, and negating it moves it symmetrically to the other side. **Neither orientation
produces a usable detector** — which is the finding, rather than any conclusion about the sign.

**(B) The fabricated auxiliary outliers are inside the ID distribution.** Replicating
`_generate_auxiliary_outliers` (`:204-212`) on matched synthetic data (4 classes, 32-d):

| | median Mahalanobis to ID mean | p95 |
|---|---|---|
| genuine ID samples | 5.637 | 6.645 |
| **synthesised "outliers"** | **4.229** | 5.445 |
| share inside the ID p95 shell | **100.0%** | — |

They are *closer* to the ID centre than ID samples are. This gives a clean mechanistic account of the
base orientation's apparent success on the small ablation subset: because the auxiliary loss drives
**central** features to low energy, genuinely far-out OOD samples end up in a *higher*-energy regime, so
`+logsumexp` ranks them as OOD. That is an artefact of the broken pseudo-outliers, not the paper's
mechanism — and it does not survive the full sweep (0.520).

The source comment at `:298-305` reaches the same empirical observation but attributes it to *"limited
training data (≤82 samples)"*. The real cause is that the auxiliary set consists of ID interpolants.

**(C) Score ranges.** base `[-1.354e+05, 3.161e+05]` (68.8% negative) over 40 datasets;
`diversemix_enh` `[-1.924, 11.91]` over its 4. Not directly comparable, since enh covers a different
(much smaller) subset.

## 4. Documentation and status records

- `diversemix_enh/CHANGES.md` diagnoses the orientation contradiction correctly and precisely, and
  reports synthetic AUROC 0.188 → 0.812. That reasoning is sound and is now **corroborated by the
  official evaluation convention** (§1), which the note did not have access to.
- `validation_status.json` (base) records `status: "FAIL"` with `auroc_gt_0p5: false` and
  `validation_auroc: 0.1513` — yet `discrepancy_count` is `{CRITICAL: 0, MODERATE: 0, MINOR: 0}`. A
  method that fails its own validation check should not report zero discrepancies. Same defect recorded
  for DIVERSIFY and DEEDEE.
- **The benchmark runs the variant that FAILED validation** (40 datasets) while the variant that PASSED
  has 4. That inversion should be resolved explicitly rather than left implicit.

## 5. Recommendations

1. **Report DiverseMix as a negative result and exclude it from the headline table**, or relabel it. With
   mean AUROC 0.5200 across 40 datasets (0.5018 on TSB-U, 9/21 below chance) it carries no signal in
   either orientation.
2. **Adopt `−logsumexp` as the canonical orientation** if the row is kept at all. This is now settled by
   `eval_ood_detection.py`, not a judgement call — and it is `diversemix_enh`'s choice. Note that doing
   so gives 0.4800, i.e. it does not rescue the method.
3. **State plainly that the defining mechanism is absent**: the paper diversifies a *collected* auxiliary
   outlier set; here the "outliers" are ID interpolants that sit closer to the ID centre than ID data.
   Mixup over them cannot implement "informative extrapolation".
4. **Replace the source comment at `:298-305`.** Its empirical observation is right but the explanation
   ("≤82 training samples") is wrong; the cause is the fabricated auxiliary set.
5. **Fix `validation_status.json`** — a FAIL with zero discrepancies, and re-check why the failing
   variant is the one on the sweep.
6. **Fix the tracker**: the paper title is wrong (the docstring is correct), and a canonical repo does
   exist at `HaiyunYao/diverseMix`.
7. **If DiverseMix proper is wanted**, it requires a real auxiliary outlier corpus plus end-to-end
   training — out of scope for a post-hoc benchmark, which is a legitimate reason to drop it rather than
   mislabel it.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded
across `methods/*/VERIFICATION.md`. `diversemix` covers **21** univariate datasets; see
`methods/diversify/VERIFICATION.md` §4 and `methods/catsight/VERIFICATION.md` §4 for the candidate
explanation of the "18" figure.

## 6. Conclusion

The training machinery is reproduced in spirit — an energy head, a `relu(logsumexp + 1)` auxiliary
penalty, and the paper's score-adaptive mixup `λ ~ Beta(ŝᵢα, ŝⱼα)` are all present. But DiverseMix
exists to diversify a *collected* auxiliary outlier set, and none exists here: the auxiliary set is
manufactured by convex-combining ID features from different classes, which places 100% of it inside the
ID distribution and closer to the ID centre than real ID data. With nothing genuinely out-of-distribution
to diversify, the defining mechanism is absent. The orientation question, meanwhile, resolves cleanly
rather than being regime-dependent: the official `eval_ood_detection.py` treats `+logsumexp` as an
ID-ness score, so the OOD score is `−logsumexp` — the `_enh` choice — and the base variant contradicts
both that convention and its own training objective. It is moot in practice, because the two variants
are exact negations (`base + enh = 1.0000`) and the base sits at 0.5200 across 40 datasets. The correct
report is a negative result, with the defining mechanism named as absent.

## CLASS-D BUILD (2026-08-21)

**Faithful DiverseMix build — appendix study, both arms.** The two gaps identified
above (fabricated ID-interior aux set; wrong `+logsumexp` orientation) are now closed
in a *separate* Class-D appendix build that never touches production `ood_methods` or
the 17-method leaderboard.

- **Module:** `methods/diversemix/classd/diversemix_classd.py`
  (`DiverseMixClassD` + `EnergyHead`) — trains an energy head with the
  `relu(logsumexp+1)` auxiliary penalty and score-adaptive Beta mixup
  `λ ~ Beta(ŝ_aux·α, ŝ_id·α)`, and scores with `−logsumexp` (official
  `eval_ood_detection.py` orientation; higher = OOD).
- **Fix 1 — real aux corpus:** the outlier set is now REAL channel-matched hold-out
  TSB windows (`aux_outliers.py`, no leakage), not cross-class ID interpolants. The
  score-adaptive mixup pairs REAL aux outliers with ID samples (ID×aux, not ID×ID),
  per `CLASS_D_DECISIONS.md`.
- **Fix 2 — orientation:** `−logsumexp` throughout (the base `+logsumexp` at
  production `:306` is contradicted by the official convention).
- **Both arms:** `head_only` (backbone frozen, only the energy head trains on frozen
  features — the least-unfair arm) and `full_net` (backbone + head jointly). The
  backbone is a deep copy; the shared frozen anchor is never mutated (verified: arms
  differ on every file).
- **Verification (U split, 2 files/cell, seed 42):** all 12 DiverseMix runs produced
  FINITE per-sample AUROCs → `results/class_d_group1.csv`. Example rows: OOD_001
  head_only 0.7632 / full_net 0.7616; DRIFT cell head_only 0.8056 / full_net 0.7222 —
  clearly off-chance, unlike the 0.52 of the fabricated-corpus base variant.
- **Caveats:** BREAKS the frozen-backbone fair comparison (trains an extra head on an
  external corpus) → appendix only; the head_only arm is the *least* unfair of the
  three Group-I methods. Result quality is gated by aux-corpus quality — if
  `get_aux_windows` falls back to synthetic (no channel-matched file), keep the
  negative-result framing. Tiny test sets make per-file AUROCs high-variance.
