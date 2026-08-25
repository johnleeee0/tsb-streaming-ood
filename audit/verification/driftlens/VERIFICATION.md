# DriftLens Faithfulness Verification — ADAPTATION (relabel to "PCA-Mahalanobis" is warranted)

**Method id:** `driftlens` · **Paper:** Greco, Vacchetti, Apiletti & Cerquitelli, *Unsupervised Concept
Drift Detection from Deep Learning Representations in Real-time*, arXiv:2406.17813 (a TKDE version is
also cited by the repo)
**Official code:** `https://github.com/grecosalvatore/drift-lens`
**Implementation:** `benchmark1/models/ood_methods/driftlens.py` (`DriftLensDetector`)
**Verified:** 2026-08-20

---

## Verdict

**ADAPTATION.** The docstring does disclose the substitution — `:12` states *"For per-sample scoring,
uses squared Mahalanobis distance to the baseline distribution."* That single line is accurate and it is
the operative behaviour.

**But it is buried under a headline that claims otherwise**, and the claim the prompt asked me to check
is confirmed quantitatively:

- **Official DriftLens is window-level.** `driftlens.py:267` defines
  `compute_window_distribution_distances(E_w, Y_w)`, returning *"the per-batch
  (window_distribution_distances_dict[batch]) and the per-label … distribution distances computed for the
  passed window"*. Every available metric — Fréchet, Mahalanobis, KL, Bhattacharyya, Jensen-Shannon —
  is a **distribution-to-distribution** distance. **There is no per-sample score anywhere in the
  official code.** Even the official's Mahalanobis option
  (`distribution_distances/mahalanobis_drift_distance.py:28`) is `mahalanobis_distance(mu_x, mu_y, sigma_x)`
  — between two *means*.
- **Mine is per-sample squared Mahalanobis** (`:216-235`), which is why it ties the `mahalanobis`
  detector: mean Spearman **0.9294**, **median 0.9990**, **24/40 datasets at ρ > 0.99**, mean
  |ΔAUROC| **0.0241** (§3).
- **The Fréchet machinery is dead code.** `frechet_distance()` (`:37-71`) and `score_batch()`
  (`:242-279`) are implemented but never reached — the registry calls `score()`. The `threshold`
  computed in `fit()` (`:134-187`) is likewise **never used**.

So `driftlens` should be reported as a **PCA-Mahalanobis variant**, exactly as the tracker proposes.

**Also: the paper title is wrong in both the tracker and the code.** The actual title is *"Unsupervised
Concept Drift Detection from Deep Learning Representations in Real-time"*. The tracker says *"DriftLens:
Unsupervised Concept Drift Detection in Deep Learning Embeddings"*; `driftlens.py:4` says *"DriftLens: A
Concept Drift Detection Framework for Deep Learning"*. Seventh title error in this audit, and the first
where the code is wrong too.

---

## 1. Source accessibility

| Source | Status |
|---|---|
| `methods/driftlens/reference/` | Present and intact: `origin = https://github.com/grecosalvatore/drift-lens`, commit `0b7b943b128f8e23b56e5cd56fa40bc3dd35119e` (2026-02-11, *"Added TKDE citation"*), `HEAD → refs/heads/main`. Contains `driftlens/driftlens.py`, `_baseline.py`, `_threshold.py`, and `distribution_distances/`. |
| `arxiv.org/abs/2406.17813v2` | Fetched live — title, authors and abstract obtained. |
| `raw.githubusercontent.com/.../README.md` | **Live fetch FAILED** (`socket hang up`). |

The code comparison rests on the provenance-verified local clone, read directly. The paper's abstract
confirms the per-label emphasis (*"characterizes drift by analyzing and explaining its impact on each
label"*) but does not state granularity; the official source settles that unambiguously and is the
right authority for an implementation diff. Stated rather than implied: I did not re-fetch the official
source online.

### What the official code does

**Granularity** (`driftlens/driftlens.py:267-296`): `compute_window_distribution_distances` takes a
window `E_w` with labels `Y_w` and dispatches to one of five distribution distances, returning a dict
keyed by `[batch]` and `[per-label][label]`.

**Fréchet** (`distribution_distances/frechet_drift_distance.py:53`):

```python
return np.linalg.norm(mu_x - mu_y) + np.trace(sigma_x + sigma_y - 2*matrix_sqrt(sigma_x @ sigma_y))
```

**Mahalanobis** (`distribution_distances/mahalanobis_drift_distance.py:28-44`): distance between
`mu_x` and `mu_y` under `sigma_x` — again distribution-level.

**Two separate PCA budgets**: `batch_n_pc` and `per_label_n_pc` (`driftlens.py:29`, `:50`, `:69`).

## 2. Divergence table

| Component | Official | Mine (`driftlens.py:line`) | Changes results? |
|---|---|---|---|
| **Granularity** | **window/batch-level** distribution distance (`ref driftlens.py:267`) | **per-sample** score (`:216-235`) | **YES — defining change** |
| **Statistic** | Fréchet (or KL / Bhattacharyya / JS / distribution-Mahalanobis) | **squared Mahalanobis to a global baseline** (`:229-233`) | **YES** |
| **Per-label decomposition** | central to the method (`per_label_n_pc`, `[per-label][label]`) | **absent** — `y_id` explicitly *"not used"* (`:106`) | **YES** |
| PCA budgets | `batch_n_pc` **and** `per_label_n_pc` | single `n_components=150` (`:89`), capped `min(150, feat_dim, N−1)` (`:117`) | Yes |
| Threshold | estimated and used for drift decisions | estimated (`:134-187`) then **never used** | Dead state; wasted compute |
| Fréchet implementation | `norm(mu_x−mu_y)` — **not squared** (`:53`) | `norm(mu1−mu2) ** 2` (`:58`) — matches the textbook FID formula | **No** — dead code (§3B) |
| `score_batch()` | n/a (this *is* the official behaviour) | implemented (`:242-279`) but **never called** by the registry | Unused |
| Baseline covariance | per-label + batch estimates | single global `np.cov` + `1e-6·I` (`:123-126`) | Yes |
| Orientation | drift distance, higher = more drift | higher = OOD (`:235`) | **No** |
| Reproducibility | n/a | `np.random.choice` unseeded in `_estimate_threshold` (`:154`, `:157`) | Only affects the unused threshold |

## 3. Empirical findings

**(A) `driftlens` is a Mahalanobis variant — confirmed.** Against the `mahalanobis` detector across 40
paired datasets:

| Metric | Value |
|---|---|
| mean Spearman | **0.9294** |
| **median Spearman** | **0.9990** |
| datasets with ρ > 0.99 | **24 / 40** |
| datasets with ρ > 0.95 | 27 / 40 |
| mean \|ΔAUROC\| | **0.0241** |

Numerous datasets are identical to three decimals — e.g. `TSB-M-OOD_008` (0.991 vs 0.991, ρ = 1.0000),
`TSB-M-OOD_031` (1.000 vs 1.000, ρ = 1.0000), `TSB-M-OOD_020` (0.556 vs 0.556, ρ = 1.0000). The
residual differences come from `driftlens` using a **single global** mean/covariance in PCA space while
`mahalanobis` uses **class-conditional** means with a min over classes.

Correlation with `dfm_pca` is also high (mean ρ = **0.8064**), consistent with all three being
feature-space distance methods on PCA-reduced embeddings.

**(B) My Fréchet is more correct than the official's — but it is dead code.** The official omits the
square on the mean term:

| | Value |
|---|---|
| official (`norm`, unsquared) | 22.203099 |
| mine (`norm ** 2`) | 34.583485 |
| `‖μ₁−μ₂‖` / `‖μ₁−μ₂‖²` | 4.053925 / 16.434311 |

The standard Fréchet/FID definition squares the mean term, so `driftlens.py:58` matches the textbook and
`frechet_drift_distance.py:53` does not. This is worth noting as an upstream discrepancy, but it has
**no effect on results** because `score()` never calls `frechet_distance` — and it would be wrong to
"fix" the local version toward the official here.

**(C) Saved scores.**

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 40 | **0.8440** | — |
| TSB-U (univariate) | 21 | **0.8194** | 2/21 |

Scores span `[3.106, 2.327e+13]`, all non-negative as a squared Mahalanobis requires. The `2.3e13`
extreme is the pathological-feature-magnitude family recorded in
`methods/energy_ebo/VERIFICATION.md` §3.

**(D) A leaderboard redundancy worth flagging.** With `mahalanobis` measured at **ALL 0.8598 / TSB-U
0.8545**, the top of the benchmark is:

| Rank | Method | ALL (n=40) |
|---|---|---|
| 1 | `mahalanobis` | 0.8598 |
| 2 | **`driftlens`** | **0.8440** |
| 3 | `tdivdm` | 0.8378 |

All three are feature-space Mahalanobis/density detectors, and #1 and #2 are near-duplicates
(median ρ = 0.999). Reporting them as three independent methods overstates the diversity of the
leading results — particularly since `mahalanobis` itself carries the tied-covariance defect recorded
in `methods/mahalanobis_mds/VERIFICATION.md`.

## 4. Recommendations

1. **Relabel to "PCA-Mahalanobis"** in the registry and results tables, as the tracker proposes. The
   detector is a per-sample squared Mahalanobis distance in PCA space to a single global ID baseline.
2. **Fix the docstring headline.** `:2` announces *"Drift detection using Fréchet Distance in embedding
   space"* and `:8-10` presents the Fréchet formula as the method; only `:12` reveals that scoring is
   Mahalanobis. Lead with the actual behaviour.
3. **Remove or clearly mark the dead code** — `frechet_distance()` (`:37-71`), `score_batch()`
   (`:242-279`), and the entire `_estimate_threshold` path (`:129`, `:134-187`) whose `threshold` is
   never read. The threshold estimation runs a full Fréchet sampling loop on every `fit()` for no
   effect, and uses unseeded RNG.
4. **State the redundancy with `mahalanobis`** (median ρ = 0.999) in the thesis, and consider reporting
   one of them, or an explicit ablation, rather than both as independent top performers.
5. **Fix the paper title** in both `driftlens.py:4` and the tracker; cite arXiv:2406.17813 (and the TKDE
   version the repo now references).
6. **Optionally implement the real method** as a separate variant: `score_batch()` already computes the
   window-level Fréchet distance, so a batch-level DriftLens is close at hand — though it would need a
   window-level evaluation protocol, which the current per-window-sample AUROC setup does not provide.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded
across `methods/*/VERIFICATION.md`. `driftlens` covers **21** univariate datasets; see
`methods/diversify/VERIFICATION.md` §4 and `methods/catsight/VERIFICATION.md` §4 for the candidate
explanation of the "18" figure.

## 5. Conclusion

The official DriftLens computes one distribution distance per window, with a per-label decomposition,
over five selectable metrics — none of which produces a per-sample score. This implementation instead
returns a per-sample squared Mahalanobis distance to a single global PCA-space baseline, ignores labels,
and leaves its Fréchet machinery, batch scorer and threshold estimation entirely unused. That
substitution is disclosed at `:12`, so the ADAPTATION label is defensible, but the docstring headline
and the dead Fréchet code both imply a method that is not being run. The consequence is measurable: the
detector ties the `mahalanobis` baseline at median Spearman 0.999 across 40 datasets with mean AUROC
differing by 0.024, and the two occupy the top two places in the benchmark. It should be reported as a
PCA-Mahalanobis variant, not as DriftLens.

## CLASS-D BUILD (2026-08-21)

A faithful, batch-level DriftLens was built as a **separate Class-D appendix study** (it does not enter
the 17-method fair-comparison leaderboard). This restores the method's *native granularity* — the very
thing recommendation §6 above called for.

- **File:** `methods/driftlens/classd/driftlens_classd.py` (`DriftLensClassD`); orchestrated by
  `experiments/run_class_d.py` (registry entry `driftlens`, `eval_mode="batch_level"`).
- **Method (faithful):** offline baseline = PCA on ID (Source-1 normal) frozen-backbone embeddings →
  `(mu_b, Sigma_b)` in PCA space. Per monitoring batch of B consecutive windows: embed → PCA-transform →
  `(mu_w, Sigma_w)` → **Fréchet (Wasserstein-2) distance** using the official drift-lens formula
  `||mu_x-mu_y|| + trace(Sigma_x+Sigma_y-2·sqrtm(Sigma_x·Sigma_y))`
  (`reference/.../frechet_drift_distance.py:53`). Higher = more OOD. Runs on the **shared frozen
  backbone** (no training).
- **Eval mode:** batch-level. Monitoring batches of B consecutive non-overlapping windows on the ordered
  stream from `load_tsb(ordered_eval=True)`; batch label = 1 iff frac(OOD windows) ≥ τ (τ=0.5); AUROC
  over batches via `batch_level_auroc`. B=32 (U) / 16 (M) per `CLASS_D_DECISIONS.md`, adaptively shrunk
  only when a short stream cannot otherwise yield ≥2 mixed batches (the effective B is logged).
- **Deliberately NO per-sample proxy** — it duplicates `mahalanobis` (median ρ 0.999), so only the
  batch-level metric is emitted.
- **Honest caveats:** with few OOD windows per file there are few positive *batches*, so batch AUROC is
  noisy (observed range 0.0–1.0 on a 2-per-cell U smoke); and files whose ordered stream is almost all
  OOD (e.g. `TSB-U-STABLE_003`, 25/27 OOD) cannot form two mixed batches → batch AUROC is undefined
  (recorded as NaN, `n_eval=0`) — a genuine protocol-incompatibility, not a bug.
- **Smoke (TSB_N_PER_CELL=2, U only):** finite batch AUROCs written to `results/class_d_group2.csv`
  (e.g. OOD_002 = 1.0, DRIFT_004 = 0.25, STABLE_002 = 0.333). No production file was modified.
