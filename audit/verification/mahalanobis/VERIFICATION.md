# Mahalanobis (MDS) Faithfulness Verification — FAITHFUL (fixed 2026-08-21; within-class scatter; see FIX APPLIED)

**Method id:** `mahalanobis_mds` · **Paper:** Lee, Lee, Lee & Shin, *A Simple Unified Framework for
Detecting Out-of-Distribution Samples and Adversarial Attacks*, NeurIPS 2018 (arXiv:1807.03888)
**Core-paper target:** Gungor, Rios, Ahuja & Rosing, *TS-OOD: Evaluating Time-Series Out-of-Distribution
Detection and Prospective Directions for Progress*, AAAI-25 AI4TS (arXiv:2502.15901)
**Implementation under test:** `benchmark1/models/ood_methods/mahalanobis.py` (`MahalanobisDetector`)
**Verified:** 2026-08-19

---

## Verdict

**CORRECTED — fix required. No `_enh` variant currently exists, so the benchmark's present MDS
numbers were produced with the defective covariance.**

The method's structure is right: class-conditional Gaussians on pre-logit features, a single shared
covariance, and the minimum Mahalanobis distance as the OOD score. The two omissions relative to Lee
et al. (FGSM input perturbation, multi-layer logistic-regression ensemble) are confirmed as consistent
with the TS-OOD target, not accidental.

**But the tied covariance is computed incorrectly.** The official estimator (and Lee et al. Eq. 1, and
TS-OOD's explicit "tied covariance") is the **within-class** scatter — each sample centered on *its own
class mean* before pooling. `mahalanobis.py` pools **raw** features and lets `EmpiricalCovariance`
center on the **global** mean, yielding the **total** covariance `Σ_T = Σ_W + Σ_B`. This is a different
metric, it is **not** rank-preserving, and it systematically degrades the score. It diverges from Lee
et al., from the official code, and from TS-OOD alike.

---

## 1. Source accessibility

| Source | Status |
|---|---|
| Local clone `methods/mahalanobis_mds/reference/` | Present. `origin = https://github.com/pokaxpoka/deep_Mahalanobis_detector`, commit `90c2105e78c6f76a2801fc4c1cb1b84f4ff9af63` (2019-08-13) |
| `https://raw.githubusercontent.com/pokaxpoka/deep_Mahalanobis_detector/master/lib_generation.py` | Fetched live; matches the local clone verbatim |
| `https://ar5iv.labs.arxiv.org/html/1807.03888` (Lee et al. full text) | Fetched live |
| `https://ar5iv.labs.arxiv.org/html/2502.15901` (TS-OOD full text) | Fetched live |
| `https://github.com/kaustubhsridhar/TS-OOD` | **UNREACHABLE — HTTP 404.** The URL recorded in `tracker/PASS1_VERIFICATION_PROMPTS.md` is wrong. TS-OOD was located instead as **arXiv:2502.15901**; no official code repository for it was found. The TS-OOD cross-check below rests on the paper text only. |
| Cached features / backbone checkpoints | **NONE.** `experiments/*/*/mahalanobis/` holds only `scores.npy`, `labels.npy`, `results.json`. The real-data impact of the covariance defect therefore **could not be measured** — see §5. |

### What the official code does

`lib_generation.py:107-120` (`sample_estimator`), confirmed identical in the live fetch:

```python
for k in range(num_output):
    X = 0
    for i in range(num_classes):
        if i == 0:
            X = list_features[k][i] - sample_class_mean[k][i]
        else:
            X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
    group_lasso.fit(X.cpu().numpy())
    temp_precision = group_lasso.precision_
```

Each class's features are centered on **that class's own mean** (`- sample_class_mean[k][i]`) before
concatenation. The estimator is `sklearn.covariance.EmpiricalCovariance(assume_centered=False)`
(`:54`). The re-centering is exactly harmless here: within-class deviations sum to zero per class, so
the pooled matrix has an exactly zero mean.

Scoring, `lib_generation.py:155,162,194`:

```python
term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
...
noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
```

### What the papers say

**Lee et al., Eq. (1)** — tied covariance as averaged within-class scatter:

```
μ̂_c = (1/N_c) Σ_{i:y_i=c} f(x_i)
Σ̂   = (1/N) Σ_c Σ_{i:y_i=c} (f(x_i) − μ̂_c)(f(x_i) − μ̂_c)ᵀ
```

Eq. (2): `M(x) = max_c −(f(x) − μ̂_c)ᵀ Σ̂⁻¹ (f(x) − μ̂_c)`. Appendix A: *"all classes share the same
covariance matrix"* (the LDA assumption).

**TS-OOD (arXiv:2502.15901):**

- *"For that, they use ID class conditioned Gaussians and a **tied covariance** to then compute the OOD score via the Mahalanobis distance."*
- *"We experimented with extracting features from several depths of the backbones tested and obtained on average superior results using the pre-logit layer."*
- *"Note that for multivariate time-series, due to the absence of large-scale (foundation-like) DNNs, we train the backbone with the ID class samples only."*

## 2. Divergence table

| Component | Original (Lee et al. / official code) | Mine (`file:line`) | Changes results? |
|---|---|---|---|
| Class-conditional means | `μ̂_c` per class (`lib_generation.py:102-104`) | `feats_class.mean(axis=0)` (`mahalanobis.py:74`) | **No** |
| **Tied covariance** | **Within-class scatter**: pool *class-centered* features (`lib_generation.py:112-114`), Lee Eq. 1 | **Total covariance**: pools **raw** features (`mahalanobis.py:75, 78`) and `EmpiricalCovariance().fit(...)` centers on the **global** mean (`:81-82`) | **YES — `Σ_T = Σ_W + Σ_B`. Not rank-preserving. See §5.** |
| Precision matrix | `group_lasso.precision_` (`:118`) | `np.linalg.inv(cov)`, `pinv` fallback (`:89-92`) | **No** — equivalent given the same `cov` |
| Covariance regularisation | none | `+ 1e-6·I` ridge (`:85`) | Negligible; aids invertibility on small TS sets |
| Score statistic | `max_c −½·d²_c` (`:155,194`), higher = ID | `min_c √(d²_c)` (`:130-131,139`), higher = OOD | **No** — monotone-equivalent: `min_c √(d²) = √(min_c d²)` and `max_c(−½d²) = −½·min_c d²` |
| Feature layer | multi-layer; per-layer `precision[layer_index]` | single pre-logit via `_forward_features` (`:54,112`) | **No** vs TS-OOD (prescribed); documented simplification vs Lee |
| FGSM input perturbation | present (`lib_generation.py:169-179`) | **absent** | **No** vs TS-OOD (not applied there); documented simplification vs Lee |
| Multi-layer LR ensemble | `OOD_Regression_Mahalanobis.py` fits logistic regression over layers | **absent** | See §4 — partially confirmed against TS-OOD |
| Label requirement | supervised class means | raises if `y_id is None` (`:46-50`) | **No** — correct |
| Spatial pooling | `torch.mean(out_features, 2)` over spatial dims (`:75`) | n/a — TS embedding is already 1-D | **No** — domain-appropriate |

## 3. The covariance defect in detail

The two estimators differ by the between-class scatter:

| Estimator | Formula | What it measures |
|---|---|---|
| Official (`Σ_W`) | `(1/N) Σ_c Σ_{i∈c} (x_i − μ_c)(x_i − μ_c)ᵀ` | within-class spread only |
| Mine (`Σ_T`) | `(1/N) Σ_i (x_i − μ_global)(x_i − μ_global)ᵀ` | within-class **+** between-class spread |

Because `Σ_T = Σ_W + Σ_B`, inverting `Σ_T` **shrinks precisely the directions along which the class
means separate** — the discriminative directions the method depends on. The distortion vanishes only
when `Σ_B = 0` (all class means coincide) and grows with class separation.

### Minimal fix (3 lines, `mahalanobis.py`)

Replace the raw-feature pooling at `:75` / `:78` with class-centered pooling:

```python
# in the per-class loop, replace:  all_feats_list.append(feats_class)
all_feats_list.append(feats_class - self.class_means[class_label])
```

`class_means[class_label]` is already computed on the preceding line (`:74`), so the change is local.
Everything downstream (`:81-92`) is unchanged and remains correct.

**I have not applied this fix** — it would change every saved MDS score, so the existing
`experiments/*/*/mahalanobis/` results would need regeneration. That is your call.

## 4. Are the two omissions prescribed, or accidental?

| Omission | TS-OOD evidence | Assessment |
|---|---|---|
| Pre-logit single layer | *"obtained on average superior results using the pre-logit layer"* | **Confirmed prescribed.** |
| FGSM input perturbation | TS-OOD never applies or mentions input perturbation for MDS | **Confirmed consistent** — no perturbation step exists in the target protocol. |
| Multi-layer LR ensemble | TS-OOD states the pre-logit layer is preferred but does **not** explicitly say the logistic-regression combiner is dropped | **Partially confirmed.** Using one layer makes the combiner vacuous (there is nothing to weight), so the omission follows from the layer choice rather than from an explicit statement. Honest wording for the thesis: implied by the single-layer choice, not separately stated. |

The tied covariance is **not** in this category: TS-OOD explicitly prescribes *"a tied covariance"*, so
the total-covariance substitution has no sanction from either reference.

## 5. Empirical status of the defect

**Real-data magnitude: not measurable.** No cached features or backbone checkpoints exist anywhere in
the repo, so the covariance cannot be recomputed on the actual benchmark. No estimate of the real-data
effect is offered here.

**Mechanism and direction: demonstrated on synthetic data** (32-dim features, K=4 classes, 12 seeds,
OOD = inflated-variance samples on the same manifold; both estimators otherwise identical including
the `1e-6` ridge):

| Class separation | AUROC (within, official) | AUROC (total, mine) | Δ | Spearman ρ |
|---|---|---|---|---|
| 0.0 | 0.9325 | 0.9320 | −0.0005 | 0.9996 |
| 2.0 | 0.9298 | 0.9262 | −0.0036 | 0.9842 |
| 5.0 | 0.9307 | 0.9243 | −0.0064 | 0.9774 |
| 10.0 | 0.9307 | 0.9231 | −0.0076 | 0.9749 |

Reading:

- At zero class separation the two coincide (Δ ≈ 0, ρ ≈ 1.0), exactly as `Σ_B = 0` predicts — this
  validates the test setup.
- As separation grows, the total-covariance variant is **monotonically worse**, and Spearman ρ falls
  to ≈ 0.975, confirming the two produce **materially different rankings**. The divergence is therefore
  **not** metric-invariant.
- In this synthetic regime the AUROC penalty is small (≤ 0.008). It is consistently negative — the
  defect costs accuracy, never gains it — but nothing here licenses extrapolating that magnitude to
  the real benchmark.

## 6. Correction to the prior notes

`discrepancy_report.md:19` states: *"Reference fits a single pooled `EmpiricalCovariance` over
class-centered features and uses `precision_`. Local fits `EmpiricalCovariance` over pooled features,
adds `1e-6·I` ridge, then inverts... **Same estimator**; local adds light regularisation"* — graded
**MINOR / MINOR**.

The observation was correct: the note *did* spot that the reference centers per class and the local
code does not. The **grading is wrong**. These are not the same estimator — they are `Σ_W` versus
`Σ_T = Σ_W + Σ_B`. The note attributes the difference to "light regularisation" (the ridge), when the
substantive difference is the missing per-class centering. The row should be regraded
**MAJOR / MAJOR**, since it diverges from Lee et al., the official code, and TS-OOD's explicit "tied
covariance". The summary's conclusion that *"no corrective `_enh` variant is required"* does not hold.

## 7. Minor observations (no action required)

- `mahalanobis.py:118-139` scores with a nested Python loop over samples × classes, recomputing
  `delta @ P @ delta` per pair. This is `O(N·K)` interpreted operations and is fully vectorisable
  (`einsum` over a stacked delta tensor). Performance only — the arithmetic is correct.
- `np.sqrt(max(0, ...))` at `:131` guards against small negative values from numerical error. Sound.
- The official's `EmpiricalCovariance(assume_centered=False)` on already class-centered data
  re-subtracts a mean that is exactly zero by construction, so it is a genuine no-op — the official
  estimator is exactly Lee Eq. (1).

## 8. Conclusion

MDS is structurally correct and its two documented simplifications are legitimate for the TS-OOD
target, but the tied covariance is computed as the total covariance rather than the within-class
scatter. This contradicts Lee et al. Eq. (1), the official `sample_estimator`, and TS-OOD's explicit
prescription; it is not rank-preserving and it systematically degrades the score. The fix is three
lines. Until it is applied and the MDS results regenerated, the benchmark's Mahalanobis numbers should
be reported as produced by a total-covariance variant, not as faithful MDS.

**Open item shared with the other verifications:** the 18-vs-21 univariate dataset-count discrepancy
recorded in `methods/msp/VERIFICATION.md` §5, `methods/odin/VERIFICATION.md` §7 and
`methods/energy_ebo/VERIFICATION.md` §6 remains unresolved.

---

## FIX APPLIED (2026-08-20)

**File changed:** `benchmark1/models/ood_methods/mahalanobis.py` (`MahalanobisDetector.fit`, around
the per-class loop at `:75`). No other file was touched. The class name (`MahalanobisDetector`), the
registry keys (`mahalanobis`, `mds`), and the `BaseOODDetector` `fit(x_id, y_id)` / `score(x)`
interface are all unchanged.

### What changed — covariance estimator

**Old (defective) — total covariance `Σ_T`:**

```python
self.class_means[class_label] = feats_class.mean(axis=0)
all_feats_list.append(feats_class)                      # raw features pooled
...
all_feats_concat = np.concatenate(all_feats_list, axis=0)
cov = EmpiricalCovariance().fit(all_feats_concat).covariance_   # centers on GLOBAL mean
```

This pooled **raw** features, so `EmpiricalCovariance` centered on the global mean and produced
`Σ_T = Σ_W + Σ_B` (within-class **plus** between-class scatter).

**New (faithful) — within-class scatter `Σ_W`:**

```python
self.class_means[class_label] = feats_class.mean(axis=0)
all_feats_list.append(feats_class - self.class_means[class_label])   # class-centered
...
all_feats_concat = np.concatenate(all_feats_list, axis=0)
cov = EmpiricalCovariance().fit(all_feats_concat).covariance_        # pooled deviations, zero mean
```

Each class's features are now centered on **its own mean** before pooling, so the tied covariance is
the within-class scatter `Σ_W`, matching Lee et al. 2018 Eq. (1) and the official `sample_estimator`
(`reference/lib_generation.py:112-114`), which pool `list_features[k][i] - sample_class_mean[k][i]`.
Because the per-class deviations sum to zero, the pooled matrix already has an exactly zero mean, so
the subsequent `EmpiricalCovariance().fit(...)` re-centering is the intended no-op. Everything
downstream (the `1e-6·I` ridge, the inverse/pinv fallback, and the entire `score()` method) is
unchanged and was already correct.

**Score unchanged and confirmed:** `score()` returns `min_c √((x − μ_c)ᵀ Σ_W⁻¹ (x − μ_c))`, the
minimum Mahalanobis distance over ID classes, with **higher = OOD** (monotone-equivalent to Lee et
al.'s `max_c −½·d²_c`).

### Smoke test (venv `C:\THESIS\.venv\Scripts\python.exe`)

A dummy `torch.nn.Module` linear backbone (8→16) plus a linear classifier head (16→4) passed via
`config["classifier"]` was used to instantiate `MahalanobisDetector`, `fit()` on 40 random ID samples
with 4 class labels (with class separation), then `score()` on 25 random test samples:

```
PASS: fit + score succeeded
  n classes fitted        : 4
  precision matrix shape   : (16, 16)
  score() output length    : 25 (expected 25)
  all finite               : True
  score range              : [6.7354, 19.6062]
  ID-like sample score     : 0.0004
  far-OOD sample score     : 134.6894
  OOD > ID (higher=OOD)    : True
```

Outputs are finite, correct length, non-negative, and orientation is correct (a far sample scores far
higher than an ID-centroid sample). The full benchmark was **not** run.

### New verdict: **FAITHFUL**

With the within-class scatter now used for the tied covariance, the implementation matches Lee et al.
2018 Eq. (1), the official `sample_estimator`, and TS-OOD's explicit "tied covariance" prescription.
The two remaining simplifications (single pre-logit layer, no FGSM perturbation, no multi-layer LR
ensemble) are confirmed consistent with the TS-OOD target (§4 above), not defects. The detector is
therefore a **faithful** reproduction of MDS for the TS-OOD protocol.

**Regeneration note (unchanged from §3):** this fix changes every saved MDS score, so all
`experiments/*/*/mahalanobis/` results must be regenerated and the leaderboard re-checked. Because
`driftlens` ties `mahalanobis` at ρ≈0.999 (per-sample PCA-Mahalanobis), evaluate the two together
after regeneration.
