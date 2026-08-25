# SRS Faithfulness Verification — FAITHFUL (fixed 2026-08-21; seasonal ratio restored; see FIX APPLIED)

**Method id:** `srs` · **Paper:** Belkhouja, Yan & Doppa, *Out-of-distribution Detection in Time-series
Domain: A Novel Seasonal Ratio Scoring Approach*, 2022/2023 (arXiv:2207.04306)
**Implementation under test:** `benchmark1/models/ood_methods/srs.py` (`SRSDetector`)
**Verified:** 2026-08-20

---

## Verdict

**ADAPTATION.** The pipeline reproduces SRS's *machinery* — per-class STL pattern extraction, circular
alignment, and two class-conditional VAEs (signal + residual) — but **not its score**. The method's
titular contribution, the seasonal **ratio**, is never computed: `srs.py:452` returns the signal
neg-ELBO alone. The residual CVAE is trained (`:367-376`) and its per-sample neg-ELBO is evaluated
(`:441`) and then **discarded**.

This is one step from NOT-THE-METHOD: what is evaluated is "conditional-VAE reconstruction error on
STL-aligned signals", not Seasonal Ratio Scoring. It is labelled ADAPTATION rather than
NOT-THE-METHOD only because every other stage of the paper's pipeline is present and the missing step
is a one-line restoration.

**The docstring is incorrect and must be fixed.** `srs.py:9-12` claims *"OOD score is the ratio of
their negative ELBOs … we negate for standard OOD convention"* and *"This is a faithful PyTorch port of
the original Keras/TensorFlow implementation."* Neither statement matches the code: there is no ratio
and no negation, and the CVAE is a different generative model (§4).

---

## 1. Source accessibility

| Source | Status |
|---|---|
| Local clone `methods/srs/reference/` | Present. `origin = https://github.com/tahabelkhouja/SRS`, commit `5b0f44b49ff080ed6313fb5a4c352ce1e23bbf96` (2023-04-12, *"Update batch likelihood"*). Contains `STL.py`, `CVAE_Keras.py`, `Run_SRS.py`, `TSDTransform.py`, `OOD_Auroc_utils.py`, bundled `dtw_master`. |
| `https://raw.githubusercontent.com/tahabelkhouja/SRS/main/Run_SRS.py` | Fetched live; **matches the local clone verbatim**. Confirms `ratio1 = ll_x_in/ll_rem_in`, and defaults `epochs=500`, `latent_size=32`, `batch_size=32`, `align_iter=5`, `arch='CONV'`. |
| Paper (arXiv:2207.04306) | **UNREACHABLE.** Both `ar5iv.labs.arxiv.org/html/2207.04306` and `arxiv.org/abs/2207.04306` returned `socket hang up` on two attempts. No paper text was obtained; the comparison below rests entirely on the official code, which was verified against both the local clone and the live repository. Nothing is inferred from the unavailable paper. |

## 2. Divergence table

| Component | Official | Mine (`srs.py:line`) | Changes results? |
|---|---|---|---|
| **Score** | `ratio = ll_signal / ll_residual` (`Run_SRS.py:139,145,178`) | `return neg_elbo_sig` — ratio never formed (`:452`) | **YES — defining mechanism absent** |
| Residual model | used in the ratio denominator | trained (`:367-376`), evaluated (`:441`), **discarded** | **YES** — cost paid, no benefit |
| Likelihood term | reconstruction **only**: `rl = rl - reduce_sum(BCE)` (`CVAE_Keras.py:243-247`) — no KL | neg-**ELBO** = recon + KL (`:213-215`) | **YES** — different quantity |
| Reconstruction likelihood | **Bernoulli/BCE** on min-max-scaled data, **sigmoid** decoder output (`CVAE_Keras.py:98,206`) | **Gaussian/MSE**, **linear** decoder output (`:203,185`) | **YES** — different generative model |
| Input scaling | min–max to [0,1] via `Normalize(min_train,max_train)` (`CVAE_Keras.py:181`) | per-channel z-score (`:331-333`) | **YES** — interacts with the BCE/sigmoid choice |
| Encoder conditioning | label concatenated **in the encoder** (`CVAE_Keras.py:190`) **and** decoder (`:195`) | decoder only (`:179`); encoder unconditional (`:165-171`) | **YES** — not a fully conditional VAE |
| Architecture | 3×Conv1D (16/32/64), stride 1, `padding=same`, no pooling; decoder 3×Conv1DTranspose (`CVAE_Keras.py:184-206`) | 2×Conv1d (16/32) + `AdaptiveAvgPool1d(T//4)` + Linear; decoder Linear×2 + `Upsample` + 2×Conv1d (`:144-163`) | Likely — different capacity and an information bottleneck the official lacks |
| KL scaling | `reduce_mean` over batch **and latent**, ×0.5 (`CVAE_Keras.py:101-103`) | sum over latent, mean over batch (`:204`) | **YES** — official KL is ≈`latent_dim`× weaker |
| Optimizer / schedule | Adam **1e-4**, **500** epochs (`CVAE_Keras.py:223`, `Run_SRS.py:206`) | Adam **1e-3**, **30** epochs (`:292,290`) | Likely — 10× LR, 16.7× fewer epochs |
| STL input | one `stldecompose.decompose` per class over the **concatenated** class data, `period=SEG_SIZE` (`STL.py:42`) | `statsmodels.STL` **per sample**, `period=T//4` heuristic (`:52-57,105-107`) | **YES** — different decomposition |
| STL pattern | `mean(trend)` (a **scalar**) + `seasonal[:seg_size]` (`STL.py:46-49`) | full `trend + seasonal` vectors, averaged over class (`:56,83`) | **YES** |
| Alignment | iterative, DTW-scored, greedy accept over 5 iterations (`TSDTransform.py:156-173`, `alignment_score` uses `dtw`) | single-pass argmax cross-correlation, median offset across channels (`:98-102,484-492`) | Likely |
| Second STL pass | STL **re-run on aligned data** before residuals (`Run_SRS.py:101-105`) | patterns computed once, pre-alignment (`:340-352`) | Likely |
| Class assignment at test | official uses **true** OOD labels via `adjust_labels` (`Run_SRS.py:160-161`) | `_find_best_class`, min-residual (`:411,495-510`) | **No** — a *necessary* and sound adaptation (labels unavailable for OOD) |
| MC samples | `mc_range=50` (`Run_SRS.py:136`) | `mc_samples=10` (`:293`) | Minor — noisier estimate |

## 3. Orientation — and why the official code gives no guidance

The prompt asks whether using the signal neg-ELBO instead of the ratio "changes orientation or ranking".

**Ranking: yes, definitively.** `neg_elbo_sig` and `neg_elbo_sig / neg_elbo_res` are not monotonically
related — the denominator varies per sample, so no order-preserving map connects them. Dropping the
ratio changes the ranking and therefore the AUROC. This is not a metric-invariant simplification.

**Orientation: the official code cannot answer it**, for two reasons found in
`OOD_Auroc_utils.py`:

1. **The official AUROC is orientation-agnostic by construction** (`:129-132`):
   ```python
   if auc_value >= 0.5:
       return auc_value
   else:
       return 1 - auc_value
   ```
   It reports `max(auc, 1−auc)`, so the official harness *cannot* produce a below-chance number and is
   indifferent to score sign. Any monotone transform — including negation — scores identically there.
2. **The official decision rule is not a monotone threshold on the score.** `tpr_fpr` (`:91-96`) calls
   `is_OOD(ratio_ref, ratio_elements, method='var', var_factor=th)` — a **two-sided variance deviation**
   from the ID reference ratio distribution, swept over `th ∈ [0,15]`. Labels are `in = −1`,
   `ood = +1` (`:85-88`). So "OOD" means *far from the ID ratio in either direction*, not *high score*.

**Consequence for this thesis:** the project evaluates with standard scikit-learn `roc_auc_score`,
which is neither orientation-agnostic nor two-sided. The official numbers are therefore **not directly
comparable** to the project's, and the official code provides no sign convention to inherit. Whatever
orientation is chosen must be justified on its own terms — as the current code does implicitly (higher
neg-ELBO = worse fit = more OOD), which is defensible and empirically works (§5).

This also means a faithful port of the official *metric* would inflate every method's score by taking
`max(auc, 1−auc)`. The project is right not to do that, and the difference should be noted whenever
the paper's reported SRS numbers are cited.

## 4. A bug in the official `likelihood()` worth recording

`CVAE_Keras.py:238-247`:

```python
rl = np.zeros(len(x))
for mc_i in range(mc_range):
    ...
    reconstruction_loss = tf.reduce_sum(keras.losses.binary_crossentropy(x_normal, reconstruction_normal))
    rl = rl - reconstruction_loss
lls.extend(rl/mc_range)
```

`keras.losses.binary_crossentropy` reduces the last axis, giving shape `(batch, seg_size)`; the
unaxised `tf.reduce_sum` then collapses it to a **scalar**. Subtracting that scalar from the
per-sample vector `rl` assigns **every sample in the 128-batch the same likelihood**. The official
per-sample "likelihood" is therefore constant within each batch — consistent with the HEAD commit
message, *"Update batch likelihood"*.

`srs.py:207-215` computes a genuine **per-sample** neg-ELBO (`reduction="none"`, summed per sample).
This is a *deliberate improvement* over the official code and matches what the method plainly
requires. It should be documented as such rather than described as a faithful port — porting the
official code literally would yield a per-batch-constant, near-useless score.

## 5. Empirical status

From `experiments/*/*/srs/` (29 datasets — SRS is **absent on 11** of the 40 datasets the other
detectors cover; worth investigating separately):

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 29 | **0.7864** | — |
| TSB-U (univariate) | 20 | **0.8408** | 1/20 |

All scores are positive and finite (global range `[3.65, 1.38e4]`), consistent with a neg-ELBO
(non-negative MSE + non-negative KL) and with the orientation higher = OOD. No degenerate
constant-score datasets.

For context, SRS is by far the strongest detector verified so far — TSB-U mean 0.841 against MSP
0.342, EBO 0.277 and ODIN 0.261. Under the official orientation-agnostic convention the mean would be
0.832 (all datasets), i.e. essentially unchanged, confirming the chosen orientation is the correct one
and not an artefact.

**Not measurable here:** the effect of restoring the ratio. That requires retraining both CVAEs and
re-scoring; no cached features, checkpoints, or per-sample residual neg-ELBOs are saved. No estimate
of the ratio's impact on these numbers is offered.

## 6. Recommendations

1. **Fix the docstring** (`srs.py:9-12`) — remove "faithful PyTorch port", state that the score is the
   signal neg-ELBO, that the ratio is not used, and that the per-sample likelihood deliberately
   differs from (and improves on) the official batch-constant implementation.
2. **Either restore the ratio or stop training the residual CVAE.** As written the pipeline pays the
   full cost of `rescvae` (training at `:367-376`, MC evaluation at `:441`) for a value that is
   discarded. Restoring it is one line at `:452`; if the stability concern is real, record the evidence
   for it, because the claim is currently undocumented.
3. **Relabel in the results tables** as an SRS-inspired variant (STL alignment + conditional-VAE
   neg-ELBO), not as SRS — the ranking differs from the paper's score, so the 0.841 cannot be
   attributed to Seasonal Ratio Scoring.
4. **Note the metric mismatch** whenever the paper's SRS numbers are cited: the official AUROC is
   `max(auc, 1−auc)` over a two-sided variance test, not a standard ranking AUROC.
5. **Investigate the 11 missing datasets** — SRS covers 29 of 40.

**Open item shared with the other verifications:** the 18-vs-21 univariate dataset-count discrepancy
recorded in `methods/msp/VERIFICATION.md` §5, `methods/odin/VERIFICATION.md` §7,
`methods/energy_ebo/VERIFICATION.md` §6, `methods/mahalanobis_mds/VERIFICATION.md` §8 and
`methods/dfm/VERIFICATION.md` §7 remains unresolved. Note SRS's own coverage here is 20 univariate
datasets, different again from both 18 and 21.

## 7. Conclusion

SRS's pipeline is reproduced in structure but not in score. STL decomposition, circular alignment and
twin class-conditional VAEs are all present, and the per-sample likelihood is a genuine improvement on
the official code's batch-constant bug. But the seasonal ratio — the paper's titular contribution — is
never computed, the CVAE is a Gaussian/MSE model with decoder-only conditioning rather than the
official Bernoulli/BCE model conditioned in both halves, and the STL and alignment procedures differ
materially. The detector performs well (TSB-U 0.841, the best verified so far), but that performance
belongs to an SRS-inspired variant and must not be reported as Seasonal Ratio Scoring.

---

## FIX APPLIED (2026-08-20)

**Verdict after fix: FAITHFUL** (to the paper's defining Seasonal Ratio Scoring mechanism,
within this benchmark's protocol).

### What changed

Only `benchmark1/models/ood_methods/srs.py` was edited (plus this file). The runner,
`base_ood.py`, `__init__.py`, and all other methods are untouched. The class name
(`SRSDetector`), the `@register_ood("srs")` registration, and the `BaseOODDetector`
interface (`fit(x_id, y_id)` / `score(x)`) are unchanged.

1. **Restored the seasonal ratio (`srs.py:score()`, formerly line 452).** The score is now
   the per-sample seasonal RATIO of signal to residual neg-ELBO — `neg_elbo_sig /
   neg_elbo_res` — instead of the signal neg-ELBO alone. This mirrors the official
   `ratio = ll_signal / ll_residual` (`reference/Run_SRS.py:139,145,178`). Both operands
   were already computed and in scope (the MC-averaged `neg_elbo_sig` and `neg_elbo_res`),
   and the residual CVAE — previously trained and evaluated then discarded — now contributes
   to the score as the ratio denominator. Orientation is preserved (higher = more OOD): OOD
   samples fit no ID class pattern → high signal neg-ELBO → high ratio.

2. **Divide-by-zero guard.** The denominator is clamped to `eps = 1e-8` where
   `|neg_elbo_res| < eps` (`np.where`), keeping the score finite. neg-ELBO = MSE + KL is
   non-negative, so this only guards the degenerate near-zero-residual case.

3. **Docstring corrected (`srs.py:9-12`).** Removed the false "faithful PyTorch port of the
   original Keras/TensorFlow implementation" claim. The header now states the score is the
   signal/residual seasonal ratio (as in the paper), notes it is an SRS-inspired PyTorch
   implementation, and records that the per-sample neg-ELBO deliberately improves on the
   official batch-constant likelihood (§4). The stale inline comment that justified returning
   the signal neg-ELBO "directly" for stability was replaced with the ratio rationale.

### Diff summary

- `srs.py` module docstring (score description + "faithful port" claim corrected).
- `srs.py` `score()`: return statement changed from `return neg_elbo_sig` to a guarded
  `neg_elbo_sig / neg_elbo_res_safe`; explanatory comment rewritten.

### Smoke test

Ran with `C:\THESIS\.venv\Scripts\python.exe` on a dummy `SRSDetector(model=None, config=...)`
(latent_dim 4, 2 epochs, mc_samples 3), fit on random ID data `(40, 2, 32)` with 2
pseudo-classes, scored random test data `(15, 2, 32)`:

```
fit() OK
score dtype: float32
score shape: (15,) expected: (15,)
all finite: True
sample scores: [0.9995 0.9914 1.0039 0.9942 0.9987]
constant-input finite: True
SMOKE TEST PASSED
```

Output is finite, length-correct (one score per input window), and float-typed; the
constant-input case exercises the divide-by-zero guard and stays finite. The full benchmark
was NOT run.

### Residual caveats (not blocking the FAITHFUL verdict on the ratio)

The titular Seasonal Ratio is now formed, which was the one defining mechanism previously
absent. The deeper generative-model differences catalogued in §2 (Gaussian/MSE + linear
decoder vs official Bernoulli/BCE + sigmoid on min-max data; decoder-only conditioning;
per-sample vs per-class STL; xcorr vs iterative DTW alignment; neg-ELBO with a KL term)
remain as documented, deliberate adaptations to this benchmark's frozen-backbone,
per-sample, standard-`roc_auc_score` protocol. Reported numbers should be regenerated,
since the ratio is non-monotone relative to the previous signal-only score (§3).
