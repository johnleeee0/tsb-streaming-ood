# AE-ADWIN-LSTM Faithfulness Verification — NOT-THE-METHOD (temporal components non-functional on shuffled windows; orientation inverted)

**Method id:** `ae_adwin_lstm` · **Paper:** *A Novel Concept Drift Detection Model for Handling
Evolving Patterns in Multivariate Time Series*, **IEEE APCI 2025**,
doi:10.1109/APCI65531.2025.11136854 (IEEE Xplore doc 11136854)
**Implementation:** `benchmark1/models/ood_methods/ae_adwin_lstm.py` (`AEADWINLSTMDetector`)
**Verified:** 2026-08-20

---

## Verdict

**NOT-THE-METHOD.** The docstring does disclose an adaptation (`:4` "Adapted from:", `:8` "This
lightweight adaptation works on frozen backbone features rather than raw data"), but four findings put
this below the ADAPTATION bar:

1. **The evaluation windows are randomly permuted — confirmed in the loader.**
   `tsb_loader.py:313` calls `_balance_binary`, which at `:148` does `idx = rng.permutation(idx)`. So
   the LSTM's 9-window history (`:335`, `:351`) and ADWIN's error stream (`:368`) are built over a
   **random ordering**. Two different shuffles of the *same* data produce rankings correlated at only
   **Spearman 0.796** (§3A) — roughly a fifth of the ranking is an artefact of arrival order.
2. **"ADWIN" is not ADWIN.** A single fixed cut at `n//2` (`:78`) instead of testing all cut points; a
   std-scaled threshold `eps·(std_recent + std_older)` (`:93-94`) instead of the ADWIN Hoeffding bound;
   and `max_buckets` stored but never used (`:56`) — there is no exponential-histogram structure. It is
   a two-sample mean test at the midpoint.
3. **The paper's defining mechanism is absent.** Per the paper, ADWIN monitors LSTM residuals and, *"when
   drift is identified, the model is incrementally updated"*. No incremental update exists anywhere in
   this implementation; the drift flag becomes a 0.2-weighted binary spike in a per-window score
   (`:371`, `:375`).
4. **The orientation is inverted.** Measured across 40 datasets: **0.2529** as implemented versus
   **0.7471** flipped (§3D).

**The docstring also contradicts itself.** `:12-13` states *"OOD samples produce **high** reconstruction
error … **high** prediction error"*, while `:378-383` justifies the negation on the opposite premise —
*"OOD features … are **easier** to reconstruct/predict than ID features … negate."* The code implements
the second. Both cannot be true, and `:12-13` matches what the empirical result says should have been
used.

**Also: the docstring's title/DOI are correct; the tracker's title is not** (§1) — sixth such
tracker-metadata error.

---

## 1. Source accessibility — the paper could NOT be read

| Source | Status |
|---|---|
| `methods/ae_adwin_lstm/reference/` | **ABSENT — directory does not exist.** Only `validation_status.json`. Matches the tracker's expectation. |
| Official code | **NONE FOUND.** No repository surfaced for this paper. Supports the tracker's "NONE PUBLIC" claim, but this is absence of evidence, not confirmation. |
| `ieeexplore.ieee.org/document/11136854` | **Not retrieved** — IEEE Xplore is paywalled; no full text obtained. |

**Therefore the paper's equations were not read, and no comparison against them is claimed.** What is
available is a **secondary, search-result-derived summary**, quoted below. Nothing about the paper is
guessed.

**Tracker vs. docstring title.** The tracker gives *"Combining Autoencoder Reconstruction, ADWIN Change
Detection and LSTM Prediction for Evolving Multivariate Time Series"*. The published title is
*"A Novel Concept Drift Detection Model for Handling Evolving Patterns in Multivariate Time Series"* —
which is what `ae_adwin_lstm.py:4-6` says, with a valid IEEE DOI. **The code is right; the tracker is
wrong.**

### What secondary sources establish about the paper

- A *"Hybrid Autoencoder-LSTM-ADWIN framework that integrates unsupervised learning, deep learning, and
  adaptive drift detection to improve **forecasting accuracy** in non-stationary environments."*
- *"The ADWIN … algorithm continuously monitors **residual errors between LSTM predictions and actual
  values** to detect concept drift dynamically."*
- *"When drift is identified, **the model is incrementally updated**, allowing adaptation to evolving
  patterns while reducing unnecessary retraining overhead."*
- Targets *"both sudden and gradual drifts"*; evaluated on detection accuracy and forecasting.

So the paper is a **forecasting + drift-detection** system on an **ordered stream**, whose contribution
is the drift-triggered incremental update loop.

## 2. Divergence table

| Component | Paper (per secondary sources) | Mine (`ae_adwin_lstm.py:line`) | Changes results? |
|---|---|---|---|
| **Stream ordering** | ordered time series | **randomly permuted** eval set (`tsb_loader.py:313` → `:148`) | **YES — invalidates both temporal components** |
| **Drift-triggered incremental update** | *"the model is incrementally updated"* — the paper's contribution | **absent** — no update anywhere | **YES — defining mechanism missing** |
| **ADWIN** | ADWIN (exponential histogram, all cut points, Hoeffding bound) | midpoint two-sample mean test (`:78`, `:93-94`); `max_buckets` unused (`:56`) | **YES — not ADWIN** |
| **ADWIN input** | residuals between LSTM predictions and actuals | `recon_error_norm + pred_error_norm` (`:367`) — includes the AE term | **YES** |
| **Output** | drift decision / forecast | per-window scalar OOD score (`:375`) | **YES — different task** |
| **Orientation** | n/a (drift decision) | **negated** (`:384`), asserted at `:378-383` | **YES — 0.25 vs 0.75** |
| Input domain | raw multivariate series | frozen backbone features (`:216`, `:329`) | **YES** — disclosed adaptation |
| AE | autoencoder on raw data | MLP AE on features (`:113-141`) | Disclosed |
| LSTM | next-step predictor on the stream | next-step predictor on a 9-window random history (`:351-360`) | **YES** — input is meaningless |
| Score composition | n/a | `0.4·recon + 0.4·pred + 0.2·drift` (`:375`) | Invented weighting |
| First `seq_len−1` windows | n/a | `pred_error_norm = 0.0` (`:349`, `:353`) — different composition | **YES** — see §3B |
| ADWIN state reset | per stream | **never reset** in `score()` (`:368`) | **YES** — non-idempotent across calls |

## 3. Empirical findings

**(A) Order dependence is severe.** Replicating the `score()` loop verbatim (`:334-384`):

| Comparison | Spearman |
|---|---|
| ordered vs shuffled | **0.7762** |
| **shuffle A vs shuffle B** | **0.7961** |

Mean absolute score deviation between orderings: 0.1839 (**6.5%** relative). The second row is the
important one: two arbitrary permutations of the *same* windows disagree on ~20% of the ranking. Since
the benchmark's eval set **is** randomly permuted, this is the operating regime, not a hypothetical.

For comparison, M2N2's sequential EMA gave Spearman 0.971 under reshuffling — an order of magnitude
less order-sensitive.

**(B) A positional artefact affects ~22% of windows.** `:349` initialises `pred_error_norm = 0.0` and
`:353` only computes the LSTM term once `len(feat_history) == seq_len`. So stream positions 0–8 are
scored as `0.4·recon + 0.2·drift` while every later window is scored as
`0.4·recon + 0.4·pred + 0.2·drift`. On a 40-window eval set that is **9/40 ≈ 22%** of windows given a
structurally different score — and because the order is random, *which* windows those are is arbitrary.

**(C) The ADWIN implementation.**

| | Real ADWIN | `ae_adwin_lstm.py` |
|---|---|---|
| Window structure | exponential-histogram buckets | plain `deque(maxlen=1000)` (`:57`); `max_buckets` unused (`:56`) |
| Cut points tested | **all** | **one**, at `n//2` (`:78`) |
| Threshold | `ε_cut = sqrt((1/(2m))·ln(4/δ′))`, `δ′ = δ/n` | `ε·(std_recent + std_older)` (`:93-94`) — std-scaled, no `n` correction |

The `delta` parameter is used (`:93`), so the test is at least confidence-parameterised, but the
statistic is not ADWIN's.

**(D) Saved scores.**

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 40 | **0.2529** | — |
| TSB-U (univariate) | 21 | **0.2256** | **17/21** |
| All, **flipped** | 40 | **0.7471** | — |

Global range `[-1.933e+11, 2.988]`. The `-1.9e11` extreme comes from the pathological-feature-magnitude
family recorded in `methods/energy_ebo/VERIFICATION.md` §3 — and as with CatSight, the negation makes
those the *most* in-distribution windows.

This is the **second** method (after CatSight, also 0.25 → 0.75) whose asserted negation is worth
+0.49 mean AUROC if removed.

## 4. Recommendations

1. **Flip the orientation** (`:384`) and delete the rationale at `:378-383`. Note that `:12-13` already
   states the correct expectation — the docstring argues against itself, and the empirical result
   supports `:12-13`.
2. **Remove or disable the temporal components for this benchmark.** With a randomly permuted eval set,
   the LSTM prediction term and the ADWIN drift term are noise sources, not signals — demonstrably so
   (ρ = 0.80 between two shuffles). Either evaluate on time-ordered windows, or report the detector
   honestly as an autoencoder reconstruction score, which is the only component that is
   order-invariant.
3. **Fix the positional artefact** (`:349`, `:353`) — either drop the first `seq_len−1` windows or
   impute the prediction term, so every window has the same score composition.
4. **Rename the ADWIN class** or implement real ADWIN. As written, "ADWIN" overstates what `:43-103`
   does; `max_buckets` should be removed or used.
5. **Reset ADWIN at the start of `score()`** — it currently carries state across calls, so scoring is
   not idempotent (same issue recorded for M2N2 at `methods/m2n2/VERIFICATION.md` §3B).
6. **Relabel** from ADAPTATION to NOT-THE-METHOD, and report the current numbers as "AE reconstruction
   + two non-functional temporal terms, sign-inverted" — not as a reproduction.
7. **Fix the tracker title**; the code's is correct.
8. **State that the paper is paywalled** and no public code exists, so no equation-level comparison was
   possible — the same caveat as TD-IVDM and CatSight.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded
across `methods/*/VERIFICATION.md`. `ae_adwin_lstm` covers **21** univariate datasets; see
`methods/diversify/VERIFICATION.md` §4 and `methods/catsight/VERIFICATION.md` §4 for the candidate
explanation of the "18" figure.

## 5. Conclusion

Of the three components in the method's name, only the autoencoder functions as intended. The LSTM
predictor consumes a 9-window history assembled from a randomly permuted evaluation set, and the
"ADWIN" detector — itself a midpoint two-sample mean test rather than ADWIN — monitors an error stream
in that same arbitrary order; two different permutations of identical data agree on only about 80% of
the ranking. The paper's actual contribution, incrementally updating the model when ADWIN fires, is not
implemented at all, and the paper's task is forecasting on an ordered stream rather than per-window OOD
scoring. On top of this the score is negated on an asserted premise that the docstring elsewhere
contradicts and that the data refutes: 0.2529 as implemented against 0.7471 flipped. The adaptation is
disclosed, but what is disclosed is not a version of the paper's method — hence NOT-THE-METHOD rather
than ADAPTATION. Verification confidence is low: no public code and a paywalled paper, so the
comparison rests on secondary descriptions.

## CLASS-D BUILD (2026-08-21)

A faithful ordered-stream build was created as a **separate Class-D appendix study** (never a row in the
17-method leaderboard). It fixes every defect flagged above.

- **File:** `methods/ae_adwin_lstm/classd/ae_adwin_lstm_classd.py` (`AEADWINLSTMClassD`, plus a real
  `ADWIN` class and a `drift_delay` helper); orchestrated by `experiments/run_class_d.py` (registry entry
  `ae_adwin_lstm`, `eval_mode="ordered_per_window"`).
- **What was fixed vs the production stand-in:**
  1. **Ordered stream** — scored on `load_tsb(ordered_eval=True)["stream"]` in temporal order, so the
     LSTM history and ADWIN error stream are meaningful (no random-permutation artefact).
  2. **Real ADWIN** — exponential-histogram buckets (powers of two, ≤ `max_buckets` per size), **all**
     bucket-boundary cut points tested, Hoeffding cut `eps=sqrt(0.5·(1/n0+1/n1)·ln(4/δ'))` with
     `δ'=δ/n`, dropping the older sub-window on a detected change. Not the old midpoint two-sample test.
  3. **Drift-triggered incremental update** — the paper's defining mechanism: when ADWIN fires, the AE
     (and LSTM) take a light gradient step on the recent-window buffer.
  4. **ADWIN reset** at the start of every `score_stream` (idempotent scoring).
  5. **Positional artefact fixed** — the first `seq_len-1` windows use the normalised prediction-error
     mean (0.0) so score composition is uniform.
  6. **Orientation corrected** — higher error = more OOD, **no negation** (the production 0.25→0.75
     inversion is gone).
- **Eval mode:** ordered per-window → per-window scalar in stream order → `per_sample_auroc` directly.
  **Secondary metric:** drift-detection-delay (windows between true drift onset and first ADWIN alarm),
  written to the `drift_delay` column.
- **Honest caveats:** works on **frozen-backbone features** (a disclosed adaptation, matching the
  production domain), not raw series; ADWIN's bounded-range Hoeffding cut is used on the z-scored residual
  stream; on short ordered streams the incremental update fires rarely and per-window AUROC is noisy.
- **Smoke (TSB_N_PER_CELL=2, U only):** finite per-sample AUROCs in `results/class_d_group2.csv` (e.g.
  STABLE_002 = 0.991, OOD_001 = 0.849, delays 0/4/15). AUROC varies file-to-file (some below chance),
  reported honestly. No production file was modified.
