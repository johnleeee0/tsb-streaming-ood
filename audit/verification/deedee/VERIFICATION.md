# DEEDEE Faithfulness Verification — FAITHFUL (production uses the faithful deedee_fix variant 2026-08-21; see FIX APPLIED)

## FIX APPLIED (2026-08-21)
The base `deedee` (which treated adjacent feature dimensions as temporal neighbours) was NOT-THE-METHOD and
is retired. The runner now wires `DEEDEEFixDetector` (`methods/deedee/deedee_fix/deedee_fix.py`), which
computes DEEDEE's two statistics over the window's real time axis (episodewise mean + RBF self-similarity
over consecutive timesteps) and fits an isolation forest on ID trajectory statistics — faithful to the
paper's trajectory-statistics design. Verified end-to-end in the integration smoke test.


**Method id:** `deedee` (+ `deedee_fix`) · **Paper:** Aljaafari, Kanade, Torr & Schroeder de Witt,
*DEEDEE: Fast and Scalable Out-of-Distribution Dynamics Detection*, arXiv:2510.21638 (24 Oct 2025)
**Implementation:** `benchmark1/models/ood_methods/deedee.py` (`DEEDEEDetector`)
**Corrected variant:** `methods/deedee/deedee_fix/deedee_fix.py` (`DEEDEEFixDetector`)
**Verified:** 2026-08-20

---

## Verdict

**NOT-THE-METHOD for `deedee`** — the variant that actually produced results. Neither of DEEDEE's two
statistics is computed as defined:

1. **The "episodewise mean" is not a mean.** The docstring (`:17-18`) claims
   *"episodewise mean (μ = (1/w) Σ x_t)"*. The code computes `mean_stat = feat_vec[dim]`
   (`:110`, `:211`) — the raw feature value at one dimension. No averaging occurs at all.
2. **The RBF similarity is computed over adjacent feature dimensions, not over time**, and not against
   a training summary. `_compute_rbf_similarity` (`:131-168`) sums squared differences to
   `center_dim ± window_size//2` **neighbouring dimensions of the embedding** (`:148-159`), with the
   code itself commenting *"Use neighboring dimensions as 'temporal' neighbors"* (`:113`). Embedding
   dimensions are unordered, so this adjacency is arbitrary. The paper's statistic is an *"RBF kernel
   similarity **to a training summary**"* — there is no training summary here.
3. **It inverts the paper's headline contribution.** DEEDEE's selling point is a *"600-fold reduction
   in compute (FLOPs / wall-time)"*. This implementation makes **one `IsolationForest.score_samples`
   call per (sample, dimension)** — 150,528 separate sklearn calls for a 294-window, 512-dim dataset
   (§3B) — and fits `feat_dim` separate forests. A method whose entire claim is speed has been
   reimplemented as one of the slowest detectors in the suite.

**`deedee_fix` is substantially faithful — and has zero results.** It computes both statistics over the
**real time axis** of the raw windows: `x.mean(axis=2)` for the episodewise mean (`deedee_fix.py:35`)
and RBF over consecutive-timestep differences (`:37-38`), then fits **one** vectorised isolation forest
(`:47-49`). Its `CHANGES.md` diagnoses the defect precisely and concludes *"Use `deedee_fix`."*
**That instruction was never carried out: `deedee_fix` has 0 result directories** (§3A).

**Exclusion from the main sweep is confirmed:** `deedee` has **4** loadable result directories, all in
the `ablation` suite, versus 40 for the sweep methods.

**Also: the tracker's paper title is wrong; the code's is right.** Actual title as above; the tracker
gives *"DEEDEE: Detecting OOD Environment Dynamics from Trajectory Statistics"*. Eighth title error in
this audit, and the second where the code is correct and the tracker is not.

---

## 1. Source accessibility

| Source | Status |
|---|---|
| `methods/deedee/reference/` | **ABSENT — directory does not exist.** The folder contains `validation_status.json` and the `deedee_fix/` subdirectory. Matches the tracker's expectation. |
| Official code | **NONE FOUND.** No repository surfaced. Supports the tracker's "NONE PUBLIC" claim, but this is absence of evidence, not confirmation. |
| `arxiv.org/abs/2510.21638` | **Live fetch FAILED** (`socket hang up`). Title, authors, abstract and method summary were obtained from **search results** — a secondary source. |

**The paper's equations were not read.** What is established below about the paper comes from an
abstract-level secondary summary. No equation or symbol is asserted or guessed. This is the same
evidence limitation recorded for TD-IVDM, CatSight and AE-ADWIN-LSTM.

### What secondary sources establish about the paper

- Title / authors / date: *DEEDEE: Fast and Scalable Out-of-Distribution Dynamics Detection*; Tala
  Aljaafari, Varun Kanade, Philip Torr, Christian Schroeder de Witt; 24 October 2025.
  **`deedee.py:6, 8` is correct** (though it lists only the first author).
- Method: *"a two-statistic detector for OOD detection in **RL time series** … uses only an
  **episodewise mean** and an **RBF kernel similarity to a training summary**, capturing complementary
  global and local deviations."*
- Contribution: *"a **600-fold reduction in compute** (FLOPs / wall-time) and an average 5% absolute
  accuracy gain over strong baselines."*
- Framing: *"diverse anomaly types often imprint on RL trajectories through a small set of low-order
  statistics."*

## 2. Divergence table

| Component | Paper (secondary) | `deedee.py:line` | `deedee_fix.py:line` |
|---|---|---|---|
| **Operand** | RL trajectories / observations over time | frozen backbone features `(N, d)` (`:85`) | **raw windows `(N,C,T)`** (`:44`, `:54`) ✓ |
| **Episodewise mean** | `μ = (1/w) Σ_t x_t` over time | `feat_vec[dim]` — **no mean** (`:110`, `:211`) ✗ | `x.mean(axis=2)` over time (`:35`) ✓ |
| **RBF similarity** | to a **training summary** | to **adjacent feature dimensions** (`:148-159`) ✗ | consecutive-timestep self-similarity (`:37-38`) — right axis, but still not a training summary |
| **Detector** | consumes the two statistics | `feat_dim` separate isolation forests (`:122-129`) | **one** isolation forest (`:47-49`) ✓ |
| **Cost** | 600× *faster* than baselines | `N·d` `score_samples` calls (`:209-222`); `O(N·d·w)` Python `fit` (`:98-118`) ✗✗ | vectorised, one `fit` / one `score_samples` ✓ |
| Orientation | anomaly score | `-score_samples`, higher = OOD (`:222`) ✓ | `-score_samples` (`:57`) ✓ |
| Aggregation | n/a | mean over per-dimension scores (`:226`) | n/a — single forest |
| **Results produced** | — | **4 datasets** (ablation only) | **0 datasets** |

## 3. Empirical findings

**(A) Coverage.**

| Variant | Result dirs | Suite |
|---|---|---|
| `deedee` | **4** | all `ablation` (`TSB-M-DRIFT003`, `TSB-U-DRIFT024`, `TSB-U-OOD009`, `TSB-U-STABLE001`) |
| `deedee_fix` | **0** | — |

So the exclusion from the main sweep is confirmed, and the corrected variant that `CHANGES.md` tells the
reader to use has never been executed on real data. Its only evidence is the synthetic task
(`deedee_fix` 0.989 vs original 0.982 — a 0.007 gap on a task that, per this audit's pattern, has not
discriminated reliably for any `_enh` variant).

**(B) The cost inversion, quantified.** `deedee.score()` calls
`self.isolation_forests[dim].score_samples(...)` on a `(1, 2)` array inside a per-dimension loop
(`:209-222`):

| N windows | feat_dim | IsolationForest calls | RBF inner iterations |
|---|---|---|---|
| 110 | 64 | 7,040 | 63,360 |
| 110 | 512 | 56,320 | 506,880 |
| 294 | 512 | **150,528** | **1,354,752** |

`fit()` has the same shape — `O(N·d·w)` pure-Python statistic construction (`:98-118`) followed by
`feat_dim` separate `IsolationForest.fit` calls (`:122-129`). For a method whose stated contribution is
a 600× compute reduction, this is the defining property reversed.

**(C) Results on the 4 ablation datasets.**

| Dataset | AUROC |
|---|---|
| TSB-M-DRIFT003 | 0.558 |
| TSB-U-DRIFT024 | 0.672 |
| TSB-U-OOD009 | 0.500 |
| TSB-U-STABLE001 | 0.837 |
| **mean** | **0.6417** |

Scores span roughly `[0.36, 0.80]` with no degenerate datasets, and the orientation is correct
(flipping would give 0.3583). So the detector is not broken in the CODiT/InvAD sense — it produces a
usable signal. It simply is not DEEDEE.

## 4. Is the labelling honest?

**Mostly yes, and unusually candid in places.** `:2` says "(Adapted)"; `:10-12` states *"Original
DEEDEE: Works on RL trajectories (observations over time)"* versus *"DEEDEE-Lite: Applies trajectory
statistics on frozen backbone features"*; and the inline comments openly admit the improvisation —
*"For simplicity, treat each sample as a single-step 'trajectory' and use sliding windows within the
feature vector itself"* (`:92-93`) and *"Since we have a single feature vector per sample, we'll use the
feature value itself as a 'trajectory'"* (`:106-107`). A reader of the file is not misled about what is
happening.

**Three things undercut it:**

| Issue | Detail |
|---|---|
| Claimed statistics not computed | `:17-18` states the formula `μ = (1/w) Σ x_t`; `:110`/`:211` compute no mean. `:18-19` claims *"RBF similarity (s·exp(-d_t/σ²)) for local deviation detection"* over a trajectory; `:148-159` uses embedding-dimension adjacency. |
| Zero-discrepancy validation record | `validation_status.json`: `CRITICAL: 0, MODERATE: 0, MINOR: 0`, `status: "PASS"`, `notes: ""` — for the weakest-fidelity method in the suite. Same defect recorded for DIVERSIFY (`methods/diversify/VERIFICATION.md` §3). |
| Unactioned recommendation | `deedee_fix/CHANGES.md` ends *"Use `deedee_fix`."* It has 0 results; the excluded, unfaithful variant is the one with data. |

## 5. Recommendations

1. **Run `deedee_fix` on the full sweep, or drop DEEDEE from the thesis entirely.** As it stands the
   only DEEDEE numbers come from a variant whose two statistics are both wrong, on 4 of 40 datasets.
   `deedee_fix` is already written, vectorised and correct on the time axis — running it is cheap.
2. **Finish `deedee_fix`'s RBF term.** The paper specifies similarity *to a training summary*;
   `deedee_fix.py:37-38` computes a consecutive-timestep **self**-similarity. Storing an ID summary at
   `fit()` and measuring RBF similarity to it would close the last gap. Note this honestly if not done.
3. **Fix the `deedee.py` docstring** — remove the `μ = (1/w) Σ x_t` claim at `:17-18`, which the code
   does not implement.
4. **Fix `validation_status.json`** — a zero-discrepancy PASS is indefensible for this method.
5. **State the compute inversion** in the thesis if `deedee` is reported at all: DEEDEE's contribution
   is a 600× speedup, and this implementation is `O(N·d)` in sklearn calls. That is the single most
   important fidelity fact about it.
6. **Fix the tracker title** (`deedee.py:6` is correct) and add the full author list.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded
across `methods/*/VERIFICATION.md`. `deedee` covers only 3 univariate datasets (of 4 total), so it does
not bear on that question.

## 6. Conclusion

`deedee` is correctly described in its own source as an adaptation, and its comments are unusually frank
about the liberties taken — but the liberties are total. The episodewise mean is not averaged over
anything, the RBF similarity is computed across unordered embedding dimensions rather than over time and
not against any training summary, and the implementation makes one isolation-forest call per
sample-dimension pair, inverting the paper's central claim of a 600-fold compute reduction. That places
it below the ADAPTATION bar. The corrected `deedee_fix` variant fixes the axis, the mean, and the cost
in 58 lines and is substantially faithful — but it has never been run, and the unfaithful variant is
the one with results, on 4 ablation datasets. Verification confidence is low: no public code, and the
paper was reachable only through a secondary abstract summary, so no equation-level comparison was
possible.
