# CatSight Faithfulness Verification — ADAPTATION (orientation flipped 2026-08-21; see FIX APPLIED)

**Method id:** `catsight` · **Paper:** Flórez, Rodríguez-Moreno, Artetxe et al., *CatSight, a direct
path to proper multi-variate time series change detection: perceiving a concept drift through common
spatial pattern*, **Int. J. Machine Learning and Cybernetics**, 2023, doi:10.1007/s13042-023-01810-z
**Implementation:** `benchmark1/models/ood_methods/catsight.py` (`CatSightDetector`)
**Verified:** 2026-08-20

---

## Verdict

**ADAPTATION — the label is honest** (`:2` "(Adapted)", `:10` "Adaptation for OOD detection on frozen
backbone", `:11` "Original CatSight uses CSP on raw multivariate time series", `:12` "CatSight-Lite").
The docstring describes what the code does, and `:2`'s title/DOI are **correct** — unlike the tracker's
(§1).

**But the score orientation is not merely "asserted" — it is empirically refuted, and the fix is one
character.** `:199` returns `-np.linalg.norm(...)`, justified at `:193-197` by the claim that *"OOD
samples that lack any class-specific pattern land NEAR the centroid"*. Measured across 37 datasets:

| Orientation | mean AUROC (all) | TSB-U | below chance |
|---|---|---|---|
| **As implemented** (negated) | **0.2499** | 0.2229 | 15/18 |
| **Flipped** (plain distance) | **0.7501** | — | — |

The decisive evidence is the extreme-feature-magnitude subset — `TSB-M-STABLE_015`, `STABLE_020`,
`STABLE_043`, `STABLE_083`, `TSB-U-DRIFT_060`, where the backbone emits features of order 10⁵–10⁶.
**All five score exactly AUROC 0.000** (subset mean 0.0000). Those are precisely the datasets where OOD
windows are *farthest* from the ID centroid, so the negation ranks them as the most in-distribution.
The stated rationale predicts the opposite and is wrong.

**Two things I can confirm positively**, by mathematical verification rather than paper comparison:

- **The CSP eigenproblem is a legitimate, standard formulation** and is equivalent to the form the
  docstring states (§3A).
- **The confusing double-indexing in component selection is functionally correct** (§3B), though
  fragile.

---

## 1. Source accessibility — the paper could NOT be read

| Source | Status |
|---|---|
| `methods/catsight/reference/` | **ABSENT — directory does not exist.** Only `validation_status.json` is present. Consistent with the tracker's expectation. |
| Official code | **NONE FOUND.** Searches surfaced no repository for CatSight. This supports the tracker's "NONE PUBLIC" claim, but it is **absence of evidence**, not confirmation. |
| `link.springer.com/article/10.1007/s13042-023-01810-z` | **Paywalled** — 303 redirect to `idp.springer.com/authorize`. Auth endpoint not followed. |
| `openreview.net/forum?id=TD2H4bXOUK` | **Blocked** — served a browser-verification page, no content. |

**Therefore the paper's CSP formulation, component-selection rule, and decision statistic were not
read, and this report does not verify against them.** What is available is a **secondary,
search-result-derived summary**, quoted below. No equation or symbol from the paper is asserted, and
none is guessed. §3A/§3B below are *internal mathematical checks* of the implementation — they
establish that the code implements a standard CSP correctly, **not** that it matches the paper's
specific choices.

### What secondary sources establish about the paper

- Title, venue, year: as above — **the docstring at `:5-8` is correct**. The tracker's title
  (*"CatSight: Common Spatial Pattern Analysis for Distributional Change Detection in Multivariate
  Sensor Data"*) does not match the published title. Fifth such tracker-metadata error, after DiMMAD,
  InvAD, M2N2 and TD-IVDM.
- Authors: Flórez, Rodríguez-Moreno, Artetxe et al. (`catsight.py` lists none).
- Structure — **two steps**: *"(i) Use of Common Spatial Patterns … to maximize the difference between
  two different distributions of a multivariate temporal data, and (ii) **Machine Learning conventional
  algorithms to detect whether a change in the data flow has occurred or not**."*
- CSP is described as *"a statistical approach to deal with data streaming, closely related to
  Principal Component Analysis"*.
- Target: *"sudden or abrupt drift, the most common drift found in industrial processes"*, for
  Industry 4.0 machine monitoring.

## 2. Divergence table

| Component | Paper (per secondary sources) | Mine (`catsight.py:line`) | Changes results? |
|---|---|---|---|
| **Step (ii) decision rule** | **trained conventional ML classifier** on CSP-projected features | **normalised distance to the ID centroid** (`:198-199`) — no classifier | **YES — step (ii) replaced** |
| **Score orientation** | classifier output (change / no change) | **negated** distance (`:199`) | **YES — empirically wrong; see §4** |
| Input domain | raw multivariate time series | frozen backbone features (`:126`) | **YES** — disclosed adaptation |
| CSP eigenproblem | generalised eigenproblem (form not readable) | `linalg.eigh(C1, C1 + C2)` (`:68`) | Standard CSP; equivalent to the docstring's form (§3A) |
| Covariance | not readable | `np.cov` + `reg·I`, `reg=1e-4` (`:63-64`) | Reasonable |
| Component selection | "extreme eigenvalues" (count not readable) | 3 top + 3 bottom of 6 (`:76-84`) | Functionally correct (§3B) |
| Two-distribution setup | two windows of a stream | **first two of four pseudo-classes only** (`:136-141`) | **YES** — half the ID classes ignored |
| Unsupervised fallback | n/a | temporal half-split (`:144-147`) | Arguably *closer* to the paper's two-window setting than the labelled path |
| Reference statistics | n/a | ID mean/std in CSP space over **all** classes (`:157-159`) | Added step |

## 3. Positive verification of the CSP implementation

**(A) `eigh(C1, C1+C2)` is equivalent to the documented `C1·W = λ·C2·W`.** The docstring (`:17`) states
one form; the code (`:68`) uses the other. They give the **same eigenvectors in the same order**:

| Check | Result |
|---|---|
| `max \|sort(λ_b) − sort(λ_a/(1+λ_a))\|` | **8.68e-15** |
| `\|cos\|` between matched eigenvectors (min / mean) | **1.000000 / 1.000000** |
| Eigenvalue ordering identical | **True** |

Since `C1 W = λ_b(C1+C2)W ⟺ C1 W = [λ_b/(1−λ_b)] C2 W`, the two differ only by the monotone
reparameterisation `λ_b = λ_a/(1+λ_a)`. `eigh(C1, C1+C2)` is the standard numerically-stable CSP form
(as used in MNE-Python). **No divergence — and the docstring's formula is substantively right.**

**(B) The double-indexing is accidentally correct.** `:71-84` sorts the eigenvectors descending, then
indexes the *sorted* array with the *original* index array `ix`:

| | Selected positions | Selected eigenvalues |
|---|---|---|
| As implemented | `[11, 10, 9, 2, 1, 0]` | `[0.0001, 0.0076, 0.1387, 0.8705, 0.9858, 0.9971]` |
| Intended | `[0, 1, 2, 9, 10, 11]` | `[0.9971, 0.9858, 0.8705, 0.1387, 0.0076, 0.0001]` |
| **Same set?** | **Yes** | — |

Because `scipy.linalg.eigh` returns eigenvalues in **ascending** order, `ix` is exactly the reversal
permutation, so `ix[:k]` addresses the tail of the descending array and `ix[-k:]` its head. The
selected *set* is the intended 3-top + 3-bottom; only the column order differs, and the returned
eigenvalues are discarded at the call site (`:150`, `_`). **Functionally correct but fragile** — it
depends on an ordering guarantee that is incidental to the algorithm, and the local variable names
(`top_ix` holds bottom indices) invite future breakage.

## 4. The orientation is empirically refuted

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 37 | **0.2499** | — |
| TSB-U (univariate) | 18 | **0.2229** | **15/18** |
| **Extreme-magnitude subset** | 5 | **0.0000** | 5/5 |
| All, **flipped** orientation | 37 | **0.7501** | — |

Scores span `[-6.508e+05, -0.4294]`, all ≤ 0, confirming the negated-norm formula.

The extreme-magnitude datasets settle the question. On `TSB-M-STABLE_015` the score range is
`[-6.5e5, -9.65e4]` — OOD windows sit six orders of magnitude from the ID centroid, and the negation
ranks them as maximally in-distribution, giving AUROC exactly 0.000. The docstring's premise
(`:193-197`) that OOD samples "land NEAR the centroid" is contradicted on every dataset where the
distance signal is large.

Flipping the sign raises the mean from 0.250 to 0.750, which would make CatSight the **second-strongest
detector** in the benchmark rather than the weakest.

**Missing outputs:** 3 of 40 datasets produced no scores — `TSB-U-STABLE_062`, `STABLE_070`,
`STABLE_080`. These are **the same three** that failed for DIVERSIFY, giving CatSight a univariate count
of **18**. This corroborates the explanation in `methods/diversify/VERIFICATION.md` §4: the "18
univariate datasets" figure in the validation notes most likely reflects runs that silently dropped
these three small YAHOO series, not a defined 18-dataset subset.

## 5. Recommendations

1. **Flip the orientation** — remove the negation at `:199` and delete the rationale at `:193-197`. This
   is a one-character change worth +0.50 mean AUROC, and the current justification is empirically false.
   Re-run and re-report.
2. **Do not present the current 0.250 as CatSight's performance.** It is the performance of a
   sign-inverted detector.
3. **Disclose that step (ii) is replaced.** The paper's second stage is a trained classifier on CSP
   features; this implementation substitutes a centroid distance. That is the substantive adaptation and
   should be named as such in the thesis, not just "CSP on frozen features".
4. **Use all ID classes for CSP** or document why only the first two are used (`:136-141`). With four
   pseudo-classes, half the ID structure is currently discarded. Multiclass CSP (one-vs-rest filter
   banks) or the temporal half-split fallback (`:144-147`, arguably closer to the paper's two-window
   setting) are both better matches.
5. **Harden the component selection** at `:71-84` — index by position (`[:n_top]`, `[-n_bottom:]`) rather
   than by `ix`, and fix the swapped variable names. Currently correct only by accident.
6. **Add the author list** to the docstring, and **fix the tracker title**.
7. **State in the thesis that the paper was inaccessible** (Springer paywall, OpenReview gated) and that
   no public code exists — so the CSP formulation was verified for internal mathematical correctness,
   not against the paper's equations. Along with TD-IVDM, this is the weakest verification evidence in
   the set.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded
across `methods/*/VERIFICATION.md`. `catsight` covers **18** — the second method to do so, both times
because of the same three missing YAHOO datasets.

## 6. Conclusion

The implementation computes genuine CSP spatial filters: the generalised eigenproblem at `:68` is a
standard, numerically-stable formulation and is provably equivalent to the form documented at `:17`
(identical eigenvectors, identical ordering), and the component selection — despite confusing
double-indexing and swapped variable names — picks the intended three largest and three smallest
components. To that extent the CSP machinery is sound. Beyond it, two things diverge: the paper's
second stage, a trained classifier on CSP features, is replaced by a normalised distance to the ID
centroid; and the score is negated on the asserted premise that OOD samples lie near that centroid.
That premise is false on this data — the negation yields mean AUROC 0.250 against 0.750 flipped, and
returns exactly 0.000 on all five extreme-feature-magnitude datasets where OOD windows are farthest
from the centroid. The ADAPTATION label is honest, but the orientation is a defect requiring
correction, not a documented design choice. Verification confidence is low: no public code was found
and the paper is paywalled, so no comparison against its equations was possible and none is claimed.

## FIX APPLIED (2026-08-20)

**Change.** Removed the score-orientation negation in `benchmark1/models/ood_methods/catsight.py`
`score()` (formerly `:199`). The method now returns the plain normalised CSP-space distance to the ID
centroid:

```python
# before
ood_scores = -np.linalg.norm(normalized_diff, axis=1)
# after
ood_scores = np.linalg.norm(normalized_diff, axis=1)
```

The false rationale (formerly `:193-197`, claiming *"OOD samples … land NEAR the centroid"*) was
deleted and replaced with a comment stating the correct orientation: higher distance = more OOD,
consistent with the CSP distance direction (ID projects near the learned centroid; OOD deviates from
the learned spatial patterns). Scores are now non-negative rather than all ≤ 0.

**Scope.** This is the single load-bearing fix from `FIX_PLAN.md` §11 (Class A, effort S). The
*optional* pseudo-class improvement (multiclass CSP over all four pseudo-classes) was **not** applied:
the plan rates it "small M" rather than the low-risk S of the flip, and the instruction was to apply
the optional change only if it is low-risk. The binary-CSP path (`:136-141`) and the temporal
half-split fallback (`:144-147`) are unchanged. The class name (`CatSightDetector`), the
`@register_ood("catsight")` key, and the `BaseOODDetector` interface (`fit`/`score` signatures) are all
unchanged.

**Smoke test (venv `C:\THESIS\.venv`).** Instantiated `CatSightDetector` with a tiny dummy
frozen backbone (`(N,C,T)=(40,3,16) → feat_dim=12`), fit on random ID data with 4 pseudo-classes,
scored random test data:

- output is a finite `float64` `np.ndarray` of the correct length (`(17,)`), all values ≥ 0;
- orientation sanity check: mean score on large-magnitude OOD-like inputs (~131.6) is far higher than
  on ID inputs (~2.64) — confirming higher = more OOD. `SMOKE TEST PASSED`.

The full benchmark was not run.

**New verdict.** ADAPTATION — **orientation corrected**. The documented empirical result (as-implemented
mean AUROC 0.2499 vs flipped 0.7501) means the benchmark, once re-run, should report CatSight at
≈0.75 rather than the sign-inverted ≈0.25. Recommendations 3–7 in §5 (disclose the replaced step (ii),
optionally use all ID classes, harden component selection, add authors, state paper inaccessibility)
remain open and were intentionally left out of this minimal, low-risk fix.
