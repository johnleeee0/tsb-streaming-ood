# DiMMAD Faithfulness Verification — ADAPTATION (metric ensemble restored 2026-08-21; see FIX APPLIED)

**Method id:** `dimmad` / `dimmad_enh`
**Paper (actual):** Chaini, Bianco & Mahabal, *In Search of the Unknown Unknowns: A Multi-Metric
Distance Ensemble for Out of Distribution Anomaly Detection in Astronomical Surveys*,
NeurIPS ML4PS 2025 (arXiv:2510.23702). The method is named **DiMMAD** (Distance Multi-Metric Anomaly
Detection).
**Benchmark variant:** `methods/dimmad/dimmad_enh/dimmad_enh.py` (`DiMMADEnhDetector`)
**Base variant:** `benchmark1/models/ood_methods/dimmad.py` (`DiMMADDetector`)
**Verified:** 2026-08-20

---

## Verdict

**ADAPTATION — and honestly labelled as one** by the base variant's docstring ("DiMMAD-Lite",
"Adapted for OOD detection on frozen backbone").

**What is faithful:** the two-stage aggregation is reproduced *exactly*, down to the parameter names.
The paper specifies: *"(1) For every distance metric, aggregate the distance of the test object to
every class centroid by a statistic (class agg: min/median) … (2) Then, aggregate these single-metric
scores across all distance metrics by another statistic (metric agg: median)"*. The implementation's
`class_agg='min'` / `metric_agg='median'` (`dimmad.py:101-102`) match this precisely, as does the
orientation (*"rank them from highest (most anomalous) to lowest"*). **The prompt's second question is
answered affirmatively: min-over-classes / median-over-metrics is not merely reasonable — it is
verbatim the paper's prescription.**

**What is not faithful:** the metric ensemble — which *is* the method's contribution ("a robust
ensemble of diverse distance metrics" that "overcome[s] the metric-selection problem"). The paper uses
**16** specific metrics; the base uses 13 that overlap on only **8**, and `dimmad_enh` uses 10 that
overlap on only **7** (§4).

**The prompt's premise about Hamming/Jaccard/Dice is half right, and the correction over-reached.**
Hamming and Dice are *not* in the paper's 16 — dropping them is correct. But **Jaccard is**, and the
paper's Figure 1 plots it over a *continuous* 2-D feature space with well-behaved contours
(0.15–0.75). The real defect was using **scipy's boolean `jaccard`** rather than the continuous
Jaccard/Ružička variant DistClassiPy provides. `dimmad_enh` fixes the symptom by deleting the metric,
which removes a genuine ensemble member and leaves its paper-overlap *worse than the base's*.

**The benchmark used `dimmad_enh`** — 40 datasets, versus 4 for the base.

---

## 1. Source accessibility — three corrections to the tracker

| Source | Status |
|---|---|
| `https://github.com/DiMMAD/DistClassiPy` (registry URL, recorded in the clone's git config) | **404 — does not exist.** The tracker's suspicion is confirmed. |
| Local clone `methods/dimmad/reference/` | **FAILED CLONE — unusable.** `HEAD` → `refs/heads/.invalid`, **0 git objects, 0 refs, 0 working-tree files**. The clone never fetched anything, consistent with the 404 above. There is **no local reference** for this method. |
| `https://github.com/sidchaini/DistClassiPy` | **Exists.** *"A python package for a distance-based classifier which can use several different distance metrics"* — 43 built-in metrics (18 previously). Classes `DistanceMetricClassifier`, `EnsembleDistanceClassifier`. |
| `https://github.com/sidchaini/dimmad` | **Exists** — the reproduction code (5 Jupyter notebooks for ELAsTiCC / ZTF-ALeRCE, plus `data/`, `scripts/`, `results/`). **No README**, 3 commits. No scoring source file could be identified from the repository listing alone. |
| Paper (arXiv:2510.23702 / ML4PS 2025 paper #222) | **Obtained.** `ar5iv` was **blocked by network policy** and `arxiv.org/abs` socket-hung; text was extracted from the ML4PS PDF with `pypdf` (9 pages, 30,194 chars). All quotations below come from that extraction. |

**Three things in the tracker prompt need correcting:**

1. **The paper title is wrong.** The tracker gives *"DiMMAD: Distance-Metric Ensembles for Detecting
   Unknown-Unknown Transient Events as OOD Anomalies"*. The actual title is *"In Search of the Unknown
   Unknowns: A Multi-Metric Distance Ensemble for Out of Distribution Anomaly Detection in Astronomical
   Surveys"*. The base variant's docstring (`dimmad.py:5-9`) already has it right.
2. **`sidchaini/distclassipy` does not contain the DiMMAD scoring code.** It is *"purely a
   classifier"* — its `score()` returns *"mean accuracy of self.predict(X) wrt. y"*, and it has no OOD
   or anomaly scoring at all. The paper confirms the split: *"DiMMAD is implemented within
   DistClassiPy … while all code to reproduce the results of this paper is available here:
   https://github.com/sidchaini/dimmad/."* The library supplies the metrics; the scoring lives in the
   `dimmad` repo's notebooks.
3. **The registry link is broken and the clone is empty** — worth fixing in
   `tracker/PASS1_VERIFICATION_PROMPTS.md` and re-cloning `sidchaini/dimmad` + `sidchaini/distclassipy`.

### What the paper specifies

- Framework: *"builds on DistClassiPy … classification is based on the 'distance' of the test object
  features to the centroids of training object features."*
- Hypothesis: *"a 'true novelty' will be distant from the centroids of all known classes across a
  majority of distance metrics."*
- Scale: *"we calculate the distances of all test objects to all known classes for **all 16 distance
  metrics**."*
- Aggregation: *"(1) … class agg: min/median … (2) … metric agg: median."*
- Orientation: *"rank them from highest (most anomalous) to lowest (least anomalous)."*
- Figure 1 caption: *"A visualization of 15 (of total 16) distance metrics"* — panels label
  **Euclidean, Braycurtis, Canberra, Cityblock, Chebyshev, Clark, Cosine, Hellinger, Jaccard,
  Lorentzian, Meehl, Motyka, Soergel, Wave-Hedges, Kulczynski**, with *"The correlation metric is
  omitted from this plot"* → 16 total.

## 2. Divergence table

| Component | Paper / DistClassiPy | `dimmad.py:line` (base) | `dimmad_enh.py:line` | Changes results? |
|---|---|---|---|---|
| **Metric ensemble** | **16 named metrics** (Fig. 1 + correlation) | 13 metrics, 8 shared with paper (`:68-82`) | 10 metrics, **7 shared** (`:18-27`) | **YES — see §4** |
| Class aggregation | `min`/`median` over class centroids | `min` default (`:101`, applied `:230-231`) | inherited | **No — exact match** |
| Metric aggregation | `median` over metrics | `median` default (`:102`, applied `:243-244`) | inherited | **No — exact match** |
| Orientation | higher = more anomalous | higher = OOD (`:171`) | inherited | **No** |
| Distance target | class centroids of training features | class means (`:143-148`) | inherited | See below |
| **Central statistic** | DistClassiPy `central_stat` defaults to **median** | `np.mean` (`:145`) | inherited | **YES — likely** |
| **Per-feature dispersion scaling** | DistClassiPy scales distances by std or IQR per dimension | **absent** | absent | **YES — likely** |
| Feature source | astronomical light-curve features | deep features from frozen backbone (`:131`) | inherited | Domain adaptation, disclosed |
| Fitting | centroids from training data | centroids from ID data (`:120-164`) | inherited | **No** |
| Unsupervised fallback | n/a | single global centroid if `y_id is None` (`:150-153`) | inherited | Extra; not in paper |
| Silent metric fallback | n/a | `except: dist = euclidean(...)` (`:219-221`) | inherited | **Risk — see §6** |

## 3. What `dimmad_enh` and `CHANGES.md` get right and wrong

`dimmad_enh` overrides one line — the metric list (`:27`) — and inherits everything else.

**Right:** the diagnosis that scipy's boolean metrics distort the median is well-founded, and the
artefacts corroborate it. `scipy.spatial.distance.hamming/jaccard/dice` coerce inputs to boolean via
"non-zero", so on dense continuous features they are near-constant (hamming ≈ 1, jaccard ≈ 0,
dice ≈ 0). With 3 of 13 metrics pinned to constants, the median over metrics gets dragged toward them —
and indeed the base variant's score range is a compressed `[1, 6.34]` versus `dimmad_enh`'s
`[0.139, 2.108e6]`. Removing them genuinely un-pinned the aggregate.

**Wrong:** Hamming and Dice are absent from the paper's 16, so removing them restores fidelity. But
**Jaccard is present**, and the paper demonstrates it on continuous features. The correct fix was to
substitute DistClassiPy's continuous Jaccard/Ružička implementation, not to delete the metric. As a
result `dimmad_enh` overlaps the paper on 7 metrics where the base overlapped on 8 — the "correction"
moved *away* from the paper on ensemble membership while moving *toward* it on numerical sanity.

`CHANGES.md` states the three metrics *"are defined for boolean vectors and are ill-defined on the
continuous deep features used here"*. True of **scipy's implementations**; **not** true of Jaccard as
the paper uses it. The note should be amended.

## 4. Metric-set comparison

Against the paper's 16 (`cityblock` ≡ `manhattan`):

| | count | shared with paper | not in paper | missing from paper's 16 |
|---|---|---|---|---|
| **Paper** | 16 | — | — | — |
| **base `dimmad`** | 13 | **8** | 5: `dice`, `hamming`, `mahalanobis`, `minkowski`, `standardized_euclidean` | 8: `clark`, `hellinger`, `kulczynski`, `lorentzian`, `meehl`, `motyka`, `soergel`, `wave-hedges` |
| **`dimmad_enh`** | 10 | **7** | 3: `mahalanobis`, `minkowski`, `standardized_euclidean` | 9: the eight above **+ `jaccard`** |

Shared by both: `braycurtis`, `canberra`, `chebyshev`, `cityblock`, `correlation`, `cosine`,
`euclidean` (+ `jaccard` for the base).

So **more than half the paper's ensemble is absent** from either variant, and three metrics the paper
does not use are present. Since the paper's whole argument is that a *diverse* ensemble of geometries
provides robustness, an ensemble sharing 7 of 16 members is a different ensemble — hence ADAPTATION
rather than CORRECTED.

## 5. Measured results

`dimmad_enh` (the benchmark variant):

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 40 | **0.7696** | — |
| TSB-U (univariate) | 21 | **0.7902** | **2/21** |

**This is the second-strongest detector verified so far**, behind SRS (TSB-U 0.8408) and far ahead of
every logit-based method (MSP 0.342, EBO 0.277, GradNorm 0.247, SCALE 0.260, ODIN 0.261). Consistent
with the broader pattern that feature-space distance methods dominate on this benchmark.

Paired against the base on the **4** shared datasets:

| Dataset | `dimmad_enh` | `dimmad` | Δ |
|---|---|---|---|
| TSB-M-DRIFT003 | 0.468 | 0.457 | +0.011 |
| TSB-U-DRIFT024 | 0.588 | 0.608 | −0.020 |
| TSB-U-OOD009 | 1.000 | 1.000 | +0.000 |
| TSB-U-STABLE001 | 0.592 | 0.612 | −0.020 |
| **mean** | **0.6619** | **0.6693** | **−0.0074** |

Better on 1 of 4. As with SCALE, the correction does **not** improve real-data performance on the
available evidence — though at n=4 and with the metric-set confound (§4) this is not a clean test.

Both variants produce non-negative scores throughout, as distances require.

## 6. Minor observations

- **Silent Euclidean fallback is a correctness hazard.** `dimmad.py:217-221` wraps each metric in
  `try/except Exception` and substitutes `euclidean(...)` on any failure. If a metric raises for every
  sample, it is silently replaced by a duplicate Euclidean term, inflating Euclidean's weight in the
  median with no warning. Recommend logging, or failing loudly, so a broken metric cannot masquerade as
  a working one.
- **Internal docstring inconsistency.** `dimmad.py:20` lists 13 metrics; `:27` and `:97` say
  "all 16". The actual `DISTANCE_METRICS` has 13 entries. The "16" appears to be a half-memory of the
  paper's true count — coincidentally correct about the paper, wrong about the code.
- **`minkowski(p=3)`** is annotated *"p=3 for variety"* (`:72`). That is an invented ensemble member,
  not a paper metric — and with `p=3` it is a near-duplicate of Euclidean/Chebyshev in geometry,
  adding little diversity.
- **`mahalanobis` uses a global covariance** (`:157-158`), not class-conditional — reasonable here, but
  note it makes that member closely related to the separate `mahalanobis` detector, whose own
  covariance defect is recorded in `methods/mahalanobis_mds/VERIFICATION.md`.
- **O(N × M × C) Python loop** (`:199-225`) — one scipy call per sample × metric × class. Vectorisable
  via `scipy.spatial.distance.cdist` per metric.

## 7. Recommendations

1. **Fix the tracker entry**: correct the paper title, replace the dead `DiMMAD/DistClassiPy` link with
   `sidchaini/distclassipy` (library) **and** `sidchaini/dimmad` (reproduction code), and note that the
   scoring code lives in the latter's notebooks, not the library.
2. **Re-clone a working reference.** `methods/dimmad/reference/` is an empty failed clone and should be
   replaced or removed; right now it gives the false impression that a reference was consulted.
3. **Restore Jaccard using a continuous implementation** (Ružička / weighted Jaccard, as DistClassiPy
   provides) rather than deleting it, and amend `CHANGES.md` accordingly.
4. **Consider adding the eight missing paper metrics** (Clark, Hellinger, Kulczynski, Lorentzian,
   Meehl, Motyka, Soergel, Wave-Hedges) — all are continuous and available in DistClassiPy, so this
   would move the ensemble from 7/16 to 15/16 with modest effort. Alternatively, state plainly in the
   thesis that the ensemble is a 10-metric subset and report the detector as a DiMMAD-style variant.
5. **Switch the central statistic to median** (DistClassiPy's default) or document the choice of mean.
6. **Replace the silent Euclidean fallback** with a logged failure.
7. Retain the "DiMMAD-Lite / Adapted" labelling — it is accurate and should stay.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded in
`methods/msp/VERIFICATION.md` §5, `methods/odin/VERIFICATION.md` §7,
`methods/energy_ebo/VERIFICATION.md` §6, `methods/mahalanobis_mds/VERIFICATION.md` §8,
`methods/dfm/VERIFICATION.md` §7, `methods/srs/VERIFICATION.md` §6,
`methods/react/VERIFICATION.md` §6, `methods/dice/VERIFICATION.md` §7,
`methods/scale/VERIFICATION.md` §8 and `methods/gradnorm/VERIFICATION.md` §7 remains unresolved.
`dimmad_enh` covers 21 univariate datasets.

## 8. Conclusion

DiMMAD's *scoring framework* is faithfully reproduced: distances to class centroids, aggregated
min-over-classes then median-over-metrics, ranked highest-as-most-anomalous — matching the paper's
prescription verbatim, including the `class_agg` / `metric_agg` parameterisation. Its *ensemble
content* is not: the paper's 16 metrics overlap the base's 13 on eight members and `dimmad_enh`'s 10 on
seven, with three invented members retained and more than half the paper's metrics absent.
`dimmad_enh` correctly removes Hamming and Dice, which the paper never used, but also removes Jaccard,
which it does — the underlying issue being scipy's boolean implementation rather than the metric
itself. Combined with the shift from light-curve features to frozen deep features, this is an
adaptation, which the base variant's docstring already labels accurately. It is nonetheless the
second-best-performing detector in the benchmark (TSB-U 0.790, below chance on only 2 of 21).

---

## FIX APPLIED (2026-08-20)

**File changed:** `methods/dimmad/dimmad_enh/dimmad_enh.py` only (the variant the benchmark runs).
Base `dimmad.py`, the runner, `base_ood.py` and `__init__.py` were **not** touched.

**What changed.** Two faithfulness defects from §3–§4 were corrected inside the `DiMMADEnhDetector`
subclass:

1. **Continuous Jaccard restored (Ružička / weighted Jaccard).** The previous fix deleted Jaccard
   entirely to escape scipy's *boolean* `jaccard` (which binarises inputs via "non-zero"). That over-
   reached: Jaccard *is* one of the paper's 16 metrics and Figure 1 plots it over a continuous feature
   space. A `ruzicka(u, v) = 1 − Σ min(uᵢ,vᵢ) / Σ max(uᵢ,vᵢ)` function is now added — the continuous
   generalisation of Jaccard that DistClassiPy uses (reduces to the classic Jaccard index on binary
   vectors; equals Soergel on non-negative vectors; a small-epsilon guard handles Σmax≈0). It is
   registered as ensemble member `"jaccard"` with `needs_cov=False`, so it flows through the inherited
   `score()` path unchanged. Ensemble is now **11 continuous metrics** (the prior 10 + continuous
   Jaccard), restoring paper overlap to **8/16** — matching the base and one better than the prior 7/16
   — with no boolean metric pinning the median.

2. **Central statistic mean → median.** `fit()` is overridden so class centroids use the **median**
   central statistic instead of `np.mean` (base `dimmad.py:145`), matching DistClassiPy's default
   `central_stat='median'`. The supervised, unsupervised (global-median), covariance (Mahalanobis) and
   variance (standardized-Euclidean) branches are otherwise identical to the base.

Aggregation (`class_agg='min'`, `metric_agg='median'`), orientation (higher = OOD), the
`BaseOODDetector` interface and the class name `DiMMADEnhDetector` are all unchanged. The honest
"DiMMAD-Lite / Adapted" label still stands.

**Not done (deliberately out of scope).** The eight remaining paper metrics (Clark, Hellinger,
Kulczynski, Lorentzian, Meehl, Motyka, Soergel, Wave-Hedges), per-feature dispersion scaling, and the
silent Euclidean fallback (`dimmad.py:217-221`) were left as-is — the last lives in the base file, which
this fix must not edit. Reaching 15/16 would require those additions and a `distclassipy` dependency.

**Reference note.** `methods/dimmad/reference/` remains an empty failed clone (§1), so the Ružička form
was implemented from its standard definition (Cha 2007 intersection family / DistClassiPy), not copied
from a local reference.

**Smoke test (venv `C:\THESIS\.venv`).** Instantiated `DiMMADEnhDetector` with a dummy `nn.Module`
backbone, fit on random ID data (N=40, 3 classes) and scored random test data (M=17):
- metrics list = `[euclidean, manhattan, chebyshev, minkowski, cosine, correlation, canberra,
  braycurtis, mahalanobis, standardized_euclidean, jaccard]` (11 members).
- `ruzicka(a,b)=1.198` on continuous vectors (not a binarised 0/1), `ruzicka(a,a)=0.0` — confirms the
  metric is continuous and identity-zero.
- supervised: scores shape `(17,)`, dtype float64, all finite, range `[1.081, 1.902]`.
- unsupervised (global-median centroid): scores shape `(17,)`, all finite.
- `SMOKE TEST PASSED`. The full benchmark was **not** run.

**New verdict.** Still **ADAPTATION** (frozen deep features vs light curves; 8/16 metric overlap with
three non-paper members retained and eight paper metrics still absent), but **more faithful than
before**: the continuous Jaccard is restored rather than deleted, and the centroid central statistic now
matches DistClassiPy's median default. The two most actionable single-file recommendations from §7
(items 3 and 5) are now satisfied.
