# TD-IVDM Faithfulness Verification — ADAPTATION (relabel to "KDE density" is warranted; one docstring claim is false)

**Method id:** `tdivdm` · **Paper:** Wang, Zhu, Qin, Han & Yan, *TD-IVDM: A multi-scale concept drift
detection method for time series forecasting tasks*, **Neurocomputing**, August 2025,
doi:10.1016/j.neucom.2025.131120 (PII S0925231225017928)
**Implementation:** `benchmark1/models/ood_methods/tdivdm.py` (`TDIVDMDetector`)
**Verified:** 2026-08-20

---

## Verdict

**ADAPTATION**, and the prompt's proposed relabel to **"KDE density"** is warranted.

**The labelling is honest in most respects but contains one false claim.** The docstring correctly
discloses the adaptation (`:2` "(Adapted)", `:10` "Adaptation for OOD detection on frozen backbone",
`:11` "Original TD-IVDM detects concept drift in time series forecasting") and accurately describes the
mechanism at `:12` and `:16-19`. But **`:13` states "Detects temporal and inter-variable dependencies
at multiple scales"** — that is exactly what the implementation does *not* do. There is no temporal
modelling, no multi-scale decomposition, and no inter-variable structure beyond a 20-component PCA.

**One correction to the prompt's premise.** The prompt calls this "a generic KDE density detector, NOT
the paper's multiscale mechanism". Half right: the paper *does* use **multi-dimensional KDE** — it is
one of TD-IVDM's two pillars, used "to capture inter-variable dependencies". So the KDE choice tracks a
real component of the method rather than being arbitrary. What is missing is the **other** pillar (an
improved **TS2Vec** representation network for time dependencies), the **multi-scale** treatment of
smaller time frames and variable subsets, and the three-stage drift-detection workflow.

**Also: the acronym is wrong in both the tracker and the code.** TD-IVDM stands for **"Time
Dependency – Inter Variable Dependency"**. The tracker renders it "Time-Division and
Variable-Density"; `tdivdm.py:2` renders it "Time Division - Inverse Variable Density Measure".
Neither is correct, and both misdescribe the method as being about *time division* and *density* rather
than *two kinds of dependency*.

---

## 1. Source accessibility — the paper's equations could NOT be read

| Source | Status |
|---|---|
| `methods/tdivdm/reference/` | **ABSENT — directory does not exist.** `methods/tdivdm/` contains only `validation_status.json`. Consistent with the tracker's expectation. |
| Official code | **NONE FOUND.** Targeted searches for a TD-IVDM repository and for a code-availability statement returned no repository. This supports the tracker's "NONE PUBLIC" claim, but it is **absence of evidence** — I did not find a repo, which is weaker than confirming none exists. |
| `doi.org/10.1016/j.neucom.2025.131120` | Redirects to `linkinghub.elsevier.com`; not followed to content. |
| `sciencedirect.com/.../S0925231225017928` | **HTTP 403 Forbidden — paywalled.** |
| `researchgate.net` search | **HTTP 403 Forbidden.** |

**Therefore: the paper's equations were not read, and this report does not verify against them.** The
prompt asked to "verify against the paper's equations only"; that was not possible. What I have is a
**secondary, search-result-derived summary** of the method's components (quoted in §2), not the paper
text. Every statement below about the paper is flagged as coming from that secondary source. No
equation, symbol, or numerical detail of the paper is asserted, and none is guessed.

### What the secondary sources establish about the paper

- Title, venue, date: *"TD-IVDM: A multi-scale concept drift detection method for time series
  forecasting tasks"*, Neurocomputing, 5 August 2025.
- Authors: Xiao-li Wang, Shang-lin Zhu, Li-yang Qin, **Jie Han**, Feng Yan. (`tdivdm.py:8` omits Jie
  Han.)
- Expansion: **"Time Dependency – Inter Variable Dependency Method"**.
- Components: *"the representation learning network **TS2Vec** is improved to extract time
  dependencies and a multi-dimensional **Kernel Density Estimation (KDE)** method is used to capture
  inter-variable dependencies."*
- Motivation for "multi-scale": existing methods *"focus on changes in overall data distribution,
  which are inadequate for tackling local temporal and joint distribution drift of smaller time frames
  and subsets of variables."*
- Workflow: *"three stages: preprocessing, detection, and postprocessing."*
- Task: concept-drift detection for **forecasting**, over a stream — not per-window OOD scoring.

## 2. Divergence table

| Component | Paper (per secondary sources) | Mine (`tdivdm.py:line`) | Changes results? |
|---|---|---|---|
| **Time-dependency branch** | improved **TS2Vec** representation network | **absent** — features come from the shared frozen backbone (`:76`) | **YES — one of two pillars missing** |
| **Inter-variable KDE** | multi-dimensional KDE | `scipy.stats.gaussian_kde` on PCA features (`:96-99`) | **No in kind** — present in spirit |
| **Multi-scale** | local drift over smaller time frames **and variable subsets** | **absent** — one global KDE over one 20-d PCA space | **YES — the title contribution is missing** |
| **Workflow** | three stages: preprocessing / detection / postprocessing | single fit + score (`:65`, `:109`) | **YES** |
| **Task** | concept-drift detection over a stream, for forecasting | static per-window OOD score | **YES** — different problem |
| Dimensionality reduction | not attributed to the paper | PCA, `whiten=True`, ≤20 components (`:89-91`) | Added step |
| Bandwidth | not established from the paper | `bw_method='scott'` (`:98`) | Not comparable |
| Score | drift statistic / test | `−log_density` (`:155`), higher = OOD | Orientation sound |
| `kernel` parameter | n/a | read at `:55`, **never used** — `gaussian_kde` is Gaussian-only. Documented as configurable at `:23`, `:48` | Dead parameter |
| Subsample cap | n/a | `n_samples_kde=1000` via **unseeded** `np.random.choice` (`:82`) | Inert here (N≈82 < 1000) |

## 3. Empirical findings

**All 40 datasets loadable; none degenerate** (0/40 with score range < 1e-12).

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 40 | **0.8378** | — |
| TSB-U (univariate) | 21 | **0.8148** | **2/21** |

**This is the strongest detector verified so far on the full set.** Comparison:

| Method | ALL (n) | TSB-U (n) |
|---|---|---|
| **TD-IVDM** | **0.8378 (40)** | 0.8148 (21) |
| SRS | 0.7864 (29) | **0.8408 (20)** |
| M2N2 | 0.7565 (40) | 0.7953 (21) |
| DiMMAD | 0.7696 (40) | 0.7902 (21) |
| DIVERSIFY | 0.6993 (37) | 0.6598 (18) |

SRS edges it on TSB-U but covers only 29 of 40 datasets, so TD-IVDM's full-set figure is the more
comparable number and it is the highest.

**Two caveats on that headline result, both worth stating in the thesis:**

1. **16 of 40 datasets score exactly 1.000** — 40% of the benchmark at a perfect score. These include
   the extreme-feature-magnitude family (`TSB-M-STABLE_015` with scores up to `1.56e13`,
   `STABLE_020`, `STABLE_043`, `STABLE_083`, `TSB-U-DRIFT_060`) already recorded in
   `methods/energy_ebo/VERIFICATION.md` §3. On those datasets the backbone emits features of order
   10⁵–10⁶, so *any* density or distance method separates them trivially. The mean is materially
   inflated by datasets where the task is degenerate rather than by detector quality.
2. **The score range is `[6.539, 1.558e13]`, 100% positive.** That is consistent with
   `−log_density` in a 20-dimensional whitened space (typical values 19–25) and confirms the
   documented formula. It also confirms **the fallback paths never fired**: both fallbacks
   (`:147`, `:152`) produce `−d/d.std()`, bounded in roughly `[0, 2]`, and the observed minimum is
   6.539 — well above that. So all 40 datasets used the real KDE.

**A latent defect in one fallback.** `:150-152` computes the reference point as
`np.mean(feats_pca, axis=0)` — the mean of the **test** batch, not the training set. (The sibling
fallback at `:145` correctly uses `self.kde.dataset`.) Demonstrated: adding 10 far-out samples to a
50-sample batch changes the scores of the *unchanged* 50 by mean 3.24, with Spearman falling to
**0.4326**. Identical samples would score differently depending on batch composition. Unreachable
unless `gaussian_kde()` raises during fit — which it did not here — but it should be fixed.

## 4. Recommendations

1. **Relabel to "KDE density"** in the registry and results tables, as the tracker proposes. The
   detector is a Gaussian KDE density estimator on PCA-reduced frozen features; that is an accurate and
   self-contained description.
2. **Delete or fix `tdivdm.py:13`** — *"Detects temporal and inter-variable dependencies at multiple
   scales"* is false and asserts precisely the mechanism that is absent. This is the one genuine
   labelling failure.
3. **Fix the acronym** at `:2`: TD-IVDM is "Time Dependency – Inter Variable Dependency", not "Time
   Division - Inverse Variable Density Measure". Correct the tracker entry too, and add Jie Han to the
   author list at `:8`.
4. **Remove the `kernel` parameter** (`:55`, and its documentation at `:23`, `:48`) or implement it —
   `scipy.stats.gaussian_kde` cannot use a non-Gaussian kernel, so the option is misleading.
5. **Fix the `kde is None` fallback** at `:150-152` to use the stored training mean, and seed
   `np.random.choice` at `:82`.
6. **Disclose the 16/40 perfect scores** and the extreme-magnitude confound when reporting this as the
   top method. The result is real but partly attributable to a degenerate subset of the benchmark.
7. **State in the thesis that the paper was paywalled** and that the comparison rests on secondary
   descriptions of the method — this method has the weakest verification evidence of any so far, since
   there is neither code nor accessible paper text.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded
across `methods/*/VERIFICATION.md`. `tdivdm` covers **21** univariate datasets; see
`methods/diversify/VERIFICATION.md` §4 for a candidate explanation of the "18" figure.

## 5. Conclusion

The implementation is a Gaussian kernel-density estimator fitted to 20-component whitened PCA
projections of frozen backbone features, scored by negative log-density. Multi-dimensional KDE is
genuinely one of TD-IVDM's two components, so the choice is not arbitrary — but the TS2Vec
time-dependency branch, the multi-scale treatment of time frames and variable subsets, the three-stage
workflow, and the streaming drift-detection task are all absent. The docstring is honest about being an
adaptation except at `:13`, which claims the multi-scale dependency modelling that is missing; that line
should go, and the detector should be reported as "KDE density". Verification confidence is the lowest
of any method reviewed: no public code was found and the paper is paywalled (HTTP 403 on both
ScienceDirect and ResearchGate), so **no comparison against the paper's equations was possible** and
none is claimed here. Empirically it is the strongest detector on the full 40-dataset set (0.8378),
though 16 of those datasets return exactly 1.000 and several belong to the extreme-feature-magnitude
family where any density method succeeds trivially.

## CLASS-D BUILD (2026-08-21)

An **inspired** ordered multi-scale reconstruction was built as a **separate Class-D appendix study**
(never a leaderboard row). Consistent with the finding above that a truly faithful TD-IVDM is
unattainable here, it is captioned honestly and makes no faithfulness claim.

- **File:** `methods/tdivdm/classd/tdivdm_classd.py` (`TDIVDMClassD`); orchestrated by
  `experiments/run_class_d.py` (registry entry `tdivdm`, `eval_mode="ordered_per_window"`).
- **Caption (binding):** *"TD-IVDM-inspired (unverifiable — paper paywalled, no code)."*
- **What it does:** reconstructs the *shape* of the method's two pillars, not the method —
  (a) **time-dependency representation** → the shared frozen backbone embedding stands in for the
  unavailable improved-TS2Vec branch (disclosed substitution); (b) **inter-variable KDE** →
  multi-dimensional Gaussian KDE; (c) the **multi-scale** contribution that the production impl lacked →
  the density is measured in several PCA subspaces of increasing budget (`scales=[5,10,20]`), and the
  per-window neg-log-densities are standardised per scale and aggregated. Higher = more OOD.
- **Eval mode:** ordered per-window on `load_tsb(ordered_eval=True)["stream"]` → `per_sample_auroc`
  directly.
- **Honest caveats:** the TS2Vec branch, the paper's variable-subset scaling, and the three-stage
  streaming workflow are still not reproduced; "scales" here are PCA-budget scales in embedding space,
  not the paper's time-frame/variable-subset scales. This remains the weakest-fidelity of the Group-II
  builds; no comparison against the paper's equations is possible or claimed.
- **Smoke (TSB_N_PER_CELL=2, U only):** finite per-sample AUROCs in `results/class_d_group2.csv` (e.g.
  OOD_001 = 0.958, STABLE_002 = 0.981, DRIFT_002 = 0.629). No production file was modified.
