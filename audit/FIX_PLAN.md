# FIX PLAN — Faithful Reproduction of the 21 Non-Faithful OOD Detectors

**Author:** Stylianos Giannoulis · AUTH MSc Data and Web Science · Supervisor: John Paparrizos
**Date:** 2026-08-20
**Scope:** Planning only. No production code was edited to produce this document.

## How to read this

Each method is assigned a **feasibility class** relative to the *current* benchmark protocol
(`experiments/run_experiments.py` + `experiments/tsb_benchmark.py`): frozen 1-D ResNet backbone
(CE on 4 temporal pseudo-classes), post-hoc detectors with a linear head, **shuffled** evaluation
windows, CPU-only, **no auxiliary outlier dataset**, source-boundary split, seed 42.

- **A = CODE-FIX-NOW** — self-contained code change; faithful reproduction achievable within the
  current protocol.
- **B = NEEDS-HARNESS-CHANGE** — feasible, but requires changing the eval harness/protocol
  (ordered-stream feeding, an auxiliary outlier corpus, or a training loop over the backbone).
- **C = RESEARCH-SCALE** — faithful reproduction is a large rewrite (e.g. adversarial representation
  learning that retrains the feature extractor).
- **D = INCOMPATIBLE** — cannot be made faithful on this protocol at all; should be **excluded or
  explicitly relabelled**, not "fixed".

---

## Summary triage table

| method | current verdict | class | effort | one-line fix |
|---|---|---|---|---|
| react | CORRECTED (`react_enh` faithful) | A | S | None needed — `react_enh` is faithful; only reword `CHANGES.md` (ReAct+MSP is a supported variant, not a defect) |
| scale | CORRECTED (`scale_enh` faithful, wrong percentile) | A | S | Change `scale_enh` default `percentile` 65→85 (paper's validated value) and re-run |
| gradnorm | CORRECTED (`gradnorm_enh` faithful) | A | S | None needed — `gradnorm_enh` is faithful; quarantine base `gradnorm.py` and delete its false citation |
| mahalanobis | CORRECTED — fix required (no `_enh`) | A | S | Use within-class scatter: subtract each class mean before pooling covariance (`mahalanobis.py:75`) |
| dfm (dfm_pca) | ADAPTATION (faithful to TS-OOD target) | A | S | Already faithful to TS-OOD; optionally swap fixed `n_components=32` for variance-retention; fix citation |
| srs | ADAPTATION (seasonal ratio absent) | A | S | Restore the ratio at `srs.py:452`: return `neg_elbo_sig / neg_elbo_res` (both already in scope) |
| dimmad (dimmad_enh) | ADAPTATION (metric ensemble divergent) | A | S–M | Restore a continuous Jaccard (Ružička) and switch central stat mean→median; optionally add missing paper metrics |
| diversify | ADAPTATION (three corrections; no OOD score in original) | D | S (relabel only) | Original defines no OOD score and needs adversarial backbone retraining — relabel as `diversify_lite`, cannot be "faithful" here |
| m2n2 | ADAPTATION (honest label) | B | S (as-is) / M (faithful) | On-protocol: reset `trend_mean` in `score()`. Faithful needs ordered stream + raw-series TTA AE (harness change) |
| tdivdm | ADAPTATION (relabel to "KDE density") | D | S (relabel only) | No public code + paywalled paper; multi-scale/TS2Vec pillars absent — relabel as KDE density, faithful undefined |
| catsight | ADAPTATION (orientation wrong) | A | S | Remove the negation at `catsight.py:199` (mean AUROC 0.25→0.75); optionally use all 4 pseudo-classes |
| driftlens | ADAPTATION (relabel to "PCA-Mahalanobis") | D | S (relabel) / L (faithful) | Window-level Fréchet needs coherent, ordered windows — impossible on shuffled per-sample eval; relabel + disclose it duplicates mahalanobis |
| dice (dice_enh) | NOT-THE-METHOD (both variants) | A | S | Add `fit()` building a static ID-mean contribution mask (global p=90) + sparsify head **weights**, then energy |
| codit | NOT-THE-METHOD | A | M | Restore multi-draw random transforms + correct Fisher term count + flip orientation + seed the split |
| invad | NOT-THE-METHOD (inert reconstruction) | A | S–M | Feed a constant (not `z_ood`) into the inverse pass so reconstruction error is informative; add half-permutation |
| ae_adwin_lstm | NOT-THE-METHOD (temporal comps dead on shuffled windows) | D | S (relabel) | LSTM/ADWIN need ordered stream; per-window OOD API is incompatible with the paper's forecasting+incremental-update — relabel as AE + flip sign |
| deedee | NOT-THE-METHOD (`deedee`); `deedee_fix` is faithful but never wired | A | S | Wire `DEEDEEFixDetector` (`deedee_fix.py`) into the runner and run it; drop base `deedee` |
| outlier_exposure | NOT-THE-METHOD (no training) | D | S (relabel) | Needs an auxiliary outlier corpus + classifier fine-tuning — neither exists; relabel as "Energy (EBO)" |
| divoe | NOT-THE-METHOD (no synthesis/training) | D | S (relabel) | Needs aux outliers + PGD synthesis + fine-tuning; remove inert mean-centring and relabel/exclude as energy |
| diversemix | NOT-THE-METHOD (fabricated outliers) | D | S (relabel) | Needs a real aux outlier corpus + end-to-end training — report as a negative result (at chance either orientation) |
| diffad | NOT-THE-METHOD (denoises from pure noise) | A | S | Wire `diffad_fix.py` (input-conditioned partial-noise reconstruction, non-negated) into the runner; verify `t0` |

### Class counts

- **A = CODE-FIX-NOW: 13** — react, scale, gradnorm, mahalanobis, dfm, srs, dimmad, catsight, dice, codit, invad, deedee, diffad
- **B = NEEDS-HARNESS-CHANGE: 1** — m2n2
- **C = RESEARCH-SCALE: 0**
- **D = INCOMPATIBLE (exclude/relabel): 7** — diversify, tdivdm, driftlens, ae_adwin_lstm, outlier_exposure, divoe, diversemix

**Note on class C:** no method lands in pure C. The nearest candidates (diversify's adversarial
representation learning; m2n2's raw-series TTA AE) resolve to D and B respectively, because
diversify's original defines *no* OOD score even after retraining, and m2n2's dominant blocker is the
shuffled-window protocol rather than the model size.

---

# Per-method detail

## 1. ReAct — CORRECTED — Class A — Effort S

**1. Divergences (production code).** `react_enh` is faithful to the paper's headline configuration
(penultimate activations clipped at the 90th ID percentile, then energy). The clip
(`react_enh.py:42`), the global-over-samples-and-channels percentile (`react_enh.py:34`, matching
`reference/compute_threshold.py:96`), the percentile value (90), and the energy base score
(`react_enh.py:48`) all match. Only cosmetic divergences: metric-invariant sign flip (aligns with the
paper convention), an inert temperature default, and the missing 2000-sample cap on threshold
estimation (immaterial at ~82 windows). The base `react.py` implements ReAct+**MSP** (`react.py:36`),
a paper-supported non-headline configuration.

**2. Exact fix.** None required for faithfulness. Reword `react_enh/CHANGES.md` to describe the base
variant as "ReAct+MSP (a supported, non-headline configuration)", not an "inconsistency". Either run
base `react` on all 40 datasets or retract the "decisive Phase 2 comparison" claim (currently n=4).

**3. Files to edit.** `C:\THESIS\methods\react\react_enh\CHANGES.md`;
`C:\THESIS\methods\react\react_enh\validation_status.json` (wording only). No production code change.

**4. Feasibility.** A — the benchmark already runs the faithful variant (`react_enh`, 40 datasets).

**5. Effort / risk.** S. No risk to headline results — nothing changes numerically.

---

## 2. SCALE — CORRECTED — Class A — Effort S

**1. Divergences.** `scale_enh` reproduces SCALE's scaling **numerically exactly** (verbatim-port
comparison, max abs diff ≤ 5.7e-6): penultimate activations scaled by `exp(s1/s2)`, pruning used only
to compute `s2`, original (not pruned) activations scaled, then head + energy (`scale_enh.py:36-50`).
One substantive divergence: **percentile = 65** (`scale_enh.py:30`) versus the paper's validated
**p=0.85** and the official config's `percentile: 85`. At p=65 the mean sharpening factor is ≈2.94 vs
≈5.55 at p=85 — roughly half; this is *not* rank-preserving, so AUROC genuinely changes. The base
`scale.py` z-standardises **logits** (`scale.py:30`) — wrong layer and operation, and adds a fit stage
SCALE does not have.

**2. Exact fix.** Set `scale_enh` default `percentile=85` (`scale_enh.py:30`) and re-run, or justify 65
with a validation sweep. The added `torch.relu(feats)` (`scale_enh.py:37`) is a required, documented
adaptation for the linear-embedding backbone — keep it. Optionally drop the never-binding
`clamp(max=50.0)`.

**3. Files to edit.** `C:\THESIS\methods\scale\scale_enh\scale_enh.py` (line 30);
`C:\THESIS\methods\scale\scale_enh\CHANGES.md` (correct the "Phase 2 vindicates" claim: measured
Δ = −0.025 on n=4).

**4. Feasibility.** A — per-sample post-hoc score, fully compatible with the frozen backbone / shuffled
windows / CPU.

**5. Effort / risk.** S. Re-running at p=85 changes SCALE's numbers, but SCALE is a low performer
(TSB-U mean 0.26), so no headline detector is affected.

---

## 3. GradNorm — CORRECTED — Class A — Effort S

**1. Divergences.** `gradnorm_enh` is faithful: KL-to-uniform via ones-target CE, gradient w.r.t. the
**last-layer weights** (`gradnorm_enh.py:57`), **L1** norm, T=1, negation for orientation. Two
non-obvious equivalences (detached-feature head gradient; ones-target loss = `K·KL + K·log K`) verified
to 0.000e+00. The base `gradnorm.py` is wrong on all four axes — gradient w.r.t. **input**
(`gradnorm.py:19,23`), **CE-to-argmax** loss (`:20-21`), **L2** norm (`:24`), and **inverted**
orientation (`:25-27`) — and its comment (`:25`) misattributes the orientation to Huang et al.,
which states the opposite.

**2. Exact fix.** None required for faithfulness. Delete or quarantine `gradnorm.py`; if retained for
the ablation, remove the false citation at `gradnorm.py:25`. Consider raising instead of the degenerate
feature-L1 fallback at `gradnorm_enh.py:38-41`.

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\gradnorm.py` (quarantine + remove
citation). No change to the faithful variant.

**4. Feasibility.** A — benchmark already runs `gradnorm_enh` (40 datasets).

**5. Effort / risk.** S. No risk — the reported GradNorm number already comes from the faithful variant
(and it is the one `_enh` correction that clearly helps: +0.122 over base on n=4).

---

## 4. Mahalanobis (MDS) — CORRECTED, fix required — Class A — Effort S

**1. Divergences.** Structure is correct (class-conditional Gaussians on pre-logit features, tied
covariance, min Mahalanobis distance). **The tied covariance is computed as the *total* covariance,
not the within-class scatter.** `mahalanobis.py:75,78` pools **raw** features and
`EmpiricalCovariance().fit(...)` (`:81`) centres on the **global** mean, yielding
`Σ_T = Σ_W + Σ_B`. Lee et al. Eq. 1, the official `sample_estimator`
(`reference/.../lib_generation.py:112-114`), and TS-OOD's explicit "tied covariance" all require the
**within-class** scatter (each sample centred on its own class mean before pooling). This is not
rank-preserving and systematically degrades the score (synthetic: monotonic AUROC loss up to ~0.008,
Spearman ρ falling to ≈0.975 as class separation grows). The two documented omissions (FGSM
perturbation, multi-layer LR ensemble) are confirmed consistent with the TS-OOD target. No `_enh`
variant exists, so the benchmark's MDS numbers were produced with the defective covariance.

**2. Exact fix (3 lines).** Center each class's features on its own mean before pooling. In the
per-class loop, `mahalanobis.py:75`:

```python
# replace:  all_feats_list.append(feats_class)
all_feats_list.append(feats_class - self.class_means[class_label])
```

`self.class_means[class_label]` is already computed on the preceding line (`:74`). Everything
downstream (`:81-92`) is unchanged and correct. Regenerate all `experiments/*/*/mahalanobis/` results
afterward.

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\mahalanobis.py` (line 75).

**4. Feasibility.** A — self-contained, uses ID labels (already required), frozen-backbone and
CPU-compatible, order-independent.

**5. Effort / risk.** S. **Risk to headline results: this is the top-ranked detector (ALL 0.8598).**
The fix changes every MDS score, so results must be regenerated and the leaderboard re-checked. The
synthetic AUROC penalty is small but ranking-affecting; the real-data magnitude is unmeasured (no
cached features exist). Because **driftlens ties mahalanobis at ρ 0.999** (it is per-sample
PCA-Mahalanobis), the fix will also shift the mahalanobis–driftlens relationship — evaluate them
together after regenerating.

---

## 5. DFM-PCA — ADAPTATION (faithful to TS-OOD) — Class A — Effort S

**1. Divergences.** Faithful to the TS-OOD DFM-PCA target: per-ID-class PCA on pre-logit features,
min feature-reconstruction error over class models (`dfm_pca.py:46,57-79,113-117,123`), higher = OOD.
Relative to the *cited* Ahuja 2019 it uses FRE not Gaussian/GMM NLL (matches TS-OOD, not the 2019
paper). L2 vs squared-L2 norm is rank-preserving (no AUROC effect). The one real deviation is the
**component policy**: fixed `n_components=32`, capped `min(32, d, n_c−1)` (`dfm_pca.py:25,68-72`),
whereas references hold *explained variance* constant (0.995 / 0.97). In practice the per-class sample
cap (~19) usually binds, so 32 is rarely operative.

**2. Exact fix.** No algorithmic fix needed for TS-OOD fidelity. Optionally replace fixed
`n_components` with a variance-retention policy (e.g. 0.97) to remove dataset-dependent capacity drift.
Documentation: cite Ndiour et al. (ICIP 2022 / BMVC 2023) as the FRE origin instead of the 2019 paper
(`dfm_pca.py:15-22`, `discrepancy_report.md:4`).

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\dfm_pca.py` (25, 68-72, 15-22);
`C:\THESIS\methods\dfm\discrepancy_report.md`.

**4. Feasibility.** A — one-parameter swap inside `fit`, labels already required, order-independent.

**5. Effort / risk.** S. Low risk — the sample cap already overrides 32 on this loader, so numbers
barely move. Mostly documentation.

---

## 6. SRS — ADAPTATION (seasonal ratio absent) — Class A — Effort S

**1. Divergences.** The titular **seasonal ratio is never formed**: `srs.py:452` returns the signal
neg-ELBO alone. The residual CVAE is trained (`:367-376`) and evaluated (`:441`) then **discarded**.
Additional divergences vs the official (tahabelkhouja/SRS): neg-ELBO includes a KL term (official is
reconstruction-only, `:213-215`); Gaussian/MSE + linear decoder vs official Bernoulli/BCE + sigmoid on
min-max data (`:203,185`); per-channel z-score vs min-max input scaling (`:331-333`); decoder-only
conditioning (`:179`); per-sample STL vs per-class; single-pass xcorr vs iterative DTW alignment.
The test-time min-residual class assignment (`_find_best_class`, `:411,495-510`) is a sound, necessary
adaptation. **The one-line restoration is confirmed feasible:** both operands already exist in scope
(`neg_elbo_sig`, `neg_elbo_res` at `:443-444`).

**2. Exact fix.** Restore the ratio at `srs.py:452`:

```python
# official: ratio = ll_signal / ll_residual  (Run_SRS.py:139,145,178)
return neg_elbo_sig / neg_elbo_res
```

Fix the docstring (`srs.py:9-12`) — remove "faithful PyTorch port"; state the score is the
signal/residual ratio and that the per-sample likelihood deliberately improves on the official
batch-constant behaviour. Deeper fidelity (BCE/sigmoid + min-max, both-half conditioning, per-class
STL, DTW) is optional and larger.

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\srs.py` (452; docstring 9-12).

**4. Feasibility.** A — one-line edit on values already computed; SRS trains its own CVAEs (not the
backbone), so no harness change. SRS is univariate-seasonal and is excluded on TSB-M by design
(`tsb_benchmark.py:93-97`).

**5. Effort / risk.** Ratio restoration: S. **Risk to headline: material and unmeasured** — SRS is the
strongest detector on TSB-U (0.841), but that belongs to the *signal-only* variant; the ratio is
non-monotone relative to signal-only and could move AUROC either way. VERIFICATION.md states the
ratio's impact "is not measurable here". Full generative-model fidelity: L, high risk.

---

## 7. DiMMAD — ADAPTATION (metric ensemble divergent) — Class A — Effort S–M

**1. Divergences.** The min/median aggregation matches the paper verbatim
(`dimmad.py:101-102,230-231,243-244`; orientation higher=OOD `:171`). But the **metric ensemble** —
the method's contribution — is materially divergent: the paper uses 16 named metrics; base `dimmad.py`
has 13 sharing only 8 (`:68-82`); the benchmark variant `dimmad_enh.py` has 10 sharing only **7**
(`:18-27`), and it **deleted Jaccard, which the paper uses** — the real defect was scipy's *boolean*
jaccard, not the metric. Central statistic is `np.mean` (`dimmad.py:145`) vs DistClassiPy's median
default; per-feature dispersion scaling is absent; a silent Euclidean fallback (`:219-221`) hides
failures.

**2. Exact fix.** (i) Restore Jaccard via a **continuous** implementation (Ružička / weighted Jaccard)
instead of deleting it. (ii) Switch central statistic mean→median (`dimmad.py:145`). (iii) Optionally
add the 8 missing continuous paper metrics (Clark, Hellinger, Kulczynski, Lorentzian, Meehl, Motyka,
Soergel, Wave-Hedges) to reach 15/16. (iv) Replace the silent Euclidean fallback with a logged failure
(`:217-221`). Keep the honest "DiMMAD-Lite/Adapted" label.

**3. Files to edit.** `C:\THESIS\methods\dimmad\dimmad_enh\dimmad_enh.py` (metric list 18-27 — the
benchmark variant); `C:\THESIS\benchmark1\models\ood_methods\dimmad.py` (145, 217-221, 68-82);
`C:\THESIS\methods\dimmad\dimmad_enh\CHANGES.md`.

**4. Feasibility.** A — pure changes inside fit/score; needs `distclassipy` or hand-coded metric
formulas; per-sample distances (shuffle-irrelevant), labels available, CPU-fine.

**5. Effort / risk.** Jaccard + median: S. Adding all missing metrics + dependency: M. Moderate,
unmeasured risk — `dimmad_enh` is second-strongest on TSB-U (0.790) and the metric set drives its score
range and aggregation, so restoring members can move results.

---

## 8. DIVERSIFY — ADAPTATION (original defines no OOD score) — Class D — Effort S (relabel)

**1. Divergences.** The feature extractor is **frozen** (`diversify.py:69-70,167-168`); the original
adversarially **retrains** it across latent domains via gradient reversal over seven networks.
"Adversarial" is a misnomer — `alpha` weights a centroid-repulsion term (`:104`), not a discriminator.
Only K centroids are trained (`:77,80`); `y_id` is ignored (`:64`); domain assignment uses Euclidean on
unnormalised features (`:84,174`) vs the official cosine on L2-normalised. **The OOD score is invented**
— min centroid distance (`:174-175`); the original defines *no* OOD score (it is a domain-generalisation
classifier). `validation_status.json` reports CRITICAL:0/MODERATE:0/MINOR:0 / PASS — indefensible.

**2. Exact fix.** There is no faithful-reproduction fix: the paper defines no OOD score, and its
mechanism requires retraining the backbone. Correct action is honest re-framing: replace "adversarial"
with "diversity-regularised" (`:9,32,41,79,99`); rename the registry key `diversify` → `diversify_lite`
(`:35`); fix `validation_status.json`; fix the citation venue/DOI mismatch (`:4-5`). On-protocol
partial nudge: switch domain-assignment/scoring geometry to cosine on L2-normalised features
(`:84,174`) to match `set_dlabel`.

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\diversify.py`;
`C:\THESIS\methods\diversify\validation_status.json`.

**4. Feasibility.** D — faithful reproduction of the core method needs backbone retraining (a harness
change) *and* even then yields no defined OOD score, so any score is an invention. Only cosmetic/honesty
fixes and the cosine alignment are self-contained. Thesis framing: "a feature-space distance detector
inspired by DIVERSIFY's latent-domain idea", not "a simplified DIVERSIFY".

**5. Effort / risk.** Relabel + JSON: S. Cosine alignment: S–M. Cosmetic fixes carry no numeric risk;
the cosine change could shift the current TSB-U 0.660 (third-strongest).

---

## 9. M2N2 — ADAPTATION (honest label) — Class B — Effort S (as-is) / M (faithful)

**1. Divergences.** Runs an AE on pooled frozen features (`m2n2.py:129,199`) with an EMA "detrend"
that is arithmetically identical to the official `Detrender` (verified), and test-time adaptation OFF
by default (matches official). Divergences: detrend applied unconditionally (`:213,155`) vs the
official conditional default; `trend_mean` is **never reset before scoring** and mutates across samples
(`:210`), making scores order-dependent and non-idempotent (~5% deviation, ρ≈0.97, 3.7%
non-idempotent) — and the eval windows are **shuffled**, so the EMA runs over destroyed order. Input
domain (features not raw series) and model (fixed MLP AE) are disclosed adaptations. Docstring gamma
default is stale (`:26` says 0.9; code uses 0.995).

**2. Exact fix.** On-protocol honesty fix: reset (or snapshot/restore) `trend_mean` at the start of
`score()`, mirroring the official `load_trained_model()`; disclose order-dependence; fix the gamma
docstring; add a `detrend` flag so the official no-detrend default is reachable. **Faithful** M2N2
(AE on raw ordered series with real TTA over a stream) needs an ordered stream and per-timestep
scoring.

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\m2n2.py`.

**4. Feasibility.** B — the defining sequential-EMA "trend" is meaningless on shuffled windows, so
true faithfulness needs a **harness change** to feed an ordered stream (plus a moderate model rewrite
to raw-series AE + TTA). The idempotency/docstring fixes are self-contained (A) but only make the
*adaptation* honest, not faithful.

**5. Effort / risk.** As-is fixes: S. Faithful: M + harness change. **Risk to headline: low** — M2N2 is
2nd-strongest on TSB-U (0.795); resetting `trend_mean` only removes non-idempotent noise (ρ≈0.99), so
AUROC is essentially unchanged (though saved scores won't be bit-reproducible until the reset lands).

---

## 10. TD-IVDM — ADAPTATION (relabel to "KDE density") — Class D — Effort S (relabel)

**1. Divergences.** Both paper pillars are absent: the **multi-scale** decomposition (title
contribution) is absent — one global KDE over one 20-d PCA space (`tdivdm.py:89-91`); the improved
**TS2Vec time-dependency** branch is absent — features come from the shared frozen backbone (`:76`).
What remains is `scipy.stats.gaussian_kde` on PCA features (`:96-99`) — a generic density detector. The
`kernel` parameter is read (`:55`) but never used (gaussian_kde is Gaussian-only). A latent
`kde is None` fallback uses the **test-batch** mean not the training mean (`:150-152`). The docstring
(`:13`) claims "temporal and inter-variable dependencies at multiple scales" — exactly what is absent.

**2. Exact fix.** No faithful fix is well-defined (see feasibility). Honesty actions: relabel as "KDE
density" in the registry/results; delete/fix the false `:13` docstring; fix the acronym (`:2`) and add
the missing author (`:8`); remove the dead `kernel` param (`:55,23,48`); fix the `:150-152` fallback to
use the stored training mean; seed `np.random.choice` (`:82`); disclose the 16/40 perfect-score
extreme-magnitude confound.

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\tdivdm.py`.

**4. Feasibility.** D — **"faithful" is not well-defined**: no public code was found and the paper is
paywalled (ScienceDirect/ResearchGate 403), so there is no reference to reproduce. Reconstructing the
actual method (TS2Vec branch + multi-scale time-frame/variable-subset decomposition + three-stage
streaming drift workflow) would be a research-scale rewrite *and* need a streaming protocol, and still
could not be validated. Scope = relabel + honesty fixes only.

**5. Effort / risk.** S (docstring/label/param/bug fixes). Risk is *framing*: tdivdm is the headline
"strongest over the full 40" (0.838), but 16/40 datasets score exactly 1.000 on extreme-magnitude
cases where any density method wins — the relabel/disclosure changes how it must be framed, not the
AUROC.

---

## 11. CatSight — ADAPTATION (orientation wrong) — Class A — Effort S

**1. Divergences.** The CSP machinery (`eigh(C1, C1+C2)`, `catsight.py:68`; component selection
`:71-84`) is verified mathematically correct (max Δλ 8.68e-15). But: step (ii)'s trained ML classifier
is replaced by a normalized distance to the ID centroid (`:198-199`); only the **first two** of four
pseudo-classes are used for the two-distribution CSP (`:136-141`); and, decisively, the **score
orientation is negated on a false premise** (`:199` `-np.linalg.norm(...)`, rationale `:193-197`).
Empirically refuted: as-implemented mean AUROC **0.2499** vs flipped **0.7501**.

**2. Exact fix.** Remove the negation at `catsight.py:199` and delete the rationale at `:193-197`
(one-character load-bearing change worth +0.50 AUROC); re-run and re-report. Optionally use all four
pseudo-classes (multiclass CSP or the temporal half-split fallback `:144-147`) and harden component
selection (`:71-84`).

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\catsight.py`.

**4. Feasibility.** A — the orientation flip is per-sample, order-independent, fully compatible with the
protocol. Full faithfulness is unverifiable (no public code, paywalled paper), but the CSP is
internally correct and the one clear defect is A-fixable. Multiclass CSP is also self-contained.

**5. Effort / risk.** S (flip) / small M (multiclass). **Risk to headline: high but favorable** —
flipping moves CatSight from weakest (0.250) to second-strongest (0.750). Reporting the current 0.250
as CatSight's performance would be materially misleading.

---

## 12. DriftLens — ADAPTATION (relabel to "PCA-Mahalanobis") — Class D — Effort S (relabel) / L (faithful)

**1. Divergences.** The official is a **window/batch-level** Fréchet (Wasserstein-2) distribution
distance with per-label decomposition. This implementation scores **per-sample squared Mahalanobis** to
a single global baseline in PCA space (`driftlens.py:216-235,229-233`); `y_id` is unused (`:106`); the
per-label decomposition is absent. A `frechet_distance()` (`:37-71`) and `score_batch()` (`:242-279`)
are implemented but **never called**; an entire threshold-estimation path (`:134-187`) is computed and
**never used** (dead state, unseeded RNG). The docstring headline (`:2,4,8-10`) announces Fréchet;
only `:12` reveals the actual Mahalanobis behaviour.

**2. Exact fix.** No faithful fix on this protocol (see feasibility). Honesty actions: relabel as
"PCA-Mahalanobis"; fix the docstring headline; remove/mark dead code (`frechet_distance` `:37-71`,
`score_batch` `:242-279`, `_estimate_threshold` `:134-187`); fix the paper title (`:4`); **disclose in
the thesis that it duplicates the `mahalanobis` detector** (median ρ 0.999).

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\driftlens.py`.

**4. Feasibility.** D — faithful window-level Fréchet is **impossible on the shuffled per-sample
protocol**: (i) after shuffling there is no coherent "window" whose distribution can be measured;
(ii) the method emits one score per window, incompatible with per-sample AUROC. `score_batch()` is
unreachable for exactly this reason. A faithful version would require a new window-level evaluation
protocol — a harness rewrite bordering on incompatible.

**5. Effort / risk.** Honesty/relabel/dead-code removal: S (zero score change). Faithful version:
L-to-impossible. **Risk to headline: framing risk is high** — driftlens is #2 (ALL 0.844) and a
near-duplicate of #1 `mahalanobis` (0.860; 24/40 datasets at ρ>0.99), so the top two leaderboard slots
are the **same method measured twice**. This must be disclosed; but the recommended (a) fixes leave
scores untouched. (See also mahalanobis §4 — regenerate both together.)

---

## 13. DICE — NOT-THE-METHOD (both variants) — Class A — Effort S

**1. Divergences.** The defining mechanism — a **static** FC-weight sparsification from a precomputed
ID-mean contribution — is absent in **both** variants. Both compute contribution per test sample
(`dice_enh.py:41`, `dice.py:33`), sparsify nothing (logits recomputed per sample rather than a static
`masked_w`, `dice_enh.py:44`, `dice.py:35`), and rank by **absolute** magnitude
(`dice_enh.py:42`) rather than the official signed `contrib > thresh`. `dice_enh` uses per-class
`k=20` instead of a global percentile `p=90`. Neither has a `fit()` — no ID statistics are used at all.
`dice_enh` correctly restores signed top-k sum + energy (`:43,45`) but **is not faithful** (Spearman
≈0.43 vs the official static mask; 49.2% of its selections are negative units the official keeps at 0).

**2. Exact fix.** Add a `fit()` that builds the static mask and sparsify weights in `score()`:

```python
def fit(self, x_id, y_id=None):
    feats = self._forward_features(self._to_tensor(x_id))   # (N, D)
    info = feats.mean(dim=0).cpu().numpy()                  # ID-mean feature vector
    W = self._classifier().weight.detach().cpu().numpy()    # (C, D)
    contrib = info[None, :] * W                             # signed, input-independent
    thresh = np.percentile(contrib, self.p)                 # global percentile, p=90
    self.mask = torch.from_numpy((contrib > thresh).astype(np.float32))

def score(self, x):
    feats  = self._forward_features(self._to_tensor(x))
    W      = self._classifier().weight * self.mask.to(...)  # sparsified WEIGHTS
    logits = feats @ W.T + bias
    return self._energy(logits).cpu().numpy()               # higher = OOD
```

Parameterisation changes from `top_k=20` to `p=90`. Implement as `dice_enh2` (or replace `dice_enh`)
and re-run.

**3. Files to edit.** `C:\THESIS\methods\dice\dice_enh\dice_enh.py` (or add `dice_enh2.py`); wiring in
`tsb_benchmark.py:62`; optionally `C:\THESIS\benchmark1\models\ood_methods\dice.py`.

**4. Feasibility.** A — the real method needs only ID training data (via `fit(x_id)`), frozen head
weights, and energy; no aux outliers, no ordered stream, no retrain.

**5. Effort / risk.** S. Low-to-moderate risk — `dice_enh`'s claimed advantage over base is only
+0.0004, so the current DICE row is effectively an MSP/energy-variant number.

---

## 14. CODiT — NOT-THE-METHOD — Class A — Effort M

**1. Divergences.** Only **one** p-value is produced per sample (`codit.py:230`); the official
combines `eval_n≈20` p-values from randomly sampled transforms via Fisher. The nonconformity uses the
**identity transform only** (`:210-222`). The Fisher term count is hardcoded `range(20)` applied to a
single p-value (`:260`), saturating every score into `[0.999999999969, 1.0]` (not even monotone). The
**orientation is inverted** (`:239-240`) — flipping moves mean AUROC 0.386→0.615. The calibration split
is unseeded (`:139`). `eval_n` is declared but unused (`:106`). Faithful parts: the five-transform set
(`:33`), the conformal p-value formula (`:230`), and `_fisher_value` itself.

**2. Exact fix.** (i) Fix the Fisher term count to match the number of p-values combined (`:260`).
(ii) Restore multi-draw combination — sample `eval_n` random transforms per test window, one p-value
each, multiply, then `calc_fisher_value(prod, eval_n)`. (iii) Adopt the official orientation
(OOD = −Fisher or `1 − F`). (iv) Seed the calibration split (`:139`). (v) Fix the `:11` vs `:237-239`
docstring contradiction.

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\codit.py`.

**4. Feasibility.** A — all fixes are internal: transforms are computed per window (order-independent),
uses ID calibration data via `fit`, no aux outliers/retrain. The run-length detection stage cannot be
reproduced (windows shuffled) but is not required for per-window scoring — disclose as a limitation.

**5. Effort / risk.** M (restoring multi-draw combination is the main work). Moderate-to-high risk —
19/40 datasets currently produce constant scores and the orientation is backwards, so corrected numbers
move substantially (~0.39 → ~0.61+).

---

## 15. InvAD — NOT-THE-METHOD (inert reconstruction) — Class A — Effort S–M

**1. Divergences.** The inverse pass is fed the exact forward output `cat([z_id, z_ood])`, so
`reconstruct()` is the exact inverse and reconstruction error ≡ 0 (measured MSE ~1e-14;
`invad.py:308-309,312-313`). The official replaces the residual branch with a constant before inverting
(`reference/.../model.py:51`). The 0.6-weighted reconstruction term thus contributes nothing to score
(`:325`) or training (`:253`). The coupling layer leaves the first half untouched (`z1=x1`, `:83`) with
no half-permutation, so `z_id` is bit-identical to `feats[:, :D//2]` — no learned decomposition. Net:
the score provably reduces to `0.4·(1 − max softmax)` of an MLP head (observed range `[0, 0.2987]`).

**2. Exact fix.** Substitute a constant for `z_ood` before inverting so reconstruction error becomes
informative: `z = cat([z_id, torch.full_like(z_ood, res_const)])` (around `invad.py:309`); optionally
add the official second term `MSE(z_ood, const)`. Add a half-permutation between coupling layers (or use
`y1 = x1 + f(x2)`) so `z_id` is an actual learned decomposition (`invad.py:83`). Fix the docstring
(`:14,19-20`).

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\invad.py`.

**4. Feasibility.** A — both fixes are internal to the coupling network / score; training uses ID data
via `fit`, no aux outliers/ordering/retrain, CPU-friendly.

**5. Effort / risk.** S–M. Moderate risk — the current number tracks MSP (TSB-U 0.356); fixing the
collapse yields a genuinely distinct row, with no guarantee it improves.

---

## 16. AE-ADWIN-LSTM — NOT-THE-METHOD — Class D — Effort S (relabel)

**1. Divergences.** Evaluation windows are randomly permuted (`tsb_loader.py:148`
`idx = rng.permutation(idx)`), so the LSTM's 9-window history (`ae_adwin_lstm.py:335,351`) and the
ADWIN error stream (`:368`) run over random order (two shuffles correlate only ρ≈0.80). "ADWIN" is not
ADWIN — a single fixed cut at `n//2` with a std-scaled threshold (`:78,93-94`), not the Hoeffding
bound; `max_buckets` unused (`:56`). The paper's defining incremental model update on drift is absent
(drift becomes a 0.2-weighted binary spike, `:371,375`). Orientation inverted (`:384`; measured 0.253
vs 0.747 flipped). Positional artefact: first `seq_len−1` windows get `pred_error_norm=0.0` (`:349,353`)
— which windows is arbitrary because order is random. ADWIN never reset in `score()` (non-idempotent).

**2. Exact fix.** No faithful fix on this protocol. On-protocol cleanup: flip orientation and delete
the rationale (`:384,378-383`); drop the non-functional LSTM + ADWIN terms and report honestly as an AE
reconstruction score (the only order-invariant component); fix the positional artefact (`:349,353`);
reset ADWIN at the start of `score()`.

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\ae_adwin_lstm.py` (and `tsb_loader.py`
only if order were to be preserved).

**4. Feasibility.** D — the LSTM prediction and ADWIN drift stream require temporally ordered windows;
the harness shuffles them (destroying ~20% of the ranking). Restoring order is a harness change (B),
but even ordered, the paper is a forecasting + drift-triggered incremental-update system, not a
per-window OOD scorer — incompatible with the per-window OOD API. Only the flip + reduce-to-AE cleanup
is available on-protocol.

**5. Effort / risk.** Flip + reduce-to-AE: S. Faithful: effectively impossible here. High risk on the
sign alone (0.25 → 0.75 if flipped) — do not report the current 0.253 as this method's performance.

---

## 17. DEEDEE — NOT-THE-METHOD (`deedee`); `deedee_fix` faithful but never wired — Class A — Effort S

**1. Divergences (`deedee.py`, the variant that produced results).** The "episodewise mean" is not a
mean — `mean_stat = feat_vec[dim]`, a single raw feature value (`deedee.py:110,211`; paper:
`μ = (1/w) Σ_t x_t`). RBF similarity is computed over adjacent **embedding dimensions** (unordered),
not over time and not against a training summary (`_compute_rbf_similarity` `:131-168`, comment `:113`).
It operates on frozen features `(N,d)` (`:85`) not raw trajectories, and **inverts the paper's speedup**
— up to 150,528 `IsolationForest.score_samples` calls per dataset (`:209-222`), one forest per feature
dim (`:122-129`). Base `deedee` is excluded from the scaled sweep (4 ablation dirs only).

**2. Exact fix.** **Wire `deedee_fix` into the runner and run it** (or drop DEEDEE). Confirmed contents
of `C:\THESIS\methods\deedee\deedee_fix\`: `deedee_fix.py` (58 lines, `DEEDEEFixDetector(DEEDEEDetector)`),
`CHANGES.md`, `validation_status.json`. `deedee_fix.py` **is a substantially faithful implementation
that has zero result directories** — it computes the episodewise mean over the real time axis
(`x.mean(axis=2)`, `:35`), RBF over consecutive-timestep differences (`:37-38`), fits **one vectorised**
isolation forest (`:47-49`), and operates on raw windows `(N,C,T)` via `x["x"]` (`:44,54`) as in the
paper. `CHANGES.md`: *"faithful `deedee_fix` AUROC 0.989 (original 0.982). PASS. Use `deedee_fix`."* —
an instruction never carried out. Remaining gap to fully close: its RBF term is a consecutive-timestep
**self**-similarity (`:37-38`), whereas the paper specifies similarity *to a training summary* — store
an ID summary at `fit()` and measure RBF to it.

**3. Files to edit.** `C:\THESIS\experiments\run_experiments.py` and
`C:\THESIS\experiments\tsb_benchmark.py` (register/wire `DEEDEEFixDetector`; note DEEDEE is O(N²) KDE-
adjacent and was excluded from the scaled sweep for cost — `deedee_fix` is vectorised and cheap, so it
can be included); `C:\THESIS\methods\deedee\deedee_fix\deedee_fix.py` (only to close the RBF gap);
`C:\THESIS\benchmark1\models\ood_methods\deedee.py` (docstring `:17-18` if reported at all).

**4. Feasibility.** A — `deedee_fix` receives raw windows in `fit`/`score` (paper-consistent), needs no
aux outliers, no ordered stream (within-window, order-invariant statistics), no retrain.

**5. Effort / risk.** Wire + run: S (vectorised, CPU-fine). Close the RBF training-summary gap: M.
Low risk — base `deedee` contributes only 4 ablation datasets, so this **adds** a genuine DEEDEE row
rather than perturbing headline numbers.

---

## 18. Outlier Exposure — NOT-THE-METHOD (no training) — Class D — Effort S (relabel)

**1. Divergences.** OE's defining mechanism is fine-tuning the classifier against an **auxiliary
outlier dataset** with a CE-to-uniform penalty (`reference/.../oe_tune.py:172-177`, 80M TinyImages).
This implementation does no training: `fit()` is the inherited no-op (`base_ood.py:79-80`); `score()`
returns `_energy(logits/T)` on the shared ID backbone (`outlier_exposure.py:19-25`). No aux dataset,
no OE loss, no parameter updates. It also **duplicates the EBO results** — the same numbers
(ALL 0.2949 / TSB-U 0.2770) are reported under both `outlier_exposure` and `energy_ebo`.

**2. Exact fix.** Not a scoring fix. Rename the registry entry to `energy`/`energy_ebo` and drop the
`outlier_exposure`/`oe` aliases (`:11-12`), or report the row as "Energy (EBO)"; collapse the duplicate
EBO/OE rows; add a docstring stating OE's auxiliary-outlier fine-tuning is not implemented.

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\outlier_exposure.py` (docstring +
registry); reporting table (collapse duplicate row — thesis-side).

**4. Feasibility.** D — faithful OE requires an auxiliary outlier corpus **and** an unfrozen trainable
classifier; the protocol structurally lacks both. It is legitimately an energy baseline, just
mislabelled.

**5. Effort / risk.** S. Relabelling/collapsing removes a duplicated row (one detector currently
counted as up to three across EBO/OE/DivOE) — corrects double-counting, changes no per-detector number.

---

## 19. DivOE — NOT-THE-METHOD (no synthesis/training) — Class D — Effort S (relabel)

**1. Divergences.** DivOE = OE + multi-step PGD outlier synthesis
(`reference/.../train_DivOE.py:177-197`). This implementation performs neither synthesis nor training:
`fit()` only stores `self.logit_mean` (`divoe.py:19-21`); `score()` computes `_energy` on
**mean-centred logits** (`centered = logits - self.logit_mean`, `:27-28`) — an invented step with no
counterpart in paper/code, near-inert (Spearman 0.99991 vs plain energy) and slightly **harmful**
(0.2763 vs 0.2949, −0.0186). It is a third redundant energy row (median ρ 0.9834 vs `outlier_exposure`).

**2. Exact fix.** Exclude or relabel as "Energy (mean-centred logits)"; since mean-centring costs
~0.019 AUROC with no paper basis, remove it (so it becomes plain energy) if kept; collapse redundant
energy rows; add a docstring.

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\divoe.py`; reporting table.

**4. Feasibility.** D — faithful DivOE needs the same auxiliary outlier corpus as OE (to extrapolate
from and fine-tune against) plus a trainable classifier; the protocol lacks both.

**5. Effort / risk.** S. Removing mean-centring nudges the number up ~0.019 to equal plain energy; the
honest action (exclude/collapse) removes a duplicate. No faithful-DivOE number is recoverable.

---

## 20. DiverseMix — NOT-THE-METHOD (fabricated outliers) — Class D — Effort S (relabel)

**1. Divergences.** DiverseMix diversifies a **real** auxiliary outlier corpus (official: ImageNet64x64).
This implementation has no aux data — it fabricates pseudo-outliers by convex-combining ID features
from two classes (`_generate_auxiliary_outliers`, `diversemix.py:172-219`, mix `:211-212`). 100% of
these fall inside the ID 95th-percentile shell (median Mahalanobis 4.229 vs genuine ID 5.637), so the
`relu(logsumexp+1)` auxiliary loss (`:155`) trains the head to give **low** energy to the ID interior —
opposite of intent. The base returns `+logsumexp` directly (`:306`), contradicting both the official
convention and its own training objective. Only the head trains (backbone frozen, `:100-102,110`) vs
official end-to-end. The `:298-305` comment rationalises the inversion by "≤82 samples"; the real cause
is the fabricated auxiliary set.

**2. Exact fix.** Report as a **negative result** / exclude from the headline table (0.5200 over 40,
0.5018 TSB-U — at chance either orientation). If kept, adopt `−logsumexp` (the `diversemix_enh` choice)
as canonical; replace the false comment `:298-305`; fix `validation_status.json` (FAIL with zero
discrepancies).

**3. Files to edit.** `C:\THESIS\benchmark1\models\ood_methods\diversemix.py` (orientation 306, comment
298-305); `C:\THESIS\methods\diversemix\validation_status.json`. (Existing one-line variant:
`C:\THESIS\methods\diversemix\diversemix_enh\diversemix_enh.py:34` = `-energies`.)

**4. Feasibility.** D — faithful DiverseMix requires a real collected auxiliary outlier corpus (absent)
plus end-to-end training (backbone frozen); the pseudo-outlier fabrication cannot be repaired because
there is nothing genuinely OOD to diversify. **The orientation is NOT regime-dependent** — it is a pure
sign flip (`base + enh = 1.0000` exactly on all 4 shared datasets); one canonical orientation exists
(`−logsumexp`), and `diversemix_enh` uses it — but it does not rescue the method (0.5200 → 0.4800,
at chance both ways). The benchmark deliberately runs the base orientation (documented note), which is
the variant that failed its own validation.

**5. Effort / risk.** S (orientation flip already written). Low risk to substantive conclusions —
flipping only moves 0.5200 → 0.4800, still at chance; the honest framing is a negative result. Faithful
version is research-scale and out of scope.

---

## 21. DiffAD — NOT-THE-METHOD (denoises from pure noise) — Class A — Effort S

**1. Divergences.** The reverse process starts from **pure noise, independent of the input**
(`diffad.py:253` `x_t = torch.randn_like(feat_i)  # Start from noise`), runs all `n_steps` (`:255`),
and measures MSE between that unconditional draw and the input (`:277`) — so the "reconstruction" is a
random ID sample, and the score analytically collapses to an inverted distance-to-generator-mean
(Spearman 0.971 vs −distance-to-mean). The official test path feeds the input to a conditional
generator (imputation by partial noising + denoising). Orientation is **negated** on a false premise
(`:288,282-287`). Real data: mean AUROC 0.2871 (flipped 0.7129; all 5 extreme-magnitude datasets score
exactly 0.000).

**2. Exact fix.** **Wire `diffad_fix` into the runner and report it** (or drop DiffAD). Confirmed
contents of `C:\THESIS\methods\diffad\diffad_fix\`: `diffad_fix.py`, `CHANGES.md`,
`validation_status.json`. `diffad_fix.py` subclasses `DiffADDetector`, inherits `fit()`, and reimplements
`score()` (~35 lines) as genuine **input-conditioned partial noising**: noise the input feature `x0` to
step `t0` (`max(1, n_steps//2)`, `:29`) via
`xt = sqrt_alphas_cumprod[t0]*x0 + sqrt_one_minus_alphas_cumprod[t0]*noise` (`:48-50`), denoise from
`t0` back to 0 conditioned on that noised input (`:52-61`), and return the **non-negated**
reconstruction error (`:65`). This is the faithful imputation-style procedure. **Open item:** the exact
author `t0` was never read from the official `networks.define_G()`, so `t0=n_steps//2` is a
reasonable-but-unverified default — verify before claiming full faithfulness. Also fix the base
docstring (`diffad.py:9-11`) and delete the false rationale (`:282-287`).

**3. Files to edit.** `C:\THESIS\methods\diffad\diffad_fix\diffad_fix.py` (verify `t0`);
`C:\THESIS\experiments\run_experiments.py` and `C:\THESIS\experiments\tsb_benchmark.py`
(register/wire `DiffADFixDetector` — currently unwired, referenced only in `methods/_validation/*.py`
and `results/evaluate_ablation.py`, with 0 result dirs vs 40 for base `diffad`);
`C:\THESIS\benchmark1\models\ood_methods\diffad.py` (docstring/rationale).

**4. Feasibility.** A — unlike OE/DivOE/DiverseMix, DiffAD needs no aux outliers and no backbone
retraining; it trains its own small DDPM on frozen features in `fit()`. The faithful score() already
exists — the only work is wiring + verifying `t0`.

**5. Effort / risk.** S (score() written; ~35 lines). CPU-feasible (N × recon_samples × n_steps tiny
forward passes). **Risk to headline: high magnitude but corrective** — the current 0.2871 row is a
sign-inverted distance-to-mean artefact; the faithful variant will materially change DiffAD's number
(a genuine fidelity improvement, not a cosmetic flip). Verify `t0` first.

---

## Cross-cutting findings

1. **Redundant energy rows.** `energy_ebo`, `outlier_exposure`, and `divoe` are the same energy
   detector reported up to three times (OE has 0 result dirs of its own; DivOE's mean-centring is
   inert). Collapse to one "Energy (EBO)" row.

2. **Top-2 leaderboard duplication.** `driftlens` (#2, PCA-Mahalanobis) duplicates `mahalanobis` (#1)
   at median ρ 0.999. Disclose; consider reporting one plus an explicit ablation. The mahalanobis fix
   (§4) and this relabel should be resolved together.

3. **Orientation-inverted detectors.** `catsight`, `codit`, `ae_adwin_lstm`, and base `diffad` all ship
   an inverted score (roughly `AUROC → 1 − AUROC`). Their currently-reported sub-chance numbers should
   not be presented as the methods' performance — flip (catsight, codit) or wire the corrected variant
   (diffad) or relabel (ae_adwin_lstm) before reporting.

4. **Unwired corrected variants that already exist.** `deedee_fix` (faithful, PASS) and `diffad_fix`
   (faithful input-conditioned) are written but never registered in the runner. `diversemix_enh` exists
   as a one-line sign flip but does not rescue the method.

5. **Two "headline strongest" claims rest on confounds.** `tdivdm` (0.838 over 40) has 16/40 datasets
   at exactly 1.000 on extreme-magnitude cases; `srs` (0.841 on TSB-U) is the signal-only variant, not
   the seasonal-ratio method. Both need reframing.

6. **Open item (unresolved across all verifications).** The 18-vs-21 univariate dataset-count
   discrepancy is likely caused by 3 silently-missing output dirs (`TSB-U-STABLE_062/070/080`, small
   YAHOO series) per the diversify verification — worth confirming before the final results tally.
