# DiffAD Faithfulness Verification — FAITHFUL (production uses the faithful diffad_fix variant 2026-08-21; see FIX APPLIED)

## FIX APPLIED (2026-08-21)
The base `diffad` ran the reverse process from PURE NOISE (input-independent) and negated the score —
NOT-THE-METHOD. The runner now wires `DiffADFixDetector` (`methods/diffad/diffad_fix/diffad_fix.py`), which
partially noises the input to an intermediate step t0 and denoises back (imputation-style, input-conditioned)
and reports the non-negated reconstruction error, faithful to the published procedure. Verified end-to-end
in the integration smoke test.


**Method id:** `diffad` (+ `diffad_fix`) · **Paper:** Xiao et al., *Imputation-based Time-Series Anomaly
Detection with Conditional Weight-Incremental Diffusion Models*, KDD 2023,
doi:10.1145/3580305.3599391
**Official code:** `https://github.com/ChunjingXiao/DiffAD`
**Implementation:** `benchmark1/models/ood_methods/diffad.py` (`DiffADDetector`, `n_steps=20`,
`recon_samples=2`)
**Corrected variant:** `methods/diffad/diffad_fix/diffad_fix.py` (`DiffADFixDetector`)
**Verified:** 2026-08-20

---

## Verdict

**NOT-THE-METHOD.** The reverse process starts from **pure noise, independent of the input**
(`diffad.py:253`: `x_t = torch.randn_like(feat_i)  # Start from noise`), runs all `n_steps` denoising
steps, and then measures MSE between that unconditionally generated sample and the input (`:277`). The
input never enters the generation, so the "reconstruction" is a random draw from the learned ID
distribution.

**Mathematically the detector collapses to an inverted distance-to-mean score.** Because the generated
sample `g` is independent of the input `x`,
`E‖x − g‖² = ‖x − μ_g‖² + Var(g)` — so the score is `−(squared distance to the generator's mean) − const`.
Verified: `spearman(score, −distance_to_generator_mean) = 0.97077`, and on a controlled setup where OOD
lies far from the ID mean the implemented orientation gives **AUROC 0.0000** against **1.0000**
un-negated (§3B).

**That is exactly what happens on real data.** Across 40 datasets the mean AUROC is **0.2871**
(TSB-U **0.2636**, 15/21 below chance); flipped it is **0.7129**. On the five
extreme-feature-magnitude datasets — where OOD windows are furthest from the ID mean — **all five score
exactly 0.000** (§3C). The negation at `:288`, justified by an asserted rationale at `:282-287`, ranks
the most distant samples as the most in-distribution.

**`diffad_fix` implements the correct procedure and has never been run.** It noises the **input** to an
intermediate step `t0 = n_steps // 2` and denoises back (`diffad_fix.py:46-61`), returning the
non-negated reconstruction error (`:65`). It has **0** result directories; `diffad` has 40. This is the
same pattern recorded for DEEDEE (`methods/deedee/VERIFICATION.md`).

**This is the third asserted negation in the audit worth roughly +0.43 AUROC**, alongside CatSight
(0.250 → 0.750) and AE-ADWIN-LSTM (0.253 → 0.747).

---

## 1. Source accessibility — the official reverse process was only partially readable

| Source | Status |
|---|---|
| `methods/diffad/reference/` | **ABSENT — directory does not exist.** The folder holds `validation_status.json` and `diffad_fix/`. |
| `github.com/ChunjingXiao/DiffAD` (README) | Fetched live. Confirms the paper and lists `time_train.py`, `time_test.py`, `model/`, `core/`, `config/`. **The README does not describe the reverse process or the anomaly score.** |
| `time_test.py` | Fetched live — but it is an orchestration layer only; it calls `diffusion.test(continous=False)` and delegates to `Model` / `Metrics`. |
| `model/model.py` | Fetched live. **Partial answer obtained**: the test path calls `self.netG.module.super_resolution(self.data['SR'], continous=continous, ...)` — i.e. it **passes the input** as conditioning — whereas the separate `sample()` path calls `self.netG.module.sample(batch_size, continous)` with no input. The exact noising schedule lives in `networks.define_G()`, which was **not fetched**. |
| Paper (KDD 2023) | Not fetched. |

**What this supports and what it does not.** The existence of a conditional entry point that receives
the input (`super_resolution(self.data['SR'], ...)`), distinct from an unconditional `sample()`, is
direct evidence that the official test path is **input-conditioned**. Combined with the paper's own
title — *"**Imputation**-based …"*, and imputation is by definition conditioned on observed values — the
conclusion that input conditioning is part of the method is well supported. **But I did not read the
reverse-loop initialisation itself**, so the precise partial-noising level `t0` used by the authors is
not established here and is not asserted. `diffad_fix`'s choice of `t0 = n_steps // 2`
(`diffad_fix.py:29`) is a reasonable default, not a verified match.

The finding that *this* implementation is not input-conditioned needs none of that: it is visible
directly at `diffad.py:253`.

## 2. Divergence table

| Component | Official (per §1) | Base `diffad.py:line` | `diffad_fix.py:line` |
|---|---|---|---|
| **Reverse-process start** | input passed to a conditional generator (`super_resolution(self.data['SR'], …)`) | **`torch.randn_like(feat_i)`** — pure noise, input-independent (`:253`) | **input noised to `t0` then denoised back** (`:48-50`, `:52`) ✓ |
| **Does the reconstruction depend on the input?** | yes | **No** | yes ✓ |
| **Score** | reconstruction error (imputation residual) | MSE to an unconditional draw (`:277`) | MSE to the input-conditioned reconstruction (`:62`) ✓ |
| **Orientation** | higher error = anomalous | **negated** (`:288`), rationale `:282-287` | **not negated** (`:65`) ✓ |
| Operand | raw multivariate series | frozen backbone features (`:163`, `:239`) | inherited (features) |
| Denoiser | conditional U-Net (weight-incremental) | MLP with time embedding (`:39-84`) | inherited |
| Noise schedule | not read | linear β, 1e-4→0.02 (`:91-93`) | inherited |
| Training | conditional diffusion | standard DDPM ε-prediction on ID features (`:192-204`) | inherited (unchanged) |
| `n_steps` / `recon_samples` | not read | 20 / 2 (config; defaults 50 / 5) | inherited |
| **Results produced** | — | **40 datasets** | **0 datasets** |

## 3. Empirical findings

**(A) Coverage.** `diffad`: **40** result dirs. `diffad_fix`: **0**.

**(B) The pure-noise reduction, demonstrated.** Replicating `:251-288` with an input-independent
generator (32-d, 400 ID + 400 OOD, OOD displaced from the ID mean):

| Quantity | Value |
|---|---|
| `spearman(score, −distance_to_generator_mean)` | **0.97077** |
| AUROC **as implemented** (negated) | **0.0000** |
| AUROC **un-negated** | **1.0000** |

Since `E‖x − g‖² = ‖x − μ_g‖² + Var(g)`, averaging over `recon_samples` draws estimates the squared
distance to the generator's mean plus a constant. The diffusion model contributes only that mean and
variance — the trained denoiser plays no discriminative role beyond locating the ID centroid.

**(C) Real-data results.**

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 40 | **0.2871** | — |
| TSB-U (univariate) | 21 | **0.2636** | **15/21** |
| **extreme-magnitude subset** | 5 | **0.0000** | 5/5 |
| All, **flipped** | 40 | **0.7129** | — |

Scores span `[−2.644e+11, −1.213]`, all ≤ 0 as a negated MSE requires. The five
extreme-magnitude datasets (`TSB-M-STABLE_015`, `STABLE_020`, `STABLE_043`, `STABLE_083`,
`TSB-U-DRIFT_060`) all return exactly 0.000 — on `STABLE_015` the score range is
`[−2.6e11, −5.2e9]`, i.e. OOD windows are eleven orders of magnitude from the generator mean and are
therefore ranked as maximally in-distribution.

The rationale at `:282-287` claims *"OOD features collapse near the backbone's uncertainty mean (low
variance)… geometrically CLOSER to OOD features"*. The data contradicts this on every dataset where the
distance signal is large.

**(D) Cost.** `N × recon_samples × n_steps` denoiser forward passes, each on a `(1, d)` tensor:
4,400 for N=110 and **11,760** for N=294. Every one of those passes is spent generating samples that do
not depend on the input being scored.

## 4. Documentation

`diffad_fix/CHANGES.md` and the module docstring (`diffad_fix.py:1-12`) diagnose the defect precisely —
*"ran the reverse diffusion from PURE NOISE, independent of the input, so the reconstruction did not
depend on the sample being scored, and the resulting error was negated"* — and state that the published
procedure is imputation-based. That analysis is correct and is corroborated here.

The base `diffad.py` docstring (`:9-11`) says *"Test samples are reconstructed through the diffusion
process, and reconstruction error (MSE) serves as the OOD score"*. That is misleading in two ways: the
sample is not reconstructed (it is regenerated from noise, independent of the input), and the score is
the **negated** error, not the error.

## 5. Recommendations

1. **Run `diffad_fix` and report that instead**, or drop DiffAD. The corrected variant already exists
   (35 lines of `score()`), implements input-conditioned partial-noising, and removes the negation.
   Running it is cheap relative to the value — the current row is an inverted distance-to-mean detector.
2. **Do not report 0.2871 as DiffAD's performance.** It is the performance of a sign-inverted,
   input-independent generator. If a number is needed before `diffad_fix` runs, note that flipping alone
   gives 0.7129 — but that is still not DiffAD.
3. **Fix the base docstring** (`:9-11`) and delete the rationale at `:282-287`, which is empirically
   false.
4. **Verify `t0`** against the official `networks.define_G()` before treating `diffad_fix` as faithful.
   Its `t0 = n_steps // 2` is a sensible default but was not matched to the paper; §1 explains why that
   could not be checked here.
5. **State the evidence limitation** in the thesis: the official reverse-loop initialisation was not
   read (README silent, `time_test.py` delegates, `model/model.py` delegates further). What is
   established is that the official test path passes the input to a conditional generator, and that this
   implementation does not.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded
across `methods/*/VERIFICATION.md`. `diffad` covers **21** univariate datasets; see
`methods/diversify/VERIFICATION.md` §4 and `methods/catsight/VERIFICATION.md` §4 for the candidate
explanation of the "18" figure.

## 6. Conclusion

DiffAD is an imputation method: the input is partially corrupted and reconstructed, so the residual
measures how well the learned ID diffusion model can explain *that specific sample*. This
implementation instead draws an unconditional sample from the learned ID distribution and measures its
distance to the input, which — because the draw is independent of the input — reduces analytically and
empirically to a negative distance-to-ID-mean score (ρ = 0.971 against that quantity). The negation then
inverts even that, giving mean AUROC 0.2871 across 40 datasets, 0.7129 flipped, and exactly 0.000 on all
five datasets where OOD features are most distant. The defining mechanism — conditioning the reverse
process on the input — is absent, not simplified. The corrected `diffad_fix` variant implements it and
has zero results; it should be run, with `t0` checked against the official implementation first.
