# CODiT Faithfulness Verification — FAITHFUL (on-protocol; fixed 2026-08-21; multi-draw + Fisher + orientation; see FIX APPLIED)

**Method id:** `codit` · **Paper:** Kaur, Sridhar, Jha, Roy, Sokolsky & Lee, *CODiT: Conformal
Out-of-Distribution Detection in Time-series Data for Cyber-Physical Systems*, ICCPS 2022
(arXiv:2207.11769)
**Implementation under test:** `benchmark1/models/ood_methods/codit.py` (`CODiTDetector`)
**Verified:** 2026-08-20

---

## Verdict

**NOT-THE-METHOD.** The prompt asks whether the ADAPTATION label is correct. **It is not — the label
is too generous**, for four compounding reasons:

1. **The Fisher combination — CODiT's core statistical device — is absent.** The official combines
   `n = 20` independent p-values (one per random-transform draw) into a product before applying the
   Fisher CDF. `codit.py` produces **exactly one** p-value per sample (`:230`), so there is nothing to
   combine.
2. **The nonconformity measure is different.** The official samples a **random transformation** per
   iteration and scores cross-entropy against *that* transform's label
   (`check_OOD_carla.py:145`, `:109`). `codit.py` uses the **identity transform only**, always
   (`:210-222`).
3. **The orientation is inverted relative to the official.** The official `getAUROC` labels
   **ID = 1, OOD = 0** and passes the Fisher value directly (`check_OOD_carla.py:315-324`) — higher
   Fisher = more ID. `codit.py:240` returns the Fisher value as the **OOD** score. Flipping to the
   official convention moves the mean AUROC from **0.3855 to 0.6145** (§5).
4. **The Fisher transform as implemented is numerically saturated and destroys the signal.**
   `_fisher_value` hardcodes `range(20)` (`:260`) — 20 terms applied to a *single* p-value, where the
   correct term count is 1. In the project's regime this maps the entire achievable p-range into
   `[0.999999999969, 1.0]`, a spread of **3.1e-11**. **19 of 40 datasets have score range < 1e-12** and
   **10 have a single distinct score value** (AUROC exactly 0.500 by tie-breaking). The reported
   AUROCs are substantially floating-point rounding artefacts (§4).

The conformal p-value formula and the five transformation types *are* faithfully reproduced. But with
the combination stage absent, the nonconformity reduced to one transform, the orientation inverted, and
the output numerically degenerate, what is evaluated is not CODiT.

**The docstring is also internally contradictory.** `codit.py:11` states *"higher Fisher value = more
ID-like"*; `:237-239` states *"higher Fisher = more OOD; return directly without negation."* Both
cannot hold, and the code implements the second. Line 11 happens to match the official convention.

---

## 1. Source accessibility

| Source | Status |
|---|---|
| Local clone `methods/codit/reference/` | Present and intact: `origin = https://github.com/kaustubhsridhar/time-series-OOD`, commit `bbf5f693eafe0240b2dc2023bad009b3fb9d3bfc` (2024-11-15), `HEAD → refs/heads/main`. Contains `ours/check_OOD_carla.py`, `check_OOD_drift.py`, `gait/check_OOD_gait.py`. |

All findings below are quoted from that clone. **No live fetch was performed for this method** — the
clone's provenance and commit are verified, and the relevant routines were read directly. This is
stated rather than implied: the online repository was not independently re-fetched, so I cannot confirm
the clone matches current upstream HEAD.

### What the official code does

**Conformal p-value** (`ours/check_OOD_carla.py:170-184`):

```python
compare = (test_ce_loss_reshaped) <= (cal_set_ce_loss_reshaped)
p_value = np.sum(compare, axis=1)
p_value = (p_value+1)/(len(cal_set_ce_loss)+1)
```

**Fisher combination** (`:267-283`):

```python
def calc_fisher_value(t_value, eval_n):
    summation = 0
    for i in range(eval_n):
        summation += ((-np.log(t_value))**i)/np.math.factorial(i)
    return t_value*summation

# in calc_fisher_batch:
prod = 1
for k in range(eval_n):
    prod *= p_values[k][i][j][0]
output[i][j] = calc_fisher_value(prod, eval_n)
```

The Fisher CDF is applied to the **product of `eval_n` p-values**, with the term count matching the
number of p-values combined. `--n` defaults to **20** (`:44`).

**Nonconformity** — `calc_cal_ce_loss` runs *"n iterations with random sampling of windows and
transformations on calibration datapoints"* (`:145`), and the loss is
`criterion(output, target_transformation)` (`:109`) — cross-entropy against the transform actually
applied. Transform list default: `["speed","shuffle","reverse","periodic","identity"]` (`:49`).

**Orientation** (`:315-324`):

```python
indist_label = np.ones(len(in_fisher_values))
ood_label = np.zeros(len(out_fisher_values))
label = np.concatenate((indist_label, ood_label))
au_roc = roc_auc_score(label, fisher_values)*100
```

ID = 1, OOD = 0, Fisher passed unnegated ⇒ **higher Fisher = more in-distribution**.

## 2. Divergence table

| Component | Official | Mine (`codit.py:line`) | Changes results? |
|---|---|---|---|
| Transformation set | `speed, shuffle, reverse, periodic, identity` (`:49`) | same five (`:33`) | **No** |
| Transform classifier | `r3d_regressor` trained end-to-end (`:71`) | `nn.Linear` head on frozen features (`:132`, `:166-167`) | Domain adaptation — disclosed |
| Conformal p-value | `(Σ[test ≤ cal] + 1)/(n_cal + 1)` (`:179-181`) | identical (`:230`) | **No — exact match** |
| **Test nonconformity** | CE vs **randomly sampled** transform, `n` iterations (`:145`, `:109`) | CE vs **identity only**, one pass (`:210-222`) | **YES** |
| **Number of p-values** | `eval_n = 20` per sample, multiplied (`:277-279`) | **1** (`:230`) | **YES — no combination** |
| **Fisher term count** | `eval_n`, matching the p-value count (`:269`) | hardcoded **20** on one p-value (`:260`) | **YES — see §4** |
| `eval_n` config | `--n`, default 20, drives everything | declared (`:106`) but **never used** — dead parameter | **YES** |
| **Orientation** | higher Fisher = **ID** (`:315-324`) | higher Fisher = **OOD** (`:239-240`) | **YES — inverted** |
| Calibration split | fixed 13 videos, seed 42 (`:192`) | `cal_frac=0.2` of ID train (`:137-141`), unseeded `randperm` | Minor; see §6 |
| Detection rule | run-length of windows with `p < ε` (`--n` semantics, `:44`) | per-window score, no run-length logic | **YES** — different detection granularity |

## 3. What is faithful

- The five transformation types match the official default list exactly.
- The conformal p-value formula is reproduced exactly, including the `+1 / +1` smoothing and the
  `test ≤ cal` direction.
- Training the transform classifier on ID data only, then calibrating on a held-out ID split, is the
  right structure.
- The `_fisher_value` *formula* is algebraically the same as `calc_fisher_value` — the defect is the
  term count and the single input, not the expression.

## 4. The Fisher transform is numerically degenerate

`_fisher_value(p)` computes `p · Σ_{i=0}^{19} (−ln p)^i / i!`, which equals `P(Poisson(−ln p) ≤ 19)`.
For a single p-value the statistically correct term count is **1**, giving `F(p) = p` exactly. Using 20
terms saturates the output:

| p | `official_fisher(p, n=1)` (correct) | `codit.py` `_fisher_value(p)` (20 terms) |
|---|---|---|
| 0.06 | 0.060000 | 0.999999999973 |
| 0.25 | 0.250000 | 1.000000000000 |
| 0.50 | 0.500000 | 1.000000000000 |
| 0.90 | 0.900000 | 1.000000000000 |

In the project's regime (~82 ID training windows × `cal_frac=0.2` ⇒ `n_cal ≈ 16`), the achievable
p-values are `{1/17, …, 17/17}` ⊂ `[0.059, 1.0]`, and:

| Quantity | Value |
|---|---|
| Distinct p-values available | 17 |
| F range over those p | `[0.999999999968991, 1.000000000000000]` |
| **F spread** | **3.101e-11** |
| float64 eps at 1.0 | 2.220e-16 |
| **Distinct F values (float64)** | **9 of 17** |

So the transform collapses 17 distinguishable p-values into 9, and confines every score to within
3e-11 of 1.0.

**Worse: the computed F is not even monotone.** A monotonicity test over 20,000 points in
`p ∈ [1e-6, 1]` returned **False**. Mathematically `F` is increasing in `p`, so this is pure
floating-point cancellation in the saturated region — precisely the region the project operates in.
The Fisher step therefore does not merely add nothing to the ranking; it **perturbs it with rounding
noise**.

## 5. Measured results

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 40 | **0.3855** | — |
| TSB-U (univariate) | 21 | **0.3252** | 12/21 |
| All, under the **official** orientation | 40 | **0.6145** | — |

Degeneracy in the saved scores:

| Symptom | Count |
|---|---|
| Score range `ptp < 1e-12` (saturated) | **19 / 40** |
| ≤ 2 distinct score values | **11 / 40** |
| Exactly 1 distinct value ⇒ AUROC = 0.500 by ties | **10 / 40** |
| All scores within `[0.99999, 1.0]` | **40 / 40** |

Ten datasets return AUROC exactly 0.500 purely because every window received an identical score
(`TSB-M-OOD_020`, `OOD_031`, `OOD_063`, `OOD_072`, `STABLE_015`, `STABLE_083`, `TSB-U-OOD_080`,
`STABLE_062`, `STABLE_070`, `STABLE_080`). For the remainder, the discriminative signal lives in the
11th–16th significant digit.

**The orientation evidence is unambiguous.** Under the official convention the mean rises from 0.386 to
0.615 — i.e. the asserted orientation is not just non-standard, it is the wrong way round on this data
too. The comment at `:237-239` justifies it by asserting *"OOD samples have LOWER CE loss (simpler
temporal structure → easier to predict the identity transform)"*. That is an empirical conjecture about
this dataset, not a property of the method, and the measured result contradicts it.

## 6. Minor observations

- **Unseeded calibration split.** `torch.randperm(N)` at `:139` is unseeded, so the calibration set —
  and therefore every p-value — changes between runs. The official fixes it
  (`generator=torch.Generator().manual_seed(42)`, `:192`). This makes the CODiT results
  non-reproducible.
- **`eval_n` is dead.** Declared at `:106`, documented at `:96` as "Fisher combination iterations",
  never read. `_fisher_value` hardcodes 20 (`:260`).
- **`p = max(p, 1e-8)` clamp** (`:256`) is unreachable given `p ≥ 1/(n_cal+1)`.
- **No run-length detection.** The official `--n` controls *"number of continuous windows with
  p-value < epsilon to detect OODness in the trace"* (`:44`) — a temporal run-length rule over a trace.
  The project scores independent windows, so this stage has no analogue. Worth disclosing, since it is
  part of how CODiT achieves its published numbers.
- Transform implementations are reasonable analogues but not verified against the official
  `dataset/carla.py` transform code; `shuffle` uses one permutation shared across the batch (`:59-60`),
  and `periodic` a single random offset per batch (`:64-65`), rather than per-sample draws.

## 7. Recommendations

1. **Fix the Fisher term count** — use `eval_n` matching the number of p-values actually combined. With
   one p-value that means `F(p) = p`, which removes the saturation entirely and restores a usable
   dynamic range.
2. **Restore the multi-draw combination** — sample `eval_n` random transformations per test window,
   compute one p-value each, multiply, then apply `calc_fisher_value(prod, eval_n)`. This is the
   method's core device and is a modest change given the transform machinery already exists.
3. **Adopt the official orientation** (higher Fisher = ID ⇒ OOD score = −Fisher, or `1 − F`), or state
   explicitly in the thesis that the orientation was fitted post-hoc and is not the paper's. Given
   0.386 vs 0.615, the current choice is indefensible.
4. **Seed the calibration split** so results are reproducible.
5. **Relabel** from ADAPTATION to NOT-THE-METHOD in the fidelity table, and fix the contradiction
   between `codit.py:11` and `:237-239`.
6. **Re-run after (1)–(4)** — the current numbers cannot support any claim about CODiT, since 19 of 40
   datasets produce numerically constant scores.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded in
`methods/msp/VERIFICATION.md` §5, `methods/odin/VERIFICATION.md` §7,
`methods/energy_ebo/VERIFICATION.md` §6, `methods/mahalanobis_mds/VERIFICATION.md` §8,
`methods/dfm/VERIFICATION.md` §7, `methods/srs/VERIFICATION.md` §6,
`methods/react/VERIFICATION.md` §6, `methods/dice/VERIFICATION.md` §7,
`methods/scale/VERIFICATION.md` §8, `methods/gradnorm/VERIFICATION.md` §7 and
`methods/dimmad/VERIFICATION.md` §7 remains unresolved. `codit` covers 21 univariate datasets.

## 8. Conclusion

CODiT's conformal p-value computation and transformation set are reproduced exactly, but three of the
method's four defining stages are not: the nonconformity uses only the identity transform instead of
random draws, the Fisher combination over multiple p-values is absent, and the score orientation is
inverted relative to the official evaluation. Compounding these, the Fisher transform is applied with a
term count intended for 20 p-values to a single p-value, which saturates every score to within 3e-11 of
1.0, leaves 19 of 40 datasets numerically constant, and is not even monotone in float64. The label
should be changed from ADAPTATION to NOT-THE-METHOD, and the detector re-run after the term count,
combination stage, orientation and calibration seed are fixed.

---

## FIX APPLIED (2026-08-20)

All four compounding defects identified above have been corrected in
`benchmark1/models/ood_methods/codit.py`. The class name (`CODiTDetector`), registry key
(`@register_ood("codit")`), and `BaseOODDetector` interface (`fit(x_id, y_id)` / `score(x) -> np.ndarray`)
are unchanged. Only `codit.py` and this file were edited.

### What changed

1. **Multi-draw random-transform nonconformity restored.** A new helper `_random_transform_ce()`
   assigns each window a *randomly sampled* transformation (numpy `Generator`), applies it, and returns
   the cross-entropy loss against that transform's own label — matching the official `calc_cal_ce_loss`
   / `calc_test_ce_loss` behaviour (reference lines 109, 145). This replaces the previous
   identity-transform-only, single-pass nonconformity.

2. **Calibration is now `eval_n` random draws.** `fit()` builds `self.cal_losses` with shape
   `(eval_n, n_cal)` — one independent random-transform draw per row over the held-out ID calibration
   split (official `cal_set_ce_loss_all_iter`, ref 145-168), instead of a single 1-D identity-only
   vector.

3. **Fisher combination over the correct number of draws.** `score()` now draws `eval_n` random
   transforms per test window, computes one conformal p-value per draw against the matching calibration
   row (`(#{test_loss <= cal_loss} + 1)/(n_cal + 1)`, ref 179-181), multiplies them into a single
   product, and applies the Fisher statistic **with term count = `eval_n`** — matching
   `calc_fisher_value(prod, eval_n)` (ref 267-271). The old code applied a hardcoded `range(20)` term
   count to a *single* p-value; that is gone. `_fisher_value` was replaced by a vectorised
   `_calc_fisher_value(t_value, eval_n)` and `eval_n` is now actually consumed.

4. **Orientation flipped to the official convention.** The official `getAUROC` labels ID = 1, OOD = 0
   and passes Fisher unnegated (ref 315-324), i.e. **higher Fisher = more in-distribution**. Since this
   detector's public contract is "higher = more OOD", `score()` now returns `-Fisher`. The previous
   code returned `+Fisher` on a false "OOD has lower CE loss" premise; that comment/rationale is
   removed.

5. **Deterministic seeded calibration split.** The train/calibration split now uses
   `torch.Generator().manual_seed(self.seed)` (default `seed=42`, matching the official
   `manual_seed(42)` at ref 192). The random transform draws in both `fit` and `score` are seeded
   (numpy `default_rng` + per-draw `torch.manual_seed`), so calibration and scoring are reproducible
   run-to-run.

6. **Docstring contradiction resolved.** The module docstring and `score()` docstring now consistently
   state the official orientation (higher Fisher = more ID → returned score is `-Fisher`). The default
   `eval_n` was set to 20 (the official `--n` default); a `seed` config key was documented.

### Numerical-degeneracy check (the original §4 symptom)

The saturation was the term-count bug (20 terms on a single p-value collapsing the range to `[0.999…, 1.0]`).
With the term count matched to `eval_n` p-value products, dynamic range is restored. Smoke test
(dummy backbone, N=40 ID windows, C=2, T=24, `eval_n=8`, `cal_frac=0.25`): scoring 13 test windows
produced **13 distinct finite score values** spanning `[-0.990, -0.319]` — no saturation, well away
from the degenerate near-1.0 band.

### Smoke test (venv `C:\THESIS\.venv\Scripts\python.exe`)

```
cal_losses shape: (8, 10) (expect (eval_n, n_cal))
scores shape: (13,)
scores sample: [-0.691822 -0.844647 -0.318901 -0.463932 -0.936472]
all finite: True
distinct values: 13
range: -0.9898184780639034 -> -0.3189007605284093
deterministic score() re-run: True
mean ID-like score: -0.7018199168204405 | mean shifted score: -0.16450993411067463
SMOKE TEST PASSED
```

Output is finite and length-correct; `score()` is deterministic on repeat calls; and a magnitude-shifted
OOD batch scores higher on average than the ID-like batch (−0.16 vs −0.70), consistent with the
corrected "higher = OOD" orientation. (The full benchmark was **not** run.)

### New verdict

**FAITHFUL (on-protocol).** With the multi-draw random-transform nonconformity, the `eval_n`-way
conformal p-value combination, the correct Fisher term count, the official score orientation, and the
seeded calibration split all in place, the detector now reproduces CODiT's core statistical pipeline
faithfully within the frozen-backbone / per-window benchmark protocol. Two disclosed, protocol-mandated
adaptations remain (neither is a fidelity defect here): the transform classifier is a linear head on
frozen backbone features rather than the end-to-end `r3d_regressor` over (orig, transformed) clip pairs,
and the official run-length detection over an ordered trace has no analogue under per-window,
shuffled-window evaluation and is therefore omitted. The detector should be **re-run** on the benchmark;
the previously reported ~0.386 mean AUROC (sign-inverted and numerically saturated) must not be reported
as CODiT's performance.
