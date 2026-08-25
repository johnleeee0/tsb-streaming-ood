# MSP Faithfulness Verification — FAITHFUL

**Method id:** `msp` · **Paper:** Hendrycks & Gimpel, *A Baseline for Detecting Misclassified and
Out-of-Distribution Examples in Neural Networks*, ICLR 2017 (arXiv:1610.02136)
**Implementation under test:** `benchmark1/models/ood_methods/msp.py` (`MSPDetector`)
**Verified:** 2026-08-19

---

## Verdict

**FAITHFUL.** The implementation computes the maximum softmax probability of the classifier logits
at T=1 and reports `1 − max softmax` so that higher means more out-of-distribution. Both deviations
from the original — the sign convention and the exposed temperature parameter — are
metric-invariant at defaults. No corrective `_enh` variant is required.

---

## 1. Source accessibility

All sources were reachable. Nothing in this report is guessed.

| Source | Status |
|---|---|
| `https://raw.githubusercontent.com/hendrycks/error-detection/master/Vision/CIFAR_Detection.py` | Fetched live; matches the local clone verbatim |
| `https://arxiv.org/abs/1610.02136` | Fetched live (abstract) |
| Local clone `methods/msp/reference/` | `origin = https://github.com/hendrycks/error-detection`, commit `276d605bfa9a9bd7701bd88937c537c3fcab94cf` (2018-12-26) |

Live fetch confirms the three decisive facts:

- Final layer is a softmax: `network = DenseLayer(avg_pool, num_units=nout, W=HeNormal(), nonlinearity=softmax)` (`Vision/CIFAR_Detection.py:157`)
- Score is the max softmax probability: `right_wrong_fn = theano.function([input_var, target_var], [right, kl, T.max(test_prediction, axis=1)])` (`:252`)
- "Neither temperature scaling nor input preprocessing/perturbation mechanisms appear in this file."

The paper's abstract states the orientation: *"Correctly classified examples tend to have greater
maximum softmax probabilities than erroneously classified and out-of-distribution examples."*
Higher = in-distribution.

## 2. Divergence table

| Component | Original (`reference/Vision/CIFAR_Detection.py`) | Mine (`file:line`) | Changes results? |
|---|---|---|---|
| Features used | Softmax output of the classifier head | `base_ood.py:98` → `_forward_logits`, chains classifier head at `base_ood.py:48-50` | **No** |
| Score formula | `max_k softmax_k` (`:252`) | `1.0 - self._softmax_max(logits, temperature=self.temperature)` (`msp.py:19`); `softmax(logits/T).max(dim=-1)` (`base_ood.py:86-88`) | **No** — additive/monotone transform |
| Orientation | Higher = ID; score passed raw with `labels[:len(in_sample)] += 1` (`:340-345`) | Higher = OOD (`1 − c`) | **No** — all metrics are rank-based |
| Temperature | Absent (implicit T=1) | Exposed, default `1.0` (`msp.py:15`) | **No** at the default |
| Input perturbation | Absent | Absent | **No** |
| Training | Post-hoc only | `fit()` is a no-op (`base_ood.py:79-80`) | **No** |
| Framework | Theano / Lasagne | PyTorch | **No** |

Let `c = max_k softmax(logits)_k` be the official confidence. Then `1 − c = 1 + (−c)`, an additive
constant on the authors' own later PyTorch form (`hendrycks/outlier-exposure`, `CIFAR/test.py`:
`_score.append(-np.max(smax, axis=1))`). AUROC, AUPR and FPR@95 depend only on the ranking of
scores, so all three are numerically identical to the official implementation.

## 3. AUROC < 0.5 is genuine softmax overconfidence, not an orientation bug

Verified directly from the saved arrays in `experiments/*/*/msp/` (40 datasets, 2032 windows), not
from the prior notes.

**Structural proof of the formula.** The observed score range across all datasets is exactly
`[0.000000, 0.742717]`. For K classes, `1 − max softmax` is bounded to `[0, 1 − 1/K]`; here
`0.742717 ≤ 0.75 = 1 − 1/4` (K = 4 pseudo-classes), and the minimum is exactly 0.

| Candidate formula | Predicted range | Consistent with observation? |
|---|---|---|
| `1 − max softmax` (K=4) | `[0, 0.75]` | **Yes** |
| `−max softmax` | `[−1, −0.25]` | No (would be negative) |
| `max softmax` (no flip) | `[0.25, 1]` | No (min would be ≥ 0.25) |

Only `1 − max softmax` produces the observed signature.

**Mechanism confirmed.** Recovering the official confidence `c = 1 − score` and comparing
in-distribution against out-of-distribution windows, OOD windows are *more* confident in **25 of 40**
datasets overall and **16 of 21** univariate datasets.

| Dataset | AUROC | mean conf (ID) | mean conf (OOD) |
|---|---|---|---|
| TSB-U-DRIFT_051 | 0.001 | 0.671 | 0.997 |
| TSB-U-DRIFT_060 | 0.000 | 0.952 | 1.000 |
| TSB-U-OOD_046 | 0.000 | 0.796 | 1.000 |
| TSB-U-OOD_039 | 0.092 | 0.810 | 0.974 |
| TSB-U-OOD_043 | 0.654 | 0.694 | 0.642 |
| TSB-U-DRIFT_034 | 0.980 | 0.869 | 0.540 |

**Flip test.** Mean AUROC is 0.385; flipping the orientation would give 0.615. A sign error would
have to make DRIFT_051 *better* (~0.66), not 0.001. Where the network is appropriately less
confident on OOD (DRIFT_034: 0.869 vs 0.540) MSP scores 0.980. The observed pattern is only
consistent with correct orientation plus genuine data-driven overconfidence — deep ReLU networks are
free to be arbitrarily confident far from the training manifold (Nguyen et al. 2015; Hein et al.
2019), and under global normalisation the second-source windows lie there.

## 4. Measured results

| Subset | n | mean AUROC | OOD more confident than ID |
|---|---|---|---|
| All datasets | 40 | 0.3850 | 25/40 |
| TSB-U (univariate) | 21 | 0.3417 | 16/21 |
| TSB-M (multivariate) | 19 | 0.4328 | 9/19 |

## 5. Open items and caveats

**Dataset-count discrepancy (unresolved).** `MSP_VALIDATION.md` cites 18 univariate datasets,
"14 of 18" inverted, and mean 0.343; the artifacts on disk show **21**, **16 of 21**, and **0.3417**.
The means agree to ~3 decimals and the conclusion is unchanged (slightly stronger), but the counts do
not reconcile, and §9's thesis-ready paragraph quotes "fourteen of eighteen" verbatim. Similarly the
notes cite "0.386 over all 36 datasets" where the artifacts give **0.385 over 40**. No config
defining an 18-dataset target subset was found. **This needs a decision on which subset is canonical
before the thesis text is finalised.**

**Degenerate datasets (undisclosed in prior notes).** Five datasets have exactly constant scores
(`ptp = 0.0`), giving AUROC 0.500 by tie-breaking: `TSB-M-OOD_008`, `TSB-M-OOD_024`,
`TSB-M-OOD_031`, `TSB-M-OOD_072`, `TSB-M-STABLE_015`. The softmax is fully saturated (confidence
1.0 to float precision). This is not an implementation fault, but these contribute uninformative
0.5s that anchor the mean and should be disclosed.

## 6. Conclusion

The implementation is a faithful, minimal reproduction of the maximum-softmax-probability baseline.
The score is `1 − max softmax(logits/T)` with T=1, equal to the official max-softmax score up to a
metric-invariant transform, so AUROC is identical. The below-chance AUROC on streaming data is the
genuine softmax-overconfidence inversion, confirmed mechanistically from the saved scores, not an
orientation bug.