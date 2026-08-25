# Energy (EBO) Faithfulness Verification — FAITHFUL

**Method id:** `energy_ebo` · **Paper:** Liu, Wang, Owens & Li, *Energy-based Out-of-distribution
Detection*, NeurIPS 2020 (arXiv:2010.03759)
**Implementation under test:** `benchmark1/models/ood_methods/base_ood.py:90-91` (`BaseOODDetector._energy`)
**Benchmark path for EBO:** `outlier_exposure` with `score_type="energy"`
**Verified:** 2026-08-19

---

## Verdict

**FAITHFUL.** The shared `_energy` helper computes `−logsumexp(logits/T)`, which is the official
energy score `E(x) = −T·logsumexp(logits/T)` divided by the positive constant `T`. At the default
`T = 1` the two are **exactly identical**; for any other `T > 0` they differ only by a positive
multiplicative factor, which preserves ranking and therefore leaves AUROC, AUPR and FPR@95 unchanged.
Orientation matches (higher = OOD). No corrective variant is required.

---

## 1. Source accessibility — one source missing, stated not guessed

| Source | Status |
|---|---|
| `methods/energy_ebo/reference/` | **ABSENT.** `methods/energy_ebo/` contains only `EBO_VALIDATION.md`. No `reference/` directory exists, and no clone of `wetliu/energy_ood` was found anywhere under `methods/`. |
| `https://raw.githubusercontent.com/wetliu/energy_ood/master/CIFAR/test.py` | Fetched live |
| `https://arxiv.org/abs/2010.03759` (abstract) | Fetched live |
| `https://ar5iv.labs.arxiv.org/html/2010.03759` (full text) | Fetched live |
| `methods/divoe/reference/src/test.py:126` | **Local corroboration** — DivOE's codebase is derived from `energy_ood` and contains a byte-identical copy of the official energy expression |

Because the dedicated reference clone is missing, the comparison rests on the live fetch of the
official repository plus the independent local copy of the same expression in the DivOE reference. The
two agree exactly. Nothing below is inferred from an unavailable source.

**From the official code** (`wetliu/energy_ood`, `CIFAR/test.py`):

```python
# energy branch
_score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
# baseline (MSP) branch
_score.append(-np.max(smax, axis=1))
```

with `parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')` and
`parser.add_argument('--score', default='MSP', type=str, help='score options: MSP|energy')`.

**From the paper (full text):**

- Free energy: `E(x;f) = −T·log Σᵢᴷ exp(fᵢ(x)/T)` — "the negative sign is integral to the formulation".
- Orientation: higher energy indicates out-of-distribution; the decision rule is stated as
  `−E(x;f) > τ`, i.e. the paper negates for the convention that in-distribution samples score higher.
- Temperature: `T = 1` in experiments; the ablation reports that *"using larger T leads to more
  uniformly distributed predictions and makes the energy scores less distinguishable"*, supporting the
  parameter-free setting.
- Post-hoc: no retraining — a *"parameter-free measure"* applicable to any pre-trained classifier.

## 2. Divergence table

| Component | Original | Mine (`file:line`) | Changes results? |
|---|---|---|---|
| Core statistic | `logsumexp(logits/T)` (`CIFAR/test.py`, energy branch) | `torch.logsumexp(logits / temperature, dim=-1)` (`base_ood.py:91`) | **No** |
| Outer temperature factor | `× T` present | **Omitted** (`base_ood.py:91`) | **No** — positive scaling; identical at T=1 |
| Sign / orientation | `−T·logsumexp` = `E`, higher = OOD | `−logsumexp`, higher = OOD (`base_ood.py:91`) | **No** |
| Default temperature | `T = 1.` (argparse default) | `temperature = 1.0` (`base_ood.py:90`; caller default `outlier_exposure.py:16`) | **No** |
| Features used | Classifier logits | Classifier logits via `_forward_logits` (`base_ood.py:98`, head chained at `:48-50`) | **No** |
| Training | Post-hoc, parameter-free | `fit()` is a no-op (`base_ood.py:79-80`) | **No** |
| Dropped/added steps | none | none on the EBO path (`outlier_exposure.py:24`) | **No** |

### Equivalence

Let `s = −logsumexp(logits/T)` (mine) and `E = −T·logsumexp(logits/T)` (official). Then `E = T·s`.

| Case | Relation | Metric impact |
|---|---|---|
| `T = 1` (default) | `E = s` — bitwise identical | none |
| `T > 0`, `T ≠ 1` | `E = T·s`, strictly increasing in `s` | none for AUROC / AUPR / rank-based FPR@95 |

**One honest caveat.** The invariance of FPR@95 holds because the threshold is derived from the score
distribution (the 95%-TPR quantile). A *fixed absolute* threshold on the energy value would be
sensitive to the missing `T` factor. This is moot at the project default `T = 1`, where the two
expressions are identical, but it is the one condition under which the omission would matter.

## 3. Structural verification from saved scores

Verified from `experiments/*/*/outlier_exposure/` (40 datasets, 2032 windows).

| Test | Expectation for `−logsumexp` | Observed | Verdict |
|---|---|---|---|
| Sign | `logsumexp > 0` for non-degenerate logits ⇒ score < 0 | **100.0%** of scores negative; global range `[−1750135.250, −1.330]` | consistent |
| Upper bound | Uniform logits ≈ 0 give `−log K = −log 4 = −1.386` | Per-dataset maxima cluster at `−1.39` to `−1.44`; global max `−1.330` | consistent, and confirms K = 4 |
| Finiteness | all finite | all finite | consistent |

**Cross-detector identity check.** Since `max softmax = exp(max_logit − logsumexp)`,

```
max_logit = log(1 − msp_score) − energy_score
```

Combining the saved MSP and energy scores recovers a per-dataset median `max_logit`. The values are
coherent (typically 2–10 for well-behaved datasets) and expose the pathological cases directly:

| Dataset | median recovered `max_logit` | energy min |
|---|---|---|
| TSB-M-STABLE_015 | ≈ 1,284,612 | −1,750,135 |
| TSB-U-DRIFT_060 | ≈ 161,002 | −438,816 |
| TSB-M-STABLE_020 | ≈ 109,251 | −236,824 |
| TSB-M-OOD_008 | ≈ 72 | −87.96 |
| TSB-U-DRIFT_051 | ≈ 5.8 | −11.67 |

This is a three-way consistency result: the same recovered logit magnitudes independently explain
(a) why MSP saturated to exactly 0 on those datasets, and (b) why ODIN required a logit gap ≳ 17,000
to saturate at T=1000. Two detectors computed from the same logits by different formulas agree on the
underlying logit scale — which would not happen if `_energy` were computing anything other than
`−logsumexp`.

## 4. Where `_energy` is consumed (full audit)

| Consumer | Call site | What it feeds to `_energy` | Is it plain EBO? |
|---|---|---|---|
| `outlier_exposure` (`score_type="energy"`) | `outlier_exposure.py:24` | Raw classifier logits | **Yes — this is the EBO baseline** |
| `scale_enh` | `methods/scale/scale_enh/scale_enh.py:50` | Logits from scaled penultimate activations | Energy used faithfully; the scaling is SCALE's own step |
| `divoe` | `divoe.py:28` | Logits **minus a fixed mean vector** (`divoe.py:27`) | No — mean-centring by a per-class *vector* is not a scalar shift, so it changes `logsumexp` ranking. DivOE's own concern |
| `scale` (base, known-wrong) | `scale.py:31` | z-standardised logits | No — pre-transform is SCALE's documented defect |
| `diversemix` | `diversemix.py:58` | — returns `+logsumexp` from its own trained head, **not** via `_energy` | No — opposite orientation by design |

Only the `outlier_exposure.py:24` path is unmodified EBO, and that is the path the benchmark reports
as the energy result — consistent with `EBO_VALIDATION.md` §0.

**Minor code-hygiene note (not a fidelity issue).** `methods/scale/scale_enh/scale_enh.py:48` and
`methods/dice/dice_enh/dice_enh.py:37` inline-duplicate the energy expression
(`-torch.logsumexp(feats / T, dim=-1)`) on *features* instead of calling `_energy`, as a fallback for
when no classifier head is configured. These paths are unreachable when a head is present, but the
duplicated formula could drift from the helper if the helper is ever changed.

## 5. Measured results

| Detector / subset | n | mean AUROC |
|---|---|---|
| EBO (`outlier_exposure` energy) — all | 40 | 0.2949 |
| EBO — TSB-U (univariate) | 21 | 0.2770 |
| MSP — TSB-U (comparison) | 21 | 0.3417 |
| `divoe` — all | 40 | 0.2763 |
| `divoe` — TSB-U | 21 | 0.2299 |

EBO is below chance on **15 of 21** univariate datasets and inverts *more strongly* than MSP
(0.2770 vs 0.3417). The mechanism is confirmed by §3: energy reads the unnormalised logit magnitude
rather than the normalised maximum probability, so it is more exposed to the inflated logits the
backbone emits on far-off-manifold windows — in the extreme cases above, logits of order 10⁵–10⁶.

## 6. Open items

**Missing reference clone.** `methods/energy_ebo/reference/` does not exist. Verification relied on
the live repository plus the corroborating local copy in `methods/divoe/reference/src/test.py:126`.
Re-cloning `https://github.com/wetliu/energy_ood` would restore parity with the other method folders.

**Dataset-count discrepancy (unresolved, same as MSP/ODIN).** `EBO_VALIDATION.md:60-65` cites mean
AUROC **0.257** with **14/18** below chance and `divoe` at **0.204**; the artifacts give **0.2770**
over **21** datasets, **15/21** below chance, and `divoe` at **0.2299**. The qualitative conclusions
hold, but the counts and means do not reconcile. No config defining an 18-dataset subset was found.
Same open question recorded in `methods/msp/VERIFICATION.md` §5 and `methods/odin/VERIFICATION.md` §7.

## 7. Conclusion

The energy score is a faithful reproduction of EBO. The core statistic, orientation, default
temperature, feature source and post-hoc nature all match the official implementation and the paper's
Eq. for `E(x;f)`. The single divergence — the omitted outer `T` factor — is exactly zero-impact at the
default `T = 1` and a rank-preserving positive rescaling otherwise. Structural checks on the saved
scores confirm the implemented formula is `−logsumexp` with K = 4, and a cross-detector identity shows
the energy and MSP paths agree on the underlying logit scale.
