# SCALE Faithfulness Verification — CORRECTED (`scale_enh` faithful; percentile→85 fixed 2026-08-21; see FIX APPLIED)

## FIX APPLIED (2026-08-21)
The `scale_enh` default `percentile` was changed 65→85 (in `methods/scale/scale_enh/scale_enh.py`) and the
runner param updated to match, aligning the top-k pruning fraction with the official kai422/SCALE default.
`scale_enh` was already faithful in mechanism (activation scaling + energy); this closes the one remaining
hyperparameter deviation. Re-run needed to regenerate SCALE numbers.


**Method id:** `scale` / `scale_enh` · **Paper:** Xu, Lian, Liu, Jiang et al., *Scaling for Training
Time and Post-hoc Out-of-distribution Detection Enhancement*, ICLR 2024
**Benchmark variant:** `methods/scale/scale_enh/scale_enh.py` (`SCALEEnhDetector`, `percentile=65`)
**Base variant:** `benchmark1/models/ood_methods/scale.py` (`SCALEDetector`)
**Verified:** 2026-08-20

---

## Verdict

**CORRECTED.** `scale_enh` is a faithful reproduction of SCALE's post-hoc procedure. Its scaling
operation is **numerically identical** to the official `scale()` — verified by porting the reference
function verbatim and comparing outputs on the same inputs (max absolute difference 0 to 5.7e-6, i.e.
float32 noise; §4). Penultimate activations are rescaled by `exp(s1/s2)` and the scaled features are
passed through the classification head and scored with energy, exactly as the official code does.

**The base variant is not the method.** `scale.py` z-standardises the **logits** using ID mean/std
(`:22-23, 30`) and then computes energy. That is the wrong layer, the wrong operation, and it requires
a fitting stage the official method does not have. Confirmed unfaithful, as suspected.

**One substantive divergence remains in `scale_enh`: the percentile.** The paper states
*"results use p=0.85 for SCALE and ASH-S, which is verified on the validation set"*, and the official
config sets `percentile: 85` with automatic parameter search enabled. `scale_enh` defaults to **65**,
which matches only the `scale()` **function-signature** default — a value the official pipeline never
actually uses, because `forward_threshold` always passes the percentile explicitly. At p=65 the mean
sharpening factor is ≈2.94 against ≈5.55 at p=85, i.e. **roughly half the sharpening the paper
applies** (§5).

**The benchmark used `scale_enh`** — 40 datasets, versus 4 for the base variant.

---

## 1. Source accessibility

| Source | Status |
|---|---|
| Local clone `methods/scale/reference/` | Present, provenance verified: `origin = https://github.com/kai422/SCALE`, commit `6a6ab911228fae0cf95484be94484304177b77b0` (2024-03-12). An OpenOOD fork. |
| `https://raw.githubusercontent.com/kai422/SCALE/master/openood/networks/scale_net.py` | Fetched live (after one transient `socket hang up` on the `main` path); **matches the local clone verbatim**. |
| Paper, ar5iv (arXiv:2310.00227) | Fetched live. |
| `https://openreview.net/forum?id=RDSTjtnqCg` | **UNREACHABLE** — OpenReview served a browser-verification page, not content. Not used. |

### What the official code does

**`reference/openood/networks/scale_net.py:29-49`:**

```python
def scale(x, percentile=65):
    input = x.clone()
    b, c, h, w = x.shape
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    s2 = x.sum(dim=[1, 2, 3])
    scale = s1 / s2
    return input * torch.exp(scale[:, None, None, None])
```

Three details that are easy to get wrong, and all three matter:

1. `input = x.clone()` is taken **before** pruning, and the returned tensor scales the **original,
   unpruned** activations. The pruning exists only to compute `s2`. The paper confirms:
   *"Instead of pruning, it retains and scales **all** the activations."*
2. `t = x.view(...)` followed by `t.zero_().scatter_()` mutates `x` **in place** — that is how `s2`
   (line 44) becomes the top-k sum rather than the total.
3. `s1` is the total sum, `s2` the retained sum, so `s1/s2 ≥ 1` for non-negative activations.

**Wiring** (`scale_net.py:18-23`): penultimate feature → reshape to `(B, D, 1, 1)` → `scale(...)` →
flatten → `get_fc_layer()`.

**Score** (`reference/openood/postprocessors/scale_postprocessor.py:20-22`):

```python
output = net.forward_threshold(data, self.percentile)
energyconf = torch.logsumexp(output.data.cpu(), dim=1)
```

Energy, not negated (higher = ID).

**Percentile** (`reference/configs/postprocessors/scale.yml`): `percentile: 85`, with
`APS_mode: True` and `percentile_list: [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]`.

### What the paper says

- Layer: penultimate activations; *"Instead of pruning, it retains and scales all the activations."*
- Factor: `r = Σⱼ aⱼ / Σ_{aⱼ > Pₚ(a)} aⱼ`, applied as `exp(r)` — numerator the sum of all
  activations, denominator the sum of those above the p-th percentile. Matches `s1/s2`.
- Percentile: *"results use p=0.85 for SCALE and ASH-S, which is verified on the validation set."*
- Score: `S_EBO(x) = T·log Σₖ e^{zₖ/T}`, higher = ID.

## 2. Divergence table

| Component | Official | `scale_enh.py:line` | `scale.py:line` (base) | Changes results? |
|---|---|---|---|---|
| **Layer operated on** | penultimate activations (`scale_net.py:19-20`) | penultimate features via `_forward_features` (`:36`) | **logits** (`:26`) | `_enh`: **No**. base: **YES — wrong layer** |
| **Operation** | `feat * exp(s1/s2)` (`scale_net.py:49`) | `feats * exp(s1/s2)` (`:46`) | **z-standardisation** `(logits−μ)/σ` (`:30`) | `_enh`: **No** — numerically identical (§4). base: **YES — wrong operation** |
| `s1` | sum of all activations (`scale_net.py:36`) | `feats.sum(dim=1)` (`:40`) | n/a | **No** |
| `k` | `n − round(n·p/100)` (`scale_net.py:38`) | `max(1, d − round(d·p/100))` (`:41`) | n/a | **No** — `max(1,·)` only guards degenerate `d` |
| `s2` | sum after in-place top-k prune (`scale_net.py:41-44`) | `zeros.scatter_(topk).sum()` (`:42-44`) | n/a | **No** — same value, non-mutating |
| Scales original (not pruned) | `input.clone()` (`scale_net.py:30,49`) | multiplies `feats`, `pruned` is separate (`:43,46`) | n/a | **No** — correct on this subtle point |
| **Percentile** | **85** (`scale.yml`); paper p=0.85, APS sweep 50–95 | **65** (`:30`) — the function-signature default only | n/a | **YES — see §5** |
| Base score | energy `+logsumexp` (`scale_postprocessor.py:22`) | `_energy` = `−logsumexp` (`:50`) | `_energy` (`:31`) | **No** — sign flip, rank-invariant |
| Orientation | higher = ID | higher = OOD (`:50`) | higher = OOD (`:31`) | **No** — project convention |
| ReLU on features | not needed (activations already post-ReLU) | `torch.relu(feats)` added (`:37`) | n/a | Documented adaptation — see §3 |
| Numerical guards | none | `+1e-6` on `s2` (`:44`), `.clamp(max=50.0)` (`:45`) | `+1e-6` on σ (`:23`) | **No** — clamp verified never to bind (§5) |
| Requires ID fitting | **no** — fully per-sample | no `fit()` needed | **`fit()` estimates μ/σ** (`:20-23`) | base: **YES — extra stage** |
| Hyperparameter search | `APS_mode: True` | none | none | Divergence; see §6 |

## 3. The ReLU adaptation

`scale_enh.py:37` applies `torch.relu(feats)` before scaling, documented in the module docstring
(`:12-14`) and in `CHANGES.md`. This is legitimate and necessary: `s1/s2 ≥ 1` only holds for
non-negative activations, and the official method operates on post-ReLU CNN activations, whereas this
project's backbone exposes a linear embedding. Without the ReLU, negative features could make `s2`
near-zero or negative and the ratio meaningless.

It is nonetheless a real transformation the official does not perform — the features reaching the FC
head are rectified, not raw. It is correctly disclosed, and there is no better alternative given the
backbone, so it is recorded as a documented adaptation rather than a defect.

## 4. The scaling operation is numerically identical

I ported `reference/openood/networks/scale_net.py:29-49` verbatim (4-D, in-place prune, `input.clone()`)
and compared against `scale_enh.py:39-46` on identical post-ReLU inputs:

| Feature dim | percentile | max abs difference | Verdict |
|---|---|---|---|
| 512 | 65 | 0.000e+00 | identical |
| 512 | 85 | 0.000e+00 | identical |
| 64 | 65 | 1.907e-06 | identical (float32 noise) |
| 64 | 85 | 5.722e-06 | identical (float32 noise) |

The 2-D reformulation, the non-mutating `scatter_` into a zero tensor, and the `max(1, ·)` guard are
all behaviour-preserving. This is the strongest positive fidelity result in the verification set so
far.

## 5. The percentile divergence, quantified

Mean scaling exponent and factor over 2000 post-ReLU samples (D=512):

| percentile | units kept `k` | mean `s1/s2` | mean `exp(s1/s2)` | clamp hits (>50) |
|---|---|---|---|---|
| 50 | 256 | 1.0007 | 2.72 | 0 |
| **65 (mine)** | **179** | **1.0784** | **2.94** | **0** |
| 75 | 128 | 1.2567 | 3.52 | 0 |
| **85 (paper / official config)** | **77** | **1.7112** | **5.55** | **0** |
| 95 | 26 | 3.8424 | 47.69 | 0 |

At p=65 the sharpening is ≈2.94 against ≈5.55 at p=85 — **about half** the paper's operating point.
Because the factor is a *per-sample* multiplier, and the energy score `−logsumexp(clf(α·feat))`
depends on `α` non-linearly and `α` varies across samples, this is **not** a rank-preserving change:
the percentile genuinely affects AUROC. The paper's p=0.85 was *"verified on the validation set"*, so
65 is not a defensible default without its own validation.

**The `clamp(max=50.0)` guard never binds** — zero hits across p=50…95. It is harmless dead code in
practice (worth noting it would begin to matter near p=95, where the mean factor reaches ≈47.7).

## 6. Measured results

`scale_enh` (the benchmark variant):

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 40 | 0.2890 | — |
| TSB-U (univariate) | 21 | **0.2598** | 15/21 |

Paired against the base variant on the **4** shared datasets:

| Dataset | `scale_enh` | `scale` | Δ |
|---|---|---|---|
| TSB-M-DRIFT003 | 0.564 | 0.519 | +0.046 |
| TSB-U-DRIFT024 | 0.587 | 0.671 | −0.084 |
| TSB-U-OOD009 | 0.000 | 0.000 | +0.000 |
| TSB-U-STABLE001 | 0.694 | 0.755 | −0.061 |
| **mean** | **0.4613** | **0.4862** | **−0.0249** |

**The correction makes things worse on real data**, on this evidence: `scale_enh` is better on only
**1 of 4** and the mean drops by 0.025. `CHANGES.md` reports a synthetic improvement
(0.941 → 0.965) and states *"The decisive comparison is the Phase 2 sweep on real data."* That
comparison, as far as the artefacts allow, goes **against** the correction.

This does **not** mean the correction was wrong — `scale_enh` is the faithful implementation and the
base variant is not SCALE at all, so a faithful method scoring lower than an unfaithful one is a
finding about the *method's* suitability, not a reason to prefer the broken variant. But the claim that
Phase 2 would vindicate the correction should be replaced with what was actually measured. Note also
that the comparison is confounded by the untuned percentile (§5): p=65 is not the paper's operating
point, so this is not a clean test of faithful SCALE.

Structural checks: `scale_enh` spans `[−2.029e6, −1.405]`, 100% negative — consistent with
`−logsumexp`. Base `scale` spans `[−2.949, −1.37]`, a narrow band consistent with energy over
z-standardised (unit-scale) logits. The `−2.029e6` extreme is again the same order as the pathological
logits recorded in `methods/energy_ebo/VERIFICATION.md` §3.

## 7. Minor observations

- **Fallback duplicates the helper.** `scale_enh.py:48` inlines
  `-torch.logsumexp(feats_scaled / T, dim=-1)` on *features* rather than calling `_energy`, matching
  the pattern already noted in `methods/react/react_enh/react_enh.py:46` and
  `methods/dice/dice_enh/dice_enh.py:37`. Unreachable when a head exists, but the duplicated formula
  can drift.
- **No APS.** The official config enables automatic parameter search over the percentile
  (`APS_mode: True`). Neither variant sweeps it. Acceptable for a fixed-budget benchmark, but it
  should be stated, since the official numbers are produced with a tuned percentile.
- `CHANGES.md` accurately describes both the defect and the correction, including the ReLU and clamp
  adaptations. It is the most accurate of the `_enh` change notes reviewed so far — the only omission
  is the percentile question.

## 8. Recommendations

1. **Change the default percentile to 85**, or justify 65 with a validation sweep. The paper's value is
   explicit and validated; 65 is the signature default of a function whose default the official
   pipeline never exercises. Re-run `scale_enh` at p=85 before reporting SCALE.
2. **Correct the `CHANGES.md` validation claim** — the real-data comparison gives Δ = −0.025 on n=4,
   not a vindication. State the measured result and note the confound (untuned percentile, 4-dataset
   base coverage).
3. **Report the ReLU adaptation in the thesis**, as `CHANGES.md` already does — it is required by the
   backbone and does not compromise fidelity, but it is a real difference.
4. Optionally **drop the `clamp(max=50.0)`** or raise it: verified never to bind at p ≤ 85, so it adds
   no safety at the operating point while obscuring the formula.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded in
`methods/msp/VERIFICATION.md` §5, `methods/odin/VERIFICATION.md` §7,
`methods/energy_ebo/VERIFICATION.md` §6, `methods/mahalanobis_mds/VERIFICATION.md` §8,
`methods/dfm/VERIFICATION.md` §7, `methods/srs/VERIFICATION.md` §6,
`methods/react/VERIFICATION.md` §6 and `methods/dice/VERIFICATION.md` §7 remains unresolved.
`scale_enh` covers 21 univariate datasets.

## 9. Conclusion

`scale_enh` faithfully reproduces SCALE: it rescales penultimate activations by `exp(s1/s2)` with the
pruning used only to compute `s2`, scales the *original* activations rather than the pruned ones —
the subtle point the official `input.clone()` encodes and the paper states explicitly — passes the
result through the classification head, and scores with energy. A verbatim port of the reference
function reproduces its output to float32 precision. The base `scale.py` is not SCALE: it
z-standardises logits, operating on the wrong layer with the wrong operation and requiring an ID
fitting stage the method does not have. The one open issue in the corrected variant is the percentile:
65 rather than the paper's validated 0.85, which halves the sharpening factor and confounds the
original-versus-corrected comparison.
