# ReAct Faithfulness Verification — CORRECTED (`react_enh` is faithful and is the variant the benchmark used)

**Method id:** `react` / `react_enh` · **Paper:** Sun, Guo & Li, *ReAct: Out-of-distribution Detection
With Rectified Activations*, NeurIPS 2021 (arXiv:2111.12797)
**Benchmark variant:** `methods/react/react_enh/react_enh.py` (`ReACTEnhDetector`, `percentile=90`)
**Base variant:** `benchmark1/models/ood_methods/react.py` (`ReACTDetector`, ReAct+MSP)
**Verified:** 2026-08-20

---

## Verdict

**CORRECTED.** `react_enh` is a faithful reproduction of the paper's headline configuration —
penultimate activations clipped at the 90th-percentile ID threshold, then the **energy** score. The
clipping, the threshold estimation, the percentile, and the base score all match the official code and
the paper.

**The benchmark used `react_enh`.** Confirmed empirically: `react_enh` has results on **40** datasets,
the base `react` on only **4**.

One wording correction: `react_enh/CHANGES.md` describes the base variant's MSP score as an
"inconsistency". That overstates it. The paper says ReAct is *"compatible with several commonly used
OOD scoring functions"* including softmax confidence, and the official repo ships `get_msp_score`
(`score.py:7-12`) alongside `get_energy_score`. ReAct+MSP is a **supported non-headline
configuration**, not a defect. The correction to energy is still the right call — energy is the paper's
primary setting — but the base variant should be described as "ReAct+MSP, a valid but non-headline
variant", not as incorrect.

---

## 1. Source accessibility

| Source | Status |
|---|---|
| Local clone `methods/react/reference/` | Present. `origin = https://github.com/deeplearning-wisc/react`, commit `2aa35d9993ddb00409dc41824dbc008b5cc16e20` (2022-03-23). Contains `compute_threshold.py`, `score.py`, `eval.py`, `models/resnet.py`. |
| `https://raw.githubusercontent.com/deeplearning-wisc/react/master/score.py` | Fetched live; **matches the local clone verbatim**. |
| Paper (arXiv:2111.12797) via ar5iv | Fetched live. |

### What the sources say

**Paper:** `ReAct(x,c) = min(x,c)`, applied element-wise to the penultimate feature vector to *"limit
the effect of noise"*. Headline score is **energy**, though the method is *"compatible with several
commonly used OOD scoring functions"*. Threshold: the **90th percentile** — *"when p=90, it indicates
that 90% of the ID activations are less than the threshold c."* Energy uses the negative log-sum-exp
convention (higher = OOD).

**Official clipping** (`reference/models/resnet.py:306-313`):

```python
def forward_threshold(self, x, threshold=1e10):
    x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
    x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
    x = self.avgpool(x)
    x = x.clip(max=threshold)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
```

Clip the post-avgpool penultimate activations, then the FC head. (Also at `:404-410`.)

**Official threshold estimation** (`reference/compute_threshold.py`):

- Hooks `model.avgpool` output (`:78`, `:80`), spatial-mean pooled (`:87`).
- `np.percentile(activation_log.flatten(), 90)` (`:96`) — **flattened across samples *and* channels**.
- Uses the ID **val** split (`:52`), capped at `lim = 2000` samples (`:63`).

**Official scores** (`reference/score.py`):

- `get_energy_score` (`:19`): `scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()` — **not negated**.
- `get_msp_score` (`:11`): `np.max(F.softmax(logits, dim=1)...)` — not negated.

Both use the "higher = ID" convention.

## 2. Divergence table — `react_enh` (the benchmark variant)

| Component | Official | Mine (`react_enh.py:line`) | Changes results? |
|---|---|---|---|
| Clipping operation | `x.clip(max=threshold)` on post-avgpool features (`resnet.py:310`) | `torch.clamp(feats, max=self.threshold)` (`:42`) | **No** — `min(x,c)` identical |
| Clipping location | inside forward, post-avgpool, before FC (`resnet.py:309-312`) | `_forward_features` → clamp → `clf(feats)` (`:39-47`) | **No**, given `_forward_features` returns the penultimate vector and `clf` is the final head |
| Threshold statistic | `np.percentile(activation_log.flatten(), 90)` — global over samples **and** channels (`compute_threshold.py:96`) | `np.percentile(feats_numpy, 90)` — numpy flattens by default (`:34`) | **No** — exact match on the subtle point (global, not per-channel) |
| Percentile | 90 (`compute_threshold.py:95-96`; paper p=90) | `percentile=90.0` default (`:26`) | **No** |
| Threshold data | ID **val** split, capped at 2000 samples (`compute_threshold.py:52,63`) | whatever `x_id` is passed to `fit()` (`:30-34`) | Minor — no sample cap; see §4 |
| Base score | energy: `+logsumexp` (`score.py:19`) | `self._energy(logits)` = `−logsumexp` (`:48`) | **No** — sign flip only, rank-invariant |
| Orientation | higher = ID (official code); higher = OOD (paper convention) | higher = OOD (`:48`) | **No** — matches the paper's convention |
| Temperature | none (plain `logsumexp`) | exposed, default `1.0` (`:27`) | **No** at the default |
| Training | post-hoc | `fit()` only estimates the threshold (`:30-34`) | **No** |

### Base variant `react.py` — the difference is the score family only

| Component | `react_enh.py` | `react.py` | Note |
|---|---|---|---|
| Threshold | `:34`, p=90 global percentile | `:23`, identical | same |
| Clipping | `:42` `clamp(max=τ)` | `:32`, identical | same |
| Base score | **energy** `−logsumexp` (`:48`) | **MSP** `1 − softmax_max` (`:36`) | paper headline vs a supported alternative |
| No-head fallback | energy on clipped features (`:46`) | `feats.norm(dim=-1)` (`:38`) | both unreachable when a head exists |

So the base variant implements ReAct+MSP correctly; it simply is not the headline configuration.

## 3. Structural confirmation from saved scores

| Variant | Datasets | Score range | Consistent with? |
|---|---|---|---|
| `react_enh` | **40** | `[−1.056e6, −1.359]`, 100% negative | `−logsumexp`; the maximum `−1.359` sits just above `−log 4 = −1.386`, the uniform-logit floor for K=4 |
| `react` (base) | **4** | `[2.98e−5, 0.6948]` | `1 − max softmax`, bounded by `1 − 1/4 = 0.75` for K=4 |

Both ranges match their declared formulas, and both independently corroborate K=4 pseudo-classes.

## 4. Measured results

`react_enh` (the benchmark variant):

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 40 | 0.2835 | — |
| TSB-U (univariate) | 21 | **0.2504** | 16/21 |

Paired against the base variant on the **4** shared datasets:

| Dataset | `react_enh` | `react` | Δ |
|---|---|---|---|
| TSB-M-DRIFT003 | 0.549 | 0.495 | +0.055 |
| TSB-U-DRIFT024 | 0.577 | 0.533 | +0.045 |
| TSB-U-OOD009 | 0.000 | 0.188 | −0.188 |
| TSB-U-STABLE001 | 0.612 | 0.449 | +0.163 |
| **mean** | **0.4347** | **0.4159** | **+0.019** |

`react_enh` is better on 3 of 4.

**Important caveat.** `react_enh/validation_status.json` states the *"decisive comparison is Phase 2 on
real data"*, and `CHANGES.md` says *"the score-family difference is expected to matter on real
data"*. **That comparison was never actually made:** the base variant ran on only 4 of 40 datasets, so
the paired evidence is n=4 — directionally favourable but far too thin to support a claim about the
score family. Either run the base variant across all 40 datasets or drop the claim.

`react_enh`'s below-chance mean is consistent with the pattern established for the other
logit-magnitude detectors (EBO TSB-U 0.277, ODIN 0.261): clipping at the 90th percentile bounds the
activations but does not prevent the backbone's inflated logits on far-off-manifold windows from
dominating the energy score. Note the extreme value `−1.056e6`, of the same order as the pathological
logits recorded in `methods/energy_ebo/VERIFICATION.md` §3.

## 5. Minor observations

- **No-head fallback duplicates the helper.** `react_enh.py:46` inlines
  `-torch.logsumexp(feats / T, dim=-1)` on *features* rather than calling `_energy`, matching the same
  pattern noted in `methods/scale/scale_enh/scale_enh.py:48` and `methods/dice/dice_enh/dice_enh.py:37`.
  Unreachable when a classifier head is configured, but the duplicated formula can drift.
- **The official repo also contains a different, non-headline rectification**: `BasicBlock` /
  `Bottleneck.forward_threshold` (`resnet.py:86-106`, `:174-196`) zero whole channels
  (`mask = out.view(b,c,-1).mean(2) < threshold; out = mask * out`) for mid-network layers. That is not
  the paper's headline operation, and `react_enh` correctly implements the penultimate clip instead.
- **Threshold sample cap.** The official caps threshold estimation at 2000 ID samples; mine uses the
  full `x_id`. With ~82 training windows this is immaterial, but it is a difference worth recording.
- **Stale artefact.** `methods/react/react_enh/__pycache__/react_enh.cpython-313.pyc` is checked into
  the method folder. Harmless, but it is build output.

## 6. Recommendations

1. **Reword `CHANGES.md`** — describe the base variant as ReAct+MSP (a paper-supported, non-headline
   configuration), not as an "inconsistency". The paper explicitly permits MSP; only the *headline*
   claim requires energy.
2. **Either complete or retract the ablation.** Run base `react` on all 40 datasets, or remove the
   "decisive comparison in Phase 2" claim from `validation_status.json` and `CHANGES.md`. Present n=4
   evidence as n=4.
3. **Report `react_enh` as the ReAct result** and state the percentile (90) and score family (energy)
   in the results table.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded in
`methods/msp/VERIFICATION.md` §5, `methods/odin/VERIFICATION.md` §7,
`methods/energy_ebo/VERIFICATION.md` §6, `methods/mahalanobis_mds/VERIFICATION.md` §8,
`methods/dfm/VERIFICATION.md` §7 and `methods/srs/VERIFICATION.md` §6 remains unresolved.
`react_enh` covers 21 univariate datasets here.

## 7. Conclusion

`react_enh` faithfully reproduces ReAct's headline configuration: `min(x, c)` on penultimate
activations with `c` set at the 90th percentile of ID activations computed globally over samples and
channels — matching `compute_threshold.py:96` exactly, including the easily-mistaken detail that the
percentile is global rather than per-channel — followed by the energy score. The only divergences are a
metric-invariant sign flip (which actually aligns with the paper's stated convention rather than the
official code's), an exposed temperature that is inert at its default, and the absent 2000-sample cap
on threshold estimation. The benchmark used this variant. The base ReAct+MSP variant is also correctly
implemented but is a non-headline configuration and, with only 4 datasets, does not currently support
the ablation claim made for it.
