# DivOE Faithfulness Verification — NOT-THE-METHOD (energy on mean-centred logits; no synthesis, no training)

**Method id:** `divoe` · **Paper:** Zhu, Lin, Zhou, Yang, Yang, Liu et al., *Diversified Outlier
Exposure for Out-of-Distribution Detection via Informative Extrapolation*, NeurIPS 2023
**Official code:** `https://github.com/ZFancy/DivOE`
**Implementation:** `benchmark1/models/ood_methods/divoe.py` (`DivOEDetector`)
**Verified:** 2026-08-20

---

## Verdict

**NOT-THE-METHOD.** DivOE's contribution is **synthesising diversified outliers by multi-step
optimisation ("informative extrapolation") and then training with the Outlier Exposure loss on the
augmented outlier set**. This implementation performs neither synthesis nor training. It computes an
energy score on mean-centred logits from the shared ID-trained backbone.

A static scan of `divoe.py` shows the absence is total:

| Token | Present? |
|---|---|
| `extrapolate` / `num_steps` / `epsilon` / `sign()` | **No** |
| `backward()` / `optimizer` / `Adam` / `SGD` / `train()` / `requires_grad` | **No** |
| `out_set` / `ood_loader` (auxiliary outliers) | **No** |
| `cross_entropy` (the OE loss) | **No** |

`fit()` (`:19-21`) does exactly one thing: `self.logit_mean = logits.mean(dim=0, keepdim=True)`.
`score()` (`:23-29`) subtracts that vector and returns `self._energy(centered)`.

**The mean-centring is an invented step with no counterpart in the paper or official code** — and it is
very nearly inert. Vector mean-centring gives Spearman **0.99991** against plain energy on synthetic
logits (§3B); on the saved scores, `divoe` correlates with `outlier_exposure` (plain energy) at mean
ρ **0.9470** / median **0.9834**, with mean AUROC **0.2763 vs 0.2949** — i.e. the extra step slightly
*hurts* (−0.0186).

**Third redundant energy row.** `outlier_exposure` (40 dirs) is plain energy and is also the benchmark's
EBO result (`energy_ebo`: 0 dirs — see `methods/outlier_exposure/VERIFICATION.md` §3D). `divoe` is that
same energy score with a near-inert shift. So **EBO, Outlier Exposure and DivOE are three rows backed by
one detector**, two of them literally the same files.

**There is no docstring.** Like `outlier_exposure.py`, this file has no module or class docstring — no
"(Adapted)" note, no statement that synthesis and training are omitted — while being registered as
`divoe` (`:12`).

---

## 1. Source accessibility

| Source | Status |
|---|---|
| `methods/divoe/reference/` | Present and intact: `origin = https://github.com/ZFancy/DivOE`, commit `722c3369e9bf32a870ec6d14f586c0782a787ddd` (2023-10-06), `HEAD → refs/heads/main`. Contains `src/train_DivOE.py`, `train.py`, `test.py`, `main_ImageNet_DivOE.py`. |
| `raw.githubusercontent.com/ZFancy/DivOE/main/src/train_DivOE.py` | Fetched live; **matches the local clone verbatim**. |
| Paper (NeurIPS 2023) | **Not fetched.** The mechanism question is settled unambiguously by the official training code — the argparse flags are literally named `extrapolation_ratio`, `extrapolation_score`, `num_steps`, matching the paper's "informative extrapolation". Stated rather than implied: no paper text was read, and nothing below rests on it. |

### The official mechanism

**Outlier synthesis** — `reference/src/train_DivOE.py:177-197`, confirmed by live fetch:

```python
def extrapolate(model, data, epsilon, rel_step_size=1/4, num_steps=5, rand_init=True):
    x_adv = data.detach() + uniform(-epsilon, epsilon)          # random init in the eps-ball
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        if args.extrapolation_score == 'MSP':
            loss_adv = -(output.mean(1) - torch.logsumexp(output, dim=1)).mean()   # OE loss
        elif args.extrapolation_score == 'energy':
            loss_adv = torch.pow(F.relu(args.m_out - Ec_out), 2).mean()
        loss_adv.backward()
        eta = (epsilon * rel_step_size) * x_adv.grad.sign()
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)        # project
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv
```

A PGD-style multi-step ascent **in input space** on the OE objective, projected back into an L∞ ball.

**Augmentation and training** — `:211-212` and the OE loss:

```python
aug_length = int(len(out_set[0]) * args.extrapolation_ratio)
adv_outlier = extrapolate(net, out_set[0][:aug_length], args.epsilon, args.rel_step_size, args.num_steps)
data = torch.cat((in_set[0], adv_outlier.cpu(), out_set[0][aug_length:]), 0)
...
loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
```

Half the outlier batch is replaced by extrapolated versions, then the standard 0.5-weighted
cross-entropy-to-uniform OE penalty is applied and **backpropagated into the network**.

Defaults (`:60-64`): `extrapolation_ratio=0.5`, `epsilon=0.01`, `rel_step_size=1/4`, `num_steps=5`,
`extrapolation_score='MSP'`.

## 2. Divergence table

| Component | Official | Mine (`divoe.py:line`) | Changes results? |
|---|---|---|---|
| **Outlier synthesis (extrapolation)** | 5-step PGD on the OE loss, ε=0.01, step ε/4, L∞ projection (`train_DivOE.py:177-197`) | **absent** | **YES — the paper's contribution** |
| **Auxiliary outlier set** | `out_set`, half replaced by synthesised outliers (`:211-212`) | **absent** | **YES** |
| **OE training loss** | `0.5 * -(mean − logsumexp)` on outliers, backpropagated | **absent** | **YES** |
| **ID cross-entropy term** | `F.cross_entropy` on the ID half | **absent** | **YES** |
| **Parameter updates** | yes — `loss.backward()`, `optimizer.step()` | **none** — `fit()` only stores a logit mean (`:19-21`) | **YES** |
| **What is scored** | the *trained* network | the shared **ID-only** backbone (`:24`) | **YES** |
| **Mean-centring of logits** | **nowhere in the paper or code** | `centered = logits - self.logit_mean` (`:27`) | Invented; near-inert (§3B) |
| Score function | MSP or energy on the trained net (`test.py:123-126`) | `_energy(centered, T)` (`:28`) | — |
| Temperature | n/a | exposed, default 1.0 (`:16`) | Inert at default |
| Orientation | higher = more anomalous | higher = OOD | **No** |
| Labelling | — | **no docstring**; registered as `divoe` (`:12`) | See §4 |

## 3. Empirical findings

**(A) Static confirmation** — table in the Verdict. Not one element of DivOE's mechanism appears in the
file; even `logsumexp` is absent, since the energy is delegated to `BaseOODDetector._energy`.

**(B) Mean-centring by a vector is *technically* not rank-preserving, but practically almost is.**
`energy(z − c) = −logsumexp(z − c)`. For a **scalar** `c` this is exactly `energy(z) + c` — a constant
offset, rank-identical. For a **vector** `c` (what `:21` computes) the classes are reweighted
non-uniformly, so ranking can change:

| Centring | Spearman vs plain energy | Note |
|---|---|---|
| vector mean (as implemented) | **0.999910** | max score change 0.0690 |
| scalar mean (contrast) | **1.000000** | offset constant to 4.47e-07 |

So the step is a real modification, just a very small one — and it corresponds to nothing in the paper.

**(C) Saved scores.**

| | mean AUROC (40) | TSB-U (21) | below chance |
|---|---|---|---|
| `divoe` | **0.2763** | **0.2299** | 17/21 |
| `outlier_exposure` (plain energy) | 0.2949 | 0.2770 | 15/21 |
| **Δ (divoe − energy)** | **−0.0186** | −0.0471 | — |

Correlation between the two across 40 paired datasets: mean ρ **0.9470**, median **0.9834**, with
16/40 above 0.99 and 28/40 above 0.95. Scores span `[−1.75e+06, −1.368]`, 100% negative — the same
`−logsumexp` signature and the same pathological-magnitude extreme as `outlier_exposure`.

**Net effect: the invented mean-centring makes an energy baseline slightly worse and does not make it
DivOE.**

## 4. Labelling

`divoe.py` has **no docstring** — no module docstring, no class docstring, no comment noting that
synthesis and training are omitted. It is registered under the paper's name (`@register_ood("divoe")`,
`:12`), so the results table reads "DivOE".

This is the second such case, alongside `outlier_exposure.py`. Every Tier C adaptation reviewed in this
audit carried an explicit "(Adapted)" note; both Tier D training-based methods carry none. The pattern
is worth fixing together.

## 5. Recommendations

1. **Exclude `divoe` from the results, or relabel it.** It is not DivOE. If retained, the honest label
   is "Energy (mean-centred logits)" — and since that is measurably worse than plain energy
   (−0.019 AUROC) while corresponding to nothing in the paper, exclusion is the cleaner choice.
2. **Collapse the redundant energy rows.** EBO, Outlier Exposure and DivOE are one detector reported
   three times: `energy_ebo` has 0 result dirs, `outlier_exposure` and `divoe` share the same backbone
   and correlate at median ρ = 0.983. Report Energy once.
3. **Remove the mean-centring** if the row is kept for any purpose — it has no basis in the paper and
   costs accuracy.
4. **Add a docstring** stating that DivOE's extrapolation and OE training are not implemented. Same fix
   as `methods/outlier_exposure/VERIFICATION.md` §5.3.
5. **If DivOE proper is wanted**, it needs an auxiliary outlier corpus, the 5-step input-space PGD of
   `train_DivOE.py:177-197`, and a fine-tuning loop — all training-time changes, out of scope for a
   post-hoc benchmark. That is a legitimate reason to drop the method rather than mislabel it.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded
across `methods/*/VERIFICATION.md`. `divoe` covers **21** univariate datasets; see
`methods/diversify/VERIFICATION.md` §4 and `methods/catsight/VERIFICATION.md` §4 for the candidate
explanation of the "18" figure.

## 6. Conclusion

DivOE is defined by generating diversified outliers through a 5-step projected-gradient ascent on the
Outlier Exposure objective in input space, mixing them into the auxiliary outlier batch at a 0.5 ratio,
and fine-tuning the classifier with the OE loss — all verified against the local clone and a matching
live fetch of `train_DivOE.py`. None of that exists in `divoe.py`, which contains no synthesis, no
auxiliary data, no loss, no optimiser and no docstring. What it computes is the energy score on
logits shifted by a stored training-set mean vector: a modification with no counterpart in the paper
that leaves ranking almost unchanged (ρ = 0.9999 synthetically, median 0.983 on real data) and lowers
mean AUROC by 0.019 relative to plain energy. The defining mechanism is absent rather than simplified,
and the row duplicates the Energy/EBO result already reported twice elsewhere.

## CLASS-D BUILD (2026-08-21)

**Faithful OE + DivOE-synthesis build — appendix study, both arms.** The gap
identified above (no synthesis, no aux data, no OE loss) is now closed in a
*separate* Class-D appendix build that never touches production `ood_methods` or the
17-method leaderboard.

- **Module:** `methods/divoe/classd/divoe_classd.py` — `extrapolate_pgd()`
  (input-space PGD synthesis, a faithful port of `reference/src/train_DivOE.py:177-200`)
  plus `DivOEClassD`, the same ENERGY scorer as OE over the DivOE-fine-tuned net.
- **Synthesis:** for each aux batch, an `extrapolation_ratio=0.5` fraction is
  replaced by `num_steps=5` sign-gradient ascent steps on the OE-uncertainty loss,
  projected into an ε-ball (ε in NORMALISED-WINDOW units, ε=0.1; the image-domain
  `[0,1]` clamp is intentionally dropped). PGD runs in raw input space on the
  fine-tuned copy, per `CLASS_D_DECISIONS.md` §7.
- **Fine-tuning:** `experiments/run_class_d.py::finetune_divoe()` (parallel to
  `finetune()`) folds the synthesised ∪ aux outliers into `CE(id) + 0.5·CE_to_uniform`.
- **Aux corpus / arms / no-mutation:** identical infrastructure to OE — real
  channel-matched hold-out TSB windows (no leakage), BOTH arms (`head_only`,
  `full_net`), each on a deep copy so the shared frozen backbone is never mutated
  (verified: arms differ on every file).
- **Verification (U split, 2 files/cell, seed 42):** all 12 DivOE runs produced
  FINITE per-sample AUROCs → `results/class_d_group1.csv`. Example rows: OOD_001
  head_only 0.1472 / full_net 0.3296; OOD_002 head_only 0.8402 / full_net 0.6095.
- **Caveats:** BREAKS the frozen-backbone fair comparison (training + synthesis) →
  appendix only. Input-space PGD through a 1-D ResNet on CPU is the cost driver.
  On these tiny ID sets DivOE tracks plain OE closely (report the OE-vs-DivOE delta
  honestly). Tiny test sets make per-file AUROCs high-variance; aggregate over the
  full partition.
