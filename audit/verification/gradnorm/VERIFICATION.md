# GradNorm Faithfulness Verification — CORRECTED (`gradnorm_enh` is faithful; the base variant is wrong in all four respects)

**Method id:** `gradnorm` / `gradnorm_enh` · **Paper:** Huang, Geng & Li, *On the Importance of
Gradients for Detecting Distributional Shifts in the Wild*, NeurIPS 2021 (arXiv:2110.00218)
**Benchmark variant:** `methods/gradnorm/gradnorm_enh/gradnorm_enh.py` (`GradNormEnhDetector`, T=1)
**Base variant:** `benchmark1/models/ood_methods/gradnorm.py` (`GradNormDetector`)
**Verified:** 2026-08-20

---

## Verdict

**CORRECTED.** `gradnorm_enh` is a faithful reproduction. It backpropagates the KL-to-uniform
objective, takes the gradient with respect to the **last classification-layer weights**, uses the
**L1** norm, and negates it for the project's higher-is-OOD convention. Two non-obvious equivalences
between it and the official code were verified to hold **exactly** (difference 0.000e+00; §4).

**The base variant is wrong in all four respects the prompt suspected — confirmed:**

| Axis | Official / paper | `gradnorm.py` |
|---|---|---|
| Gradient variable | last-FC **weight** parameters | the network **input** (`:19,23`) |
| Loss | **KL to uniform** | **cross-entropy to the argmax label** (`:20-21`) |
| Norm | **L1** | **L2** (`:24`) |
| Orientation | higher = **ID** | higher = **OOD** (`:25-27`) |

It also **misattributes** its orientation to the paper. `gradnorm.py:25` reads
*"Higher gradient norm = more uncertain prediction = more OOD (Huang et al. 2021)"*. The paper states
the opposite: *"the magnitude of gradients is higher for in-distribution (ID) data than that for OOD
data."* The citation should be removed along with the code.

**The benchmark used `gradnorm_enh`** — 40 datasets, versus 4 for the base variant.

**Notably, this is the only `_enh` correction that clearly pays off on real data:** +0.1217 mean AUROC
over the base variant, better on 3 of 4 shared datasets (§5). Compare ReAct +0.019, DICE +0.0004,
SCALE −0.025.

---

## 1. Source accessibility

| Source | Status |
|---|---|
| Local clone `methods/gradnorm/reference/` | Present, provenance verified: `origin = https://github.com/deeplearning-wisc/gradnorm_ood`, commit `18d5d332fd32262f6b4e0eb22328b04b858a65b4` (2022-08-02). |
| `https://raw.githubusercontent.com/deeplearning-wisc/gradnorm_ood/master/test_ood.py` | **Live re-fetch FAILED** (`socket hang up`). The code comparison rests on the local clone, whose provenance and commit are verified above. |
| Paper (arXiv:2110.00218) | **Obtained**, but only after `ar5iv`, `arxiv.org/abs` and the PDF renderer all failed. Text extracted directly from the fetched PDF (17 pages, 56,827 chars) with `pypdf`. All paper quotations below are from that extraction. |

**One thing to flag about method:** when the PDF fetch returned unreadable stream data, the fetch tool
offered a summary prefixed *"from my training knowledge, not this document"*. That content was
discarded and replaced with the verbatim extraction below. No claim in this report rests on recall.

### What the paper says

- Score: *"GradNorm directly employs the vector norm of gradients, backpropagated from the KL
  divergence between the softmax output and a uniform probability distribution."*
- Orientation: *"the magnitude of gradients is higher for in-distribution (ID) data than that for OOD
  data, making it informative for OOD detection."*
- Layer: *"last layer parameters: weight parameters from the last fully connected (FC) layer"*; the
  distribution figures are axis-labelled **"L1 Norm of Last Layer Gradients"**.
- Equation (4): `∂D_KL(u‖softmax(f(x)))/∂w = (1/C) Σᵢ ∂L_CE(f(x), i)/∂w`, i.e. *"the gradient of KL
  divergence is equivalent to averaging the derivative of the categorical cross-entropy loss for all
  labels."*
- Temperature `T` appears in the cross-entropy of Eq. (1); the method needs *"no hyper-parameter
  tuning."*

### What the official code does

`reference/test_ood.py:124-145`:

```python
def iterate_data_gradnorm(data_loader, model, temperature, num_classes):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        inputs = Variable(x.cuda(), requires_grad=True)
        model.zero_grad()
        outputs = model(inputs)
        targets = torch.ones((inputs.shape[0], num_classes)).cuda()
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))
        loss.backward()
        layer_grad = model.head.conv.weight.grad.data
        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(layer_grad_norm)
    return np.array(confs)
```

with `--temperature_gradnorm default=1` (`:267`).

**`confs.append(...)` runs once per batch — but this is not a bug.** `main()` forces
`args.batch = 1` for GradNorm (`test_ood.py:220-221`), and correspondingly disables `DataParallel`
(`:232`), which would otherwise split the batch and corrupt the single accumulated gradient. The
official scoring is therefore genuinely per-sample. (Contrast `methods/srs/VERIFICATION.md` §4, where
the official code has no such guard and its per-sample likelihood really is batch-constant.)

## 2. Divergence table

| Component | Official / paper | `gradnorm_enh.py:line` | `gradnorm.py:line` (base) | Changes results? |
|---|---|---|---|---|
| **Gradient variable** | last-FC **weight** (`test_ood.py:140`) | `clf.weight.grad` (`:57`) | **input** `x_tensor.grad` (`:19,23`) | `_enh`: **No**. base: **YES** |
| **Loss** | KL-to-uniform via ones-target CE (`test_ood.py:134-136`) | identical expression (`:54-55`) | **CE to argmax label** (`:20-21`) | `_enh`: **No**. base: **YES** |
| **Norm** | **L1**, `sum(abs(·))` (`test_ood.py:142`) | `abs().sum()` (`:57`) | **L2**, `norm(p=2)` (`:24`) | `_enh`: **No**. base: **YES** |
| **Orientation** | higher = **ID** | negated → higher = OOD (`:58`) | higher = OOD, no negation (`:26-27`) | `_enh`: **No** — rank-invariant flip. base: **YES — inverted** |
| Temperature | `default=1` (`test_ood.py:267`) | `1.0` (`:32`), applied (`:53`) | **absent** | **No** at default |
| Per-sample scoring | batch forced to 1 (`test_ood.py:220-221`) | explicit per-sample loop (`:46`) | batched | `_enh`: **No** — equivalent |
| Gradient zeroing | `model.zero_grad()` (`test_ood.py:132`) | `clf.weight.grad = None` (`:51-52, 59-60`) | none | **No** — see §4 |
| Feature graph | full graph through backbone | features under `no_grad`, then head only (`:48-50`) | full graph to input | **No** — verified identical (§4) |
| Head type | `model.head.conv.weight` (1×1 conv head, BiT) | `clf.weight` (Linear) | n/a | **No** — same role |
| Bias gradient | not used | not used | n/a | **No** |

## 3. What `CHANGES.md` claims

`gradnorm_enh/CHANGES.md` states the base variant backpropagated CE to the input and reported the L2
input-gradient norm, and that the correction switches to (i) KL-to-uniform, (ii) gradient w.r.t.
classifier weights, (iii) L1 norm, (iv) negation for the project convention. **All four claims are
accurate and all four are implemented.** This is the most accurate `_enh` change note in the
verification set — it correctly identifies every defect, including the orientation inversion that the
other notes tend to gloss.

## 4. Two equivalences verified exactly

**(a) Computing features under `no_grad` does not change the head-weight gradient.** The official runs
the full graph (`model(inputs)`) and reads only `model.head.conv.weight.grad`; `gradnorm_enh` computes
features under `torch.no_grad()`, detaches them, and backprops through the head alone (`:48-56`). For
a linear head `z = W f + b`, `∂L/∂W = (∂L/∂z) ⊗ f`, which depends only on `f` and `∂L/∂z` — not on any
upstream parameter. Verified numerically on a two-layer backbone plus head:

| | official-style (full graph) | `gradnorm_enh`-style (detached) | abs diff |
|---|---|---|---|
| sample 0 | 3.8606972694 | 3.8606972694 | 0.00e+00 |
| sample 1 | 4.6638321877 | 4.6638321877 | 0.00e+00 |
| sample 2 | 5.8256111145 | 5.8256111145 | 0.00e+00 |
| sample 3 | 3.1128270626 | 3.1128270626 | 0.00e+00 |
| **max over 8 samples** | — | — | **0.000e+00** |

Identical. The detached formulation is also cheaper — no backbone gradients are ever allocated, which
is why zeroing only `clf.weight.grad` rather than the whole model is sufficient.

**(b) The ones-target loss is an affine function of KL-to-uniform, so the L1 ranking is unchanged.**
The official loss is `Σₖ −1·log softmax(z)ₖ`, not literally `D_KL(u‖p)`. With `K` classes:

```
Σₖ −log pₖ  =  K · KL(u‖p) + K · log K
```

Verified over 2000 random logit vectors: `max |ones_loss − (K·KL + K·log K)| = 0.000e+00`. The
gradient therefore differs from the paper's `∂D_KL/∂w` by exactly the positive factor `K`, leaving the
L1 ordering across samples untouched. This is precisely what the paper's Eq. (4) asserts, so the code
and the paper agree and both agree with `gradnorm_enh`.

## 5. Measured results

`gradnorm_enh` (the benchmark variant):

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 40 | 0.2890 | — |
| TSB-U (univariate) | 21 | **0.2474** | 16/21 |

Paired against the base variant on the **4** shared datasets:

| Dataset | `gradnorm_enh` | `gradnorm` | Δ |
|---|---|---|---|
| TSB-M-DRIFT003 | 0.563 | 0.526 | +0.036 |
| TSB-U-DRIFT024 | 0.619 | 0.515 | +0.103 |
| TSB-U-OOD009 | 0.000 | 0.000 | +0.000 |
| TSB-U-STABLE001 | 0.776 | 0.429 | **+0.347** |
| **mean** | **0.4892** | **0.3675** | **+0.1217** |

Better on **3 of 4**, and by a wide margin on `TSB-U-STABLE001` — where the base variant's inverted
orientation is most visible (0.429, below chance, versus 0.776). This is the clearest real-data
vindication of any `_enh` correction reviewed:

| Variant | Δ (enh − base), n=4 | enh better |
|---|---|---|
| **GradNorm** | **+0.1217** | 3/4 |
| ReAct | +0.0188 | 3/4 |
| DICE | +0.0004 | 1/4 |
| SCALE | −0.0249 | 1/4 |

The caveat that applies to all four still applies here: base coverage is 4 of 40 datasets, so n=4.
The direction and magnitude are encouraging but the ablation remains underpowered.

Structural checks: `gradnorm_enh` spans `[−3.22e8, −0.6093]` and is **100% negative** — consistent
with `score = −L1` where `L1 ≥ 0`. Base `gradnorm` spans `[7.37e−9, 1.784]` and is **100%
non-negative** — consistent with an unnegated L2 norm. Both match their declared formulas.

## 6. Minor observations

- **Per-sample Python loop.** `gradnorm_enh:46` runs one backward pass per sample, i.e. `O(N)`
  backward passes. This is unavoidable — the official does the same via `args.batch = 1` — because the
  weight gradient accumulates over a batch and cannot be decomposed per sample after the fact. It is
  faithful, just slow.
- **Degenerate fallback.** `gradnorm_enh:38-41` returns `−feats.abs().sum(dim=-1)` when no classifier
  head is available. That is a feature L1 norm, not a gradient norm — it is not GradNorm in any sense.
  It is labelled a "documented degenerate case" in the source, which is honest, but the branch would
  silently produce a non-GradNorm score if a head were ever missing. Consider raising instead.
- **No temperature in the base variant.** `gradnorm.py` has no temperature parameter at all, so it
  could not reproduce the official even with the other three axes fixed.

## 7. Recommendations

1. **Delete or clearly quarantine `gradnorm.py`.** It is wrong on all four axes and its comment
   (`:25`) attributes to Huang et al. the opposite of what the paper says. If it is retained for the
   ablation, remove the citation and add a header stating it is a known-incorrect variant.
2. **Report `gradnorm_enh` as the GradNorm result** and state T=1 and the L1-of-last-layer-weight
   gradient in the results table.
3. **Use GradNorm as the worked example** for the corrected-variant narrative — it is the case where
   the correction demonstrably matters on real data (+0.122), unlike DICE (+0.0004) and SCALE (−0.025).
4. **Systemic:** run all four base variants across the full 40 datasets, or restate the four
   "decisive Phase 2 comparison" claims as the n=4 results they are. This is one fix, not four.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded in
`methods/msp/VERIFICATION.md` §5, `methods/odin/VERIFICATION.md` §7,
`methods/energy_ebo/VERIFICATION.md` §6, `methods/mahalanobis_mds/VERIFICATION.md` §8,
`methods/dfm/VERIFICATION.md` §7, `methods/srs/VERIFICATION.md` §6,
`methods/react/VERIFICATION.md` §6, `methods/dice/VERIFICATION.md` §7 and
`methods/scale/VERIFICATION.md` §8 remains unresolved. `gradnorm_enh` covers 21 univariate datasets.

## 8. Conclusion

`gradnorm_enh` faithfully reproduces GradNorm: the KL-to-uniform objective (implemented, as in the
official code, via a ones-target cross-entropy that differs from `D_KL` only by the positive factor
`K` — verified exactly), the gradient with respect to the last classification layer's weights, the L1
norm, T=1, and a metric-invariant negation for the project's orientation convention. Its detached-feature
formulation was proved to give bitwise-identical head-weight gradients to the official full-graph
backward. The base `gradnorm.py` is wrong in variable, loss, norm and orientation simultaneously, and
misattributes its orientation to the paper. The correction is also the one that most clearly improves
real-data performance among the four `_enh` variants.
