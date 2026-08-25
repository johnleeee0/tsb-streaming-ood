# ODIN Faithfulness Verification — FAITHFUL

**Method id:** `odin` · **Paper:** Liang, Li & Srikant, *Enhancing the Reliability of
Out-of-Distribution Image Detection in Neural Networks*, ICLR 2018 (arXiv:1706.02690)
**Implementation under test:** `benchmark1/models/ood_methods/odin.py` (`ODINDetector`)
**Verified:** 2026-08-19

---

## Verdict

**FAITHFUL.** The defining ODIN mechanism is reproduced correctly: temperature scaling at T=1000,
the confidence-increasing input perturbation `x − ε·sign(∇ₓ CE(logits/T, ŷ))`, and max-softmax
scoring of the perturbed input at temperature T. The orientation flip (`1 −`) is metric-invariant and
the batched gradient is provably equivalent to the official per-sample computation.

One substantive divergence: the per-channel gradient std-normalisation is omitted, which
**reparameterises ε rather than being an identity**. This is a hyperparameter-magnitude deviation,
not a mechanism change — but the prior justification in `ODIN_VALIDATION.md:38` was incorrect and is
corrected in §4 below.

---

## 1. Source accessibility

All sources were reachable. Nothing in this report is guessed.

| Source | Status |
|---|---|
| `https://raw.githubusercontent.com/facebookresearch/odin/main/code/calData.py` | Fetched live; matches the local clone verbatim |
| `https://arxiv.org/abs/1706.02690` (abstract) | Fetched live |
| `https://ar5iv.labs.arxiv.org/html/1706.02690` (full text) | Fetched live |
| Local clone `methods/odin/reference/` | `origin = https://github.com/facebookresearch/odin`, commit `64e97962ccaed1fe979f43a089c0feb4d8b002fd` (2018-02-13) |

**From the paper (full text):**

- Perturbation, Eq. 2: `x̃ = x − ε·sign(−∇ₓ log S_ŷ(x,T))`. Since cross-entropy to the predicted
  label is `CE = −log S_ŷ`, this is identically `x̃ = x − ε·sign(∇ₓ CE)`.
- Temperature: *"We use T=1000 for all settings"* (searched over {1,2,5,10,20,50,100,200,500,1000}).
- ε: *"The noise magnitude ε was selected on a separate validation dataset, which is different from
  the out-of-distribution test sets."* Selected values 0.0014 (CIFAR-10) and 0.002 (CIFAR-100),
  searched over *"21 evenly spaced numbers starting from 0 and ending at 0.004"*.
- Score: `max_i p(x̃, T)`; OOD if score ≤ δ, so **higher = in-distribution**.
- Post-hoc: ODIN *"does not require any change to a pre-trained neural network."*

**From the official code:** temperature `outputs = outputs / temper` (`calData.py:55`); pseudo-label
and loss `maxIndexTemp = np.argmax(nnOutputs)` / `loss = criterion(outputs, labels)` / `loss.backward()`
(`:59-62`); sign `gradient = torch.ge(inputs.grad.data, 0); gradient = (gradient.float() - 0.5) * 2`
(`:65-66`); per-channel division by `63.0/255.0`, `62.1/255.0`, `66.7/255.0` (`:68-70`); perturbation
`tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)` (`:72`); rescoring
`outputs = net1(Variable(tempInputs)); outputs = outputs / temper` then softmax and `np.max`
(`:73-80`). Defaults `magnitude=0.0014`, `temperature=1000` (`main.py:41-44`).

## 2. Divergence table

| Component | Original | Mine (`odin.py:line`) | Changes results? |
|---|---|---|---|
| Features used | Logits of pre-trained classifier | `base_ood.py:98` → `_forward_logits`, classifier head chained at `base_ood.py:48-50` | **No** |
| Temperature | T=1000 (`calData.py:55`, `main.py:43`, paper §3) | `temperature=1000.0` (`:17`), applied at `:23` and `:30` | **No** |
| Gradient objective | CE(T-scaled logits, argmax pseudo-label) (`calData.py:59-61`) | `F.cross_entropy(logits / self.temperature, pred)` where `pred = logits.argmax(dim=-1)` (`:22-23`) | **No** |
| Perturbation direction | `x − ε·sign(∇ₓCE)` (`calData.py:65-66,72`; paper Eq. 2) | `x_tensor - self.epsilon * grad_sign` (`:25-26`) | **No** — correct ODIN direction |
| Score statistic | `max_i p(x̃, T)`, higher = ID (`calData.py:73-80`) | `1.0 - self._softmax_max(logits_pert, temperature=self.temperature)`, higher = OOD (`:29-30`) | **No** — rank-invariant |
| **Per-channel grad ÷ std** | **Present**, ÷ 0.2471 / 0.2435 / 0.2616 (`calData.py:68-70`) | **Omitted** — raw `.sign()` (`:25`) | **Magnitude only** — see §4 |
| ε | 0.0014, tuned per dataset on validation OOD (paper §4) | Fixed `0.001`, untuned (`:18`) | Magnitude only; see §4 |
| Training | Post-hoc, no retraining | `fit()` is a no-op (`base_ood.py:79-80`) | **No** |
| Batch handling | 1 image per iteration (`calData.py:38-42`) | Batched, `reduction='mean'` (`:23`) | **No** — proven in §3 |
| Sign at exactly zero | `ge(g,0)` maps `0 → +1` (`calData.py:65`) | `torch.sign` maps `0 → 0` (`:25`) | **No** — measure zero |

## 3. Two equivalences verified rather than assumed

**Batching is safe.** The official code runs batch-size 1; mine calls `F.cross_entropy` with the
default `reduction='mean'` over a batch. Under `model.eval()` (`base_ood.py:18-24`, so BatchNorm uses
running statistics) samples are independent through the network, therefore
`∂L/∂xᵢ = (1/B)·∂CEᵢ/∂xᵢ`. Because `sign()` is scale-invariant, the resulting perturbation is
**bitwise identical** to the official per-sample computation. This is not a divergence.

**The perturbation is genuinely active.** ODIN scores differ from MSP scores on **39 of 40** datasets
(Pearson correlation 0.13–1.00 across datasets). The single identical case
(`TSB-M-STABLE_015`) is degenerate — every score is exactly 0.

## 4. The per-channel std-normalisation: correcting the prior justification

`ODIN_VALIDATION.md:38` justified the omission as *"negligible on z-normalised TS (per-channel
std ≈ 1)"*. **That reasoning does not hold.**

The official code divides the gradient sign by the **CIFAR-10 raw-pixel** standard deviations
(`63.0/255 = 0.2471`, etc.), *not* by the standard deviation of the normalised tensor — which is also
≈ 1 for CIFAR, yet the code still divides by 0.2471. The purpose of the division is to express ε in
**raw [0,1] pixel units**: a perturbation of ε in raw space equals ε/std in the normalised space the
network actually consumes. Omitting the division therefore does not reduce to the identity; it
**reparameterises ε from raw units into normalised (σ) units**.

Effective perturbation magnitude in the network's input space:

| Setting | Effective magnitude |
|---|---|
| Official, ε = 0.0014 ÷ 0.2471 (ch0) | **0.005667 σ** |
| Official, ε = 0.0014 ÷ 0.2435 (ch1) | **0.005749 σ** |
| Official, ε = 0.0014 ÷ 0.2616 (ch2) | **0.005352 σ** |
| Mine, ε = 0.001 × 1 | **0.001000 σ** |
| **Ratio (official ch0 / mine)** | **5.67× stronger than mine** |

Expressed in the authors' own units, my effective ε corresponds to a raw-units
ε ≈ 0.001 × 0.2471 ≈ **0.00025**. The paper searched ε over *"21 evenly spaced numbers starting from
0 and ending at 0.004"* and selected **0.0014**. My setting therefore sits near the **bottom** of the
authors' search grid (roughly bin 2 of 21), not at their selected value — a weak-perturbation regime,
closer to plain max-softmax-at-T=1000 than to tuned ODIN. This is consistent with the strong but
imperfect ODIN–MSP correlation measured in §3.

**Assessment.** This is a hyperparameter-magnitude deviation, not a mechanism deviation: ODIN's ε is
tuned per dataset in the original, so no single value is canonical, and there is no meaningful "raw
pixel space" analogue for z-normalised time series. The verdict remains FAITHFUL. But the deviation
is **not** negligible, and the note should read *"ε is reparameterised into normalised units; the
effective perturbation is ~5.7× below the official default and near the bottom of the paper's search
grid."* Quantifying the sensitivity requires a rerun at `epsilon = 0.0057`; **that rerun was not
performed, so no claim is made about its effect.**

## 5. Numerical conditioning under T = 1000 (new finding)

Across all 2032 scored windows:

| Observation | Count | Share |
|---|---|---|
| Score within 1e-2 of `0.75 = 1 − 1/K` (K=4) | 1911 | 94.0% |
| Score within 1e-3 of 0.75 | 738 | 36.3% |
| Score < 0.70 (softmax not flattened) | 26 | 1.3% |
| Score exactly 0.0 (max softmax = 1.0 even at T=1000) | 18 | 0.9% |

At T=1000, `logits/1000 ≈ 0`, so the softmax is driven towards uniform and the score concentrates
near `1 − 1/K`. This is **correct and expected** — the official implementation softmaxes the same
T-scaled logits — and AUROC depends only on ranking, so it is not a defect. float32 resolution at
0.75 is 5.96e-08 against a band width of 9.99e-03, i.e. ~168,000× headroom, so there is no precision
loss. It should nonetheless be disclosed that ODIN's entire discriminative signal lives inside a
~1e-2-wide band.

The 18 windows scoring exactly 0.0 require a logit gap of ≳ 17,000 to saturate the softmax at
T=1000. That indicates pathologically large backbone logits on a few datasets — a property of the
backbone, not of ODIN fidelity — and renders those windows unrankable ties.

## 6. Measured results

| Subset | n | mean AUROC |
|---|---|---|
| All datasets — ODIN | 40 | 0.3113 |
| All datasets — MSP (comparison) | 40 | 0.3850 |
| TSB-U (univariate) — ODIN | 21 | 0.2610 |

ODIN is below chance on **16 of 21** univariate datasets and ≤ MSP on **29 of 40** datasets overall.
This is the expected amplification of the MSP overconfidence inversion: the perturbation is designed
to raise the maximum softmax probability more for in-distribution inputs, but on far-off-manifold
streaming windows it raises OOD confidence too, deepening the inversion.

Spot examples (all reproduced exactly from the saved scores):

| Dataset | ODIN | MSP |
|---|---|---|
| TSB-U-DRIFT_051 | 0.000 | 0.001 |
| TSB-U-OOD_035 | 0.000 | 0.562 |
| TSB-U-DRIFT_034 | 0.857 | 0.980 |

## 7. Open items

**Dataset-count and mean discrepancy (unresolved).** `ODIN_VALIDATION.md:48` cites mean AUROC
**0.231** on TSB-U with **15/18** datasets below chance and ODIN ≤ MSP on 15/18. The artifacts show
mean **0.2610** over **21** datasets, **16/21** below chance, and ODIN ≤ MSP on **29/40** overall. The
qualitative conclusion is unchanged, but neither the counts nor the mean reconcile, and the gap here
(0.231 vs 0.261) is wider than the corresponding MSP gap. No config defining an 18-dataset subset was
found. This needs a decision on which subset is canonical before the thesis text is finalised — the
same open question recorded in `methods/msp/VERIFICATION.md` §5.

**ε sensitivity untested.** See §4. A rerun at `epsilon = 0.0057` would close this.

## 8. Conclusion

ODIN is faithfully reproduced. Temperature scaling (T=1000), the gradient objective (cross-entropy to
the predicted pseudo-label on T-scaled logits), the perturbation direction
`x − ε·sign(∇ₓCE)`, and perturbed-input max-softmax scoring all match the official code and the
paper's Eq. 2. The `1 −` orientation is metric-invariant; the batched gradient is provably identical
to the official per-sample gradient. The one substantive divergence is the ε parameterisation
(~5.7× weaker than the official default, near the bottom of the paper's search grid), which is a
hyperparameter choice rather than a change of mechanism.
