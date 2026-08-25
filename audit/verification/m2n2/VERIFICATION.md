# M2N2 Faithfulness Verification — ADAPTATION (val/test EMA leak fixed 2026-08-21; see FIX APPLIED)

## FIX APPLIED (2026-08-21)
The sequential EMA `trend_mean` was mutated during the val `score()` pass and never reset before the test
pass, so val data leaked into test scoring (`m2n2.py`). Fixed by snapshotting the fit-time trend
(`self._trend_init`) and resetting it at the start of every `score()` call, making each split independent
and reproducible. This is the on-protocol (class-B) fix; full fidelity to the paper's raw-series test-time
adaptation still needs an ordered stream (documented limitation). Remains labelled an ADAPTATION.


**Method id:** `m2n2` · **Official code:** `https://github.com/carrtesy/M2N2`
**Paper (per the implementation's docstring):** *When Model Meets New Normals: Test-time Adaptation for
Unsupervised Time-series Anomaly Detection*, AAAI 2024, doi:10.1609/aaai.v38i12.29210,
arXiv:2312.11976
**Implementation:** `benchmark1/models/ood_methods/m2n2.py` (`M2N2Detector`)
**Verified:** 2026-08-20

---

## Verdict

**ADAPTATION — the label is honest.** The docstring discloses it at `:2` "(Adapted)", `:11`
"Adaptation for frozen backbone architecture", `:12` "M2N2-Lite", and `:12` states plainly that
*"Original M2N2 trains autoencoders on raw time series with test-time adaptation"*. The mechanism
described matches what the code does, and the EMA update formula is **identical** to the official
`Detrender`.

**Two corrections to the tracker's framing:**

1. **"Adaptation OFF by default" is not a divergence — it matches the official.** The official
   `cfgs/test_defaults.yaml` sets `infer_options: ["offline"]` **and** `normalization: "None"`. So
   M2N2's adaptive machinery is opt-in upstream too. Having `adapt_test_time=False` (`:107`) is
   *consistent* with the reference default, not a simplification of it.
2. **`gamma` is 0.995 in the code, not 0.999.** Four different values are claimed across the
   documentation (§4).

**One thing genuinely undisclosed:** the sequential EMA makes scores **order-dependent and
non-idempotent**, and neither the module docstring nor the config documentation says so. Confirmed
empirically at 5.0% mean score deviation under reshuffling (§3).

---

## 1. Source accessibility

| Source | Status |
|---|---|
| `methods/m2n2/reference/` | Present and intact: `origin = https://github.com/carrtesy/M2N2`, commit `616b2270b6f2eab88ee5caa37c45507d2d041d22` (2025-05-29), `HEAD → refs/heads/master`. Contains `models/Normalizer.py`, `Exp/Tester.py`, `Exp/MLP.py`, `cfgs/test_defaults.yaml`. |
| Paper (AAAI / arXiv:2312.11976) | **Not fetched.** The mechanism comparison is done against the official code, which is the authoritative source for an implementation diff. No claim below rests on the paper text. Stated rather than implied. |
| Tracker title | The tracker gives *"M2N2: Memory-based Test-time Adaptation for Multivariate Time-series Anomaly Detection"*; the docstring gives *"When Model Meets New Normals: …"* with a real DOI and arXiv ID. **Unresolved**, though the docstring's version carries verifiable identifiers. Third instance of this tracker-metadata problem, after DiMMAD and InvAD. |

### What the official code does

**Detrender** (`models/Normalizer.py:5-38`):

```python
def __init__(self, num_features: int, gamma=0.99):
    self.mean = nn.Parameter(torch.zeros(1, 1, self.num_features), requires_grad=False)

def _update_statistics(self, x):
    dim2reduce = tuple(range(0, x.ndim-1))
    mu = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
    self.mean.lerp_(mu, 1-self.gamma)

def _normalize(self, x):
    x = x - self.mean
```

`lerp_(mu, 1-gamma)` expands to `mean·gamma + mu·(1-gamma)` — **the same recurrence as `m2n2.py:210`.**
Defaults: `gamma = 0.99` (class) and `gamma: 0.99` (`test_defaults.yaml`); `mean` initialised to
**zeros**.

**Test-time adaptation** (`Exp/MLP.py:173-217`):

```python
self.load_trained_model()                                    # reset before online run
TT_optimizer = torch.optim.SGD(..., lr=self.args.ttlr)       # created ONCE, outside the loop
for i, batch_data in enumerate(it):
    if normalization == "Detrend":
        self.model.normalizer._update_statistics(X)          # CONDITIONAL
    Xhat = self.model(X); E = (Xhat-X)**2; A = E.mean(dim=2)
    ytilde = (A >= tau).float()                              # tau = fixed q95 from training
    ...
    mask = (ytilde == 0)
    recon_loss = (A * mask).mean()                           # MASK, not skip
    recon_loss.backward(); TT_optimizer.step()
```

Config: `ttlr: 1e-03`, `thresholding: q95.0`, `eval_batch_size: 1`, `normalization: "None"`,
`infer_options: ["offline"]`. `Tester.online` is abstract (`Tester.py:208-209`); per-model
implementations override it.

## 2. Divergence table

| Component | Official | Mine (`m2n2.py:line`) | Changes results? |
|---|---|---|---|
| Input domain | raw multivariate series `(B,L,C)` | pooled frozen backbone features `(B,D)` (`:129`, `:199`) | **YES** — disclosed adaptation |
| Model | MLP / USAD / LSTMEncDec / AnomalyTransformer / THOC | fixed 4-layer MLP AE (`:47-80`) | **YES** — disclosed |
| **EMA recurrence** | `mean·γ + mu·(1−γ)` (`Normalizer.py:29`) | `γ·trend + (1−γ)·feat` (`:210`) | **No — exact match** |
| **gamma** | **0.99** (class + yaml) | **0.995** (`:105`) | Yes — see §3(C) |
| Trend init | **zeros** (`Normalizer.py:14`) | **training feature mean** (`:140`) | Yes |
| EMA update source | mean over batch **and time** dims | the single sample's features (`:209`) | Minor — analogue given no time dim |
| **Detrend applied** | **conditional**, `normalization: "None"` by default | **always**, unconditionally (`:213`, and `:155` in training) | **YES — source of the order-dependence** |
| Adaptation default | `infer_options: ["offline"]` → **off** | `adapt_test_time=False` (`:107`) → **off** | **No — matches** |
| Sequential processing | `eval_batch_size: 1` | per-sample loop (`:205`) | **No — matches** |
| Model reset before scoring | `load_trained_model()` (`MLP.py:174`) | **none** — `trend_mean` persists (`:210`) | **YES** — see §3(B) |
| TTA threshold | fixed `tau` = q95 of training scores | **running median of scores so far** (`:227`) | **YES** (inert while off) |
| TTA selection | **mask** `(A * mask).mean()` — always steps | **skip** via `if error < threshold` (`:229`) | **YES** (inert while off) |
| TTA optimizer | SGD created **once** (`MLP.py:184`) | new SGD **per sample** (`:232`) | No for plain SGD; would differ with momentum |
| TTA lr | `ttlr = 1e-3` | `eta = 0.01` (`:106`) | Yes (inert while off) |
| Score granularity | per-timestep `A = E.mean(dim=2)` → `(B,L)` | per-window scalar (`:220`) | **YES** — no point-level scores |
| Orientation | higher recon error = anomalous | higher = OOD (`:243`) | **No** |

## 3. Empirical confirmation

Replicating the exact arithmetic of `m2n2.py:205-221` (fixed random AE, N=200, D=32, mildly drifting
features):

**(A) Order dependence — confirmed, mild.**

| Metric (gamma=0.995) | Value |
|---|---|
| mean \|score(ordered) − score(shuffled)\| | **0.2348** |
| max \|diff\| | 0.7953 |
| **relative mean deviation** | **5.01%** |
| Spearman(ordered, shuffled) | **0.9711** |

So the ranking is largely preserved (ρ = 0.97), which is what matters for AUROC — the effect is real
but mild, exactly as the tracker anticipated.

**(B) Statefulness — confirmed, and not idempotent.** `:210` mutates `self.trend_mean`, which is never
reset. Scoring the same data twice:

| Metric | Value |
|---|---|
| mean \|call1 − call2\| | 0.1729 (**3.69% relative**) |
| Spearman(call1, call2) | 0.9863 |

The official avoids this by calling `load_trained_model()` at the start of each online run
(`MLP.py:174`).

**(C) gamma sensitivity — and the official value is *worse*.**

| gamma | effective window | mean shuffle-diff | Spearman |
|---|---|---|---|
| 0.900 | 10 | 0.8119 | 0.6436 |
| **0.990 (official)** | 100 | **0.3949** | **0.9275** |
| **0.995 (mine)** | 200 | **0.2348** | **0.9711** |
| 0.999 (tracker's claim) | 1000 | 0.0621 | 0.9975 |

Higher gamma ⇒ longer memory ⇒ less order-sensitivity. So 0.995 is *more* order-stable than the
official 0.99; adopting the official value would roughly double the order effect. Worth noting before
"matching the reference" on this parameter.

## 4. The gamma documentation is inconsistent

Four values are claimed for the same parameter:

| Location | Value |
|---|---|
| `m2n2.py:26` (module docstring) | **0.9** |
| `m2n2.py:94` (class docstring) | 0.995 |
| `m2n2.py:105` (code, authoritative) | **0.995** |
| Tracker prompt | 0.999 |
| Official `Normalizer.py` / `test_defaults.yaml` | 0.99 |

The operative value is **0.995**. `:26` is stale and should be corrected; the tracker's 0.999 does not
match the code.

## 5. Measured results

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 40 | 0.7565 | — |
| TSB-U (univariate) | 21 | **0.7953** | 4/21 |

**Second-strongest detector verified so far** — SRS 0.8408, **M2N2 0.7953**, DiMMAD 0.7902,
DIVERSIFY 0.6598 — reinforcing that reconstruction/distance methods on features dominate this
benchmark while logit-space methods invert.

All 40 datasets loadable, none degenerate (0/40 with range < 1e-12). Scores span
`[0.002685, 2.594e11]`, all non-negative as an MSE requires; the `2.6e11` extreme is again the
pathological-feature dataset family recorded in `methods/energy_ebo/VERIFICATION.md` §3.

**Caveat on reproducibility:** because of §3(B), these saved scores depend on the order in which
windows were presented and on whether `score()` had been called before. They are not exactly
reproducible without replaying the same call sequence.

## 6. Recommendations

1. **Disclose the order-dependence.** Add to the docstring that the EMA trend is updated sequentially
   at test time, so scores depend on sample order (~5% deviation, ρ ≈ 0.97) and are not idempotent.
   This is the one substantive gap in an otherwise honest label.
2. **Reset `trend_mean` at the start of `score()`** (or snapshot/restore it), mirroring the official
   `load_trained_model()`. That makes scoring idempotent at no cost while keeping the streaming
   semantics within a call.
3. **Fix `:26`** — it says gamma default 0.9; the code uses 0.995. Also reconcile the tracker's 0.999.
4. **Note that always-detrending diverges from the official default** (`normalization: "None"`).
   Consider a `detrend` flag so the no-detrend configuration is reachable.
5. **If TTA is ever enabled**, align it with the official: fixed q95 threshold rather than a running
   median, masked loss rather than skipping the step, one optimizer outside the loop, and `ttlr = 1e-3`.
6. Keep the "M2N2-Lite / Adapted" labelling — it is accurate.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded
across `methods/*/VERIFICATION.md`. `m2n2` covers **21** univariate datasets; see
`methods/diversify/VERIFICATION.md` §4 for a candidate explanation of the "18" figure.

## 7. Conclusion

M2N2 is honestly labelled an adaptation and the label holds up: the EMA detrend recurrence is
arithmetically identical to the official `Detrender`, sequential per-sample processing matches
`eval_batch_size: 1`, and adaptation being off by default matches the official's own
`infer_options: ["offline"]` — so two of the three points the tracker flags as divergences are in fact
agreements. The genuine divergences are the frozen-feature input domain (disclosed), unconditional
rather than conditional detrending, a trend initialised to the training mean rather than zeros, and a
TTA path that differs in threshold, selection rule and learning rate but is inert at the default. The
sequential EMA does make scores order-dependent — confirmed at 5.0% mean deviation with Spearman 0.97,
plus 3.7% non-idempotency across repeated calls — and that behaviour is **not currently disclosed**,
which is the one thing to fix in the labelling.
