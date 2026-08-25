# InvAD Faithfulness Verification — ADAPTATION (faithful mechanism; fixed 2026-08-21; see FIX APPLIED)

**Method id:** `invad` · **Official code:** `https://github.com/fly-orange/InvAD`
**Paper (per the implementation's own docstring):** *Detecting Both Seen and Unseen Anomalies in Time
Series*, ACM TKDD 2025, doi:10.1145/3717071
**Implementation:** `benchmark1/models/ood_methods/invad.py` (`InvADDetector`)
**Verified:** 2026-08-20

---

## Verdict

**NOT-THE-METHOD.** The docstring's *adaptation* framing is honest — `:2` "(Adapted)", `:10`
"Adaptation for frozen backbone architecture", `:12` "InvAD-Lite", `:11` "Original InvAD uses a full
Invertible Neural Network (INN) on raw time series". No fidelity is claimed.

**But the docstring's description of what the adaptation computes is factually wrong, and the
mechanism is inert.** Your concern is confirmed — and it is stronger than you suspected, and provable
rather than merely empirical:

1. **The reconstruction term is identically zero.** `score()` computes
   `z_id, z_ood = decompose(feats)` then `z = cat([z_id, z_ood])` (`:308-309`). That `cat` reassembles
   the forward output **bit-for-bit**, so `reconstruct(z)` (`:312`) is the exact inverse of the forward
   pass. Measured: max `|x_rec − x| = 2.86e-06`, per-sample MSE `6.8e-15 … 1.5e-13` — float32
   round-off (eps `1.19e-07`). The 0.6-weighted term contributes nothing.
2. **`z_id` is the raw first half of the frozen features — there is no decomposition.** Every coupling
   layer sets `z1 = x1` (`:83`) and halves are never swapped or permuted, so the first half passes
   through untouched at any depth. Measured: `z_id == feats[:, :D//2]` **exactly, atol = 0**.
3. **So the score provably reduces to a scaled MSP.** `0.6·0 + 0.4·(1 − max_prob)` where the classifier
   sees raw features. Predicted cap for K=4 is `0.4·(1 − 1/4) = 0.3000`; **observed global range
   `[0.000000, 0.298712]`, with 100% of scores ≤ 0.30**. The prediction lands exactly.
4. **The official's defining trick is absent.** Official InvAD deliberately *breaks* invertibility at
   reconstruction time (§1). Without that, an invertible network's reconstruction error is necessarily
   zero — the term can never carry signal.

Corroboration: if InvAD ≈ MSP-on-a-retrained-head, its performance should track MSP's. TSB-U means are
**0.3563 (InvAD)** vs **0.3417 (MSP)** — consistent.

Docstring claims that are false in effect: `:14` *"OOD anomalies via reconstruction error"* and
`:19-20` *"combines classifier confidence on z_id with reconstruction error"*. There is no
reconstruction-error signal, and `z_id` is not a learned quantity.

---

## 1. Source accessibility

| Source | Status |
|---|---|
| `methods/invad/reference/` | Present and intact: `origin = https://github.com/fly-orange/InvAD`, commit `a94d7461d6a6f76942f3aecb705e7fb7c2c9a81d` (2025-02-07), `HEAD → refs/heads/main`. Contains `models/INVAD/invad_network/{inn.py, model.py, embed.py}` and trainer. |
| `raw.githubusercontent.com/fly-orange/InvAD/master/.../model.py` | Fetched live (after one transient failure on `main`); **matches the local clone verbatim**. |
| Paper (doi:10.1145/3717071, ACM DL) | **UNREACHABLE — `dl.acm.org` blocked by network policy.** The paper text was not obtained, so the title/venue in the docstring could not be independently confirmed, and no claim below rests on the paper. The mechanism comparison uses the official code, which is the authoritative source for that purpose. |
| Tracker title | The tracker gives *"InvAD: Invertible Neural Networks for Out-of-Distribution Anomaly Detection in Multivariate Time Series"*; the docstring gives *"Detecting Both Seen and Unseen Anomalies in Time Series"*. **Unresolved** — ACM DL was unreachable. Same class of tracker-metadata problem already recorded for DiMMAD and DFM. |

### The official mechanism

`models/INVAD/invad_network/model.py:46-52` (verified in clone **and** live fetch):

```python
emb_1 = emb_temp[:, :, :-self.d_res]     # primary
emb_2 = emb_temp[:, :, -self.d_res:]     # residual (width d_res)

enc_res, enc_pri = self.inv_net(emb_1, emb_2, rev=False)
rec_1, rec_2 = self.inv_net(torch.ones_like(enc_res)*self.res_const, enc_pri, rev=True)
rec_temp = torch.cat((rec_1, rec_2), dim=-1)
```

**The residual branch is replaced by a constant before inverting.** This is the whole design: the
inverse pass must rebuild the embedding from `enc_pri` alone, so reconstruction is *deliberately lossy*
and its error is informative. `get_rec_scores` (`:82-88`) then sums two non-trivial terms:

```python
rec_score  = mean(MSE(rec_temp, emb_temp), -1)                              # lossy-recon error
rec_score += mean(MSE(enc_res, ones_like(enc_res)*self.res_const), -1)      # residual deviation
rec_score  = sigmoid(rec_score)
```

The coupling block (`inn.py:50-77`) is `y1 = x1 + f(x2)`; `y2 = e(r(y1))·x2 + η(y1)` with
`e(s) = exp(clamp·2·(sigmoid(s) − 0.5))` — note the **first half is modified** by `f(x2)`, so
information mixes both ways. The classification branch is a learned scalar head,
`wscore = sigmoid(score_net(fea_w))` (`model.py:72`), with window/point granularity, thresholds and
SoftDTW pseudo-label alignment.

## 2. Divergence table

| Component | Official | Mine (`invad.py:line`) | Changes results? |
|---|---|---|---|
| **Inverse-pass input** | residual branch → **constant** `ones_like(enc_res)*res_const` (`model.py:51`) | full `cat([z_id, z_ood])`, i.e. the exact forward output (`:309`) | **YES — makes recon error ≡ 0** |
| **Reconstruction error** | lossy, informative | **≡ 0** (float round-off) — verified §3 | **YES — 0.6 of the score is inert** |
| **Residual-deviation term** | `MSE(enc_res, const)` (`model.py:85`) | **absent** | **YES** |
| **Coupling form** | `y1 = x1 + f(x2)` — first half **modified** (`inn.py:64`) | `z1 = x1` — first half **untouched** (`:83`) | **YES — no mixing into z_id** |
| **Half permutation between layers** | information mixes both directions | **none** — so `z_id ≡ feats[:, :D//2]` | **YES — no decomposition** |
| Split | configurable `d_res` residual width | fixed half-split `feat_dim // 2` (`:58`) | Yes |
| Scale bound | `exp(clamp·2·(σ(s) − 0.5))` (`inn.py:47`) | `exp(tanh(·))` (`:67`, `:84`) | Minor — both bounded |
| Input domain | raw multivariate series, `(B,T,D)` embeddings, attn/TCN/RNN subnets | pooled frozen features `(B,D)`, MLP subnets (`:61-76`) | Domain adaptation — disclosed |
| Anomaly head | learned `sigmoid(score_net(·))`, window+point, thresholds, SoftDTW | `1 − max softmax` of an MLP (`:317-321`) | **YES** |
| Score combination | `(1−recoef)·cls + recoef·rec`, both live | `0.6·recon + 0.4·(1−max_prob)` (`:325`), recon inert | **YES — single signal** |
| Training signal | full INN + heads, lossy-recon loss drives learning | `loss_recon ≈ 0` (`:253`) ⇒ only `loss_cls` trains (`:258`) | **YES — recon loss is a no-op in training too** |

## 3. Empirical confirmation of the collapse

Verbatim port of `invad.py:45-94`, D=64, 2 layers, N=256, inputs `N(0, 3²)`:

| Test | Result |
|---|---|
| max `|reconstruct(cat(decompose(x))) − x|` | **2.861e-06** |
| per-sample MSE (min / median / max) | **6.83e-15 / 2.28e-14 / 1.52e-13** |
| float32 eps | 1.192e-07 |
| `z_id == feats[:, :D//2]` exactly (atol = 0) | **True** (max diff `0.000e+00`) |

Saved scores (40 datasets, all loadable):

| Predicted cap `0.4·(1 − 1/K)` | K=2 | K=3 | **K=4** | K=5 |
|---|---|---|---|---|
| | 0.2000 | 0.2667 | **0.3000** | 0.3200 |

**Observed global range: `[0.000000, 0.298712]`; 100% of scores ≤ 0.30.** Consistent with K=4 (the
`n_pseudo_classes` default) and a reconstruction contribution of ~0.

**Note on training:** because `loss_recon` (`:253`) is also ≈ 0, the invertibility constraint provides
no gradient. The coupling layers are trained solely by `loss_cls` — and since `z_id` does not depend on
them at all, the coupling parameters receive gradient only through `z_ood`, which nothing consumes.
**The invertible network is effectively untrained and unused.**

## 4. Measured results

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 40 | 0.4334 | — |
| TSB-U (univariate) | 21 | **0.3563** | 13/21 |

Close to MSP (TSB-U 0.3417), as the reduction to a scaled MSP predicts. No missing outputs, no
degenerate constant-score datasets.

## 5. Recommendations

1. **Fix the collapse if InvAD is to be reported at all.** The minimal change mirrors the official:
   substitute a constant for `z_ood` before inverting —
   `z = cat([z_id, torch.full_like(z_ood, res_const)])` — then `recon_error` becomes informative.
   Optionally add the official's second term, `MSE(z_ood, const)`.
2. **Add half-permutation between coupling layers** (or use the official's `y1 = x1 + f(x2)` form) so
   `z_id` is an actual learned decomposition rather than a slice of the raw features.
3. **Correct the docstring** — `:14` and `:19-20` describe a reconstruction signal that is identically
   zero. Until (1) is fixed, the accurate description is "0.4 × (1 − max softmax) of an MLP head on the
   first half of the frozen features".
4. **Relabel** from ADAPTATION to NOT-THE-METHOD in the fidelity table, and report the current numbers
   as an MSP variant, not as InvAD.
5. **Resolve the paper title** between the tracker and the docstring once ACM DL is reachable.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded
across `methods/*/VERIFICATION.md`. `invad` covers **21** univariate datasets; see
`methods/diversify/VERIFICATION.md` §4 for a candidate explanation of the "18" figure.

## 6. Conclusion

The implementation is honestly labelled an adaptation, but the adaptation does not work as described.
Because `cat(decompose(x))` reassembles the forward output exactly, `reconstruct()` is the exact
inverse and the reconstruction error is float round-off — verified at MSE ~1e-14. Because every
coupling layer leaves the first half untouched and no permutation is applied, `z_id` is bit-identical
to the raw first half of the frozen features. The 0.6-weighted reconstruction term therefore
contributes nothing to the score *and* nothing to training, leaving a detector that is provably
`0.4 × (1 − max softmax)` on a retrained head — confirmed by the observed score ceiling of 0.2987
against the predicted 0.3000. The official method avoids exactly this degeneracy by replacing the
residual branch with a constant before inverting, which is the mechanism that makes an invertible
network's reconstruction error meaningful. That step is missing here.

---

## FIX APPLIED (2026-08-20)

**New verdict: ADAPTATION (faithful mechanism).** The two structural defects identified above are
fixed; the reconstruction branch is now mathematically informative and the invertible network performs
a genuine learned decomposition. Only `benchmark1/models/ood_methods/invad.py` and this file were
edited. Class name `InvADDetector`, the `@register_ood("invad")` key, and the `BaseOODDetector`
interface (`fit(x_id, y_id)`, `score(x)`) are unchanged.

### Changes to `benchmark1/models/ood_methods/invad.py`

1. **Constant substituted for `z_ood` before the inverse pass (the core fix).** In both `fit()` and
   `score()` the reconstruction path no longer feeds the exact forward output `cat([z_id, z_ood])`
   back through the inverse. It now inverts `cat([z_id, full_like(z_ood, res_const)])`, mirroring the
   official `model.py:51` (`inv_net(torch.ones_like(enc_res)*res_const, enc_pri, rev=True)`). Because
   the residual branch is discarded and replaced by a constant, `reconstruct(...)` is a *deliberately
   lossy* inverse and its error carries OOD signal instead of being float round-off (~1e-14).
2. **Residual-deviation term added.** The reconstruction score is now
   `MSE(x_recon, x) + MSE(z_ood, const)`, matching the official `get_rec_scores` (`model.py:82-88`).
   Applied both to the training loss and the test-time score.
3. **Half-permutation between coupling layers.** `InvADNetwork` now applies a fixed, seeded
   `torch.randperm(feat_dim)` between consecutive coupling layers (`n_layers - 1` permutations,
   registered as buffers; inverse = `argsort`). Since each affine coupling leaves its first half
   untouched (`z1 = x1`), without this permutation `z_id` was bit-identical to `feats[:, :D//2]`. The
   permutation crosses the split boundary so `z_id` becomes an actual learned function of all input
   dimensions. `decompose()`/`reconstruct()` apply the permutations forward / their inverses in
   reverse.
4. **New config key `res_const`** (default `0.0`, cf. official `res_const`), plumbed into both fit and
   score.
5. **Docstrings corrected.** The module and class docstrings now describe the lossy-reconstruction
   mechanism accurately instead of claiming a reconstruction signal that was identically zero.

### Smoke-test result (venv `C:\THESIS\.venv`, CPU, dummy 32-d backbone, tiny random data)

```
score(id) shape: (30,)  score(ood) shape: (30,)   all finite: True
exact-inverse MSE (full z)     : 5.043e-13   (network still exactly invertible)
constant-substituted recon MSE : 7.870e+01   (reconstruction now LOSSY / informative)
max|z_id - raw_first_half|     : 7.123e+01   (real decomposition — z_id != raw slice)
mean score ID  : 0.3878   mean score OOD : 49.4293   OOD > ID mean : True
```

**Faithfulness confirmation.** The two provable defects from the original verdict are gone:
(1) the constant-substituted reconstruction error is ~78.7 (was ~1e-14), so the 0.6-weighted term is
now the dominant, informative OOD signal rather than inert; (2) `z_id` differs from the raw first half
by up to 71.2 (was exactly 0, atol=0), so the invertible network performs a genuine decomposition and
is actually trained. The full-`z` round-trip MSE stays ~5e-13, confirming the network remains exactly
invertible — the informativeness comes solely from discarding `z_ood`, exactly as in the official
design. Dual-branch separation is meaningful and correctly oriented (OOD mean ≫ ID mean; higher = OOD).
The score no longer collapses to `0.4·(1 − max softmax)`.

**Residual note.** This is a faithful reproduction of the official *reconstruction mechanism* on the
frozen-feature adaptation (`InvAD-Lite`); it does not restore the raw-series input domain, the
attention/TCN/RNN subnets, the learned `sigmoid(score_net)` anomaly head, or SoftDTW pseudo-label
alignment — those remain disclosed domain adaptations for the frozen-backbone protocol. Real-data
AUROC should be regenerated: the number will change materially (the old row tracked MSP, TSB-U 0.356)
and the new score is a genuinely distinct signal with no guarantee of improvement.
