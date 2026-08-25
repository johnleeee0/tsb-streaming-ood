# DICE Faithfulness Verification — FAITHFUL (fixed 2026-08-21; static ID-mean weight mask + energy; see FIX APPLIED)

**Method id:** `dice` / `dice_enh` · **Paper:** Sun & Li, *DICE: Leveraging Sparsification for
Out-of-Distribution Detection*, ECCV 2022 (arXiv:2111.09805)
**Benchmark variant:** `methods/dice/dice_enh/dice_enh.py` (`DICEEnhDetector`, `top_k=20`)
**Base variant:** `benchmark1/models/ood_methods/dice.py` (`DICEDetector`)
**Verified:** 2026-08-20

---

## Verdict

**NOT-THE-METHOD — for both variants.**

DICE's defining mechanism is *directed sparsification of the classifier weights*: a contribution
matrix is computed **once** from in-distribution statistics (the mean ID feature vector), thresholded
at a percentile to produce a **static binary mask**, and that mask is multiplied into the FC **weights**.
The sparsified layer is then used for every input. The paper's abstract states it plainly:
DICE *"rank[s] **weights** based on a measure of contribution, and selectively use[s] the most salient
weights to derive the output for OOD detection."*

**Neither variant does this.** Both compute the contribution per *test sample* from that sample's own
features and select top-k entries per class on the fly. The weights are never sparsified; there is no
precomputed mask; no ID statistics are used at all (neither variant even implements `fit()`).

`dice_enh` fixes the two defects its `CHANGES.md` identifies — it retains the **sign** and uses
**energy** — but it does not restore the mechanism, and it introduces a third divergence its own notes
describe without recognising as one: selection by **absolute magnitude**, which admits exactly the
large-negative contributions the official mask is constructed to exclude.

Measured consequence (§4): the official static mask and `dice_enh`'s per-sample selection produce
energy scores with a Spearman correlation of only **0.43**, and **49.2%** of `dice_enh`'s selected
entries are negative contributions where the official mask keeps **0**.

**The benchmark used `dice_enh`** — 40 datasets, versus 4 for the base variant.

---

## 1. Source accessibility

| Source | Status |
|---|---|
| Local clone `methods/dice/reference/` | Present, provenance verified: `origin = https://github.com/deeplearning-wisc/dice`, commit `8acfb8d52eee997a7bf6f9d17ff5363fd91ba47b` (2022-09-28). Contains `models/route.py`, `precompute.py`, `ood_eval.py`, `util/score.py`, plus the cached `*_feat_stat.npy` files. |
| `https://raw.githubusercontent.com/deeplearning-wisc/dice/master/models/route.py` | **Live re-fetch FAILED** (`socket hang up`). The `main` branch variant returns 404 — the branch is `master`, so this was a transient network failure, not a wrong URL. The code comparison therefore rests on the local clone, whose provenance and commit are verified above. |
| `https://arxiv.org/abs/2111.09805` (abstract) | Fetched live. |
| Paper full text (ar5iv, `arxiv.org/pdf/...v3`) | **UNREACHABLE** after three attempts (`socket hang up`, then 404). The formal definition of the contribution matrix `V` could not be quoted from the paper; the mechanism below is established from the official code and the abstract only. Stated rather than inferred. |

### The official mechanism

**Step 1 — precompute ID statistics** (`reference/precompute.py:88`, `:129`):

```python
np.save(f"cache/{args.dataset}_densenet_feat_stat.npy", feat_log.mean(0))
```

`feat_log` holds penultimate features for the **entire ID training set** (`id_train_size = 50000` for
CIFAR, `1281167` for ImageNet). The saved artefact is the **mean ID feature vector**.

**Step 2 — build the mask and sparsify the weights** (`reference/models/route.py:16-24`):

```python
def calculate_mask_weight(self):
    self.contrib = self.info[None, :] * self.weight.data.cpu().numpy()
    # self.contrib = np.abs(self.contrib)
    self.thresh = np.percentile(self.contrib, self.p)
    mask = torch.Tensor((self.contrib > self.thresh))
    self.masked_w = (self.weight.squeeze().cpu() * mask).cuda()
```

Four decisive properties:

1. `self.info` is the **ID-mean feature vector**, so `contrib` is a single `(C, D)` matrix computed
   **once** — it is input-independent.
2. `np.abs` is **commented out** (`:18`). The authors considered ranking by magnitude and
   deliberately disabled it. Ranking is on the **signed** contribution.
3. The threshold is `np.percentile(self.contrib, p)` over the **whole `(C, D)` matrix** — one global
   threshold, so different classes retain **different numbers** of units.
4. The selection `contrib > thresh` is **one-sided**: with `p = 90` the threshold is positive, so every
   negative contribution is masked out.

Default `p = 90` (`route.py:8`).

**Step 3 — forward through the sparsified layer** (`route.py:26-34`):

```python
def forward(self, input):
    if self.masked_w is None:
        self.calculate_mask_weight()
    vote = input[:, None, :] * self.masked_w.cuda()
    out = vote.sum(2) + self.bias
```

**Wiring:** DICE *replaces the FC layer* — `self.fc = RouteDICE(in_planes, num_classes, p=p, info=info)`
(`models/densenet.py:113`, `models/resnet.py:194`), with `info = np.load(f"cache/{in_dataset}_{model_arch}_feat_stat.npy")`
(`ood_eval.py:119`, `:126`) and `--p` as the sparsity level (`ood_eval.py:26`).

**Base score** (`reference/util/score.py:93`): `scores = torch.logsumexp(nnOutputs, dim=1).numpy()` —
energy, not negated (higher = ID).

## 2. Divergence table

| Component | Official | `dice_enh.py:line` | `dice.py:line` (base) | Changes results? |
|---|---|---|---|---|
| **Contribution source** | **ID-mean feature vector**, precomputed over the full training set (`precompute.py:88`; `route.py:17`) | **per-test-sample features** (`:41`) | **per-test-sample features** (`:33`) | **YES — defining mechanism absent in both** |
| **What is sparsified** | the FC **weights** → static `masked_w` (`route.py:24`) | nothing; logits recomputed per sample (`:44`) | nothing (`:35`) | **YES** |
| **Ranking criterion** | **signed** `contrib > thresh`; `np.abs` explicitly commented out (`route.py:18,23`) | `contrib.abs().topk(k)` (`:42`) | `contrib.abs().topk(k)` (`:34`) | **YES — admits negatives the official excludes** |
| **Selection rule** | global percentile `p=90` over the whole `(C,D)` matrix → **unequal** units per class (`route.py:22`) | fixed `k=20` **per class** (`:40,42`) | fixed `k=20` per class (`:34`) | **YES** |
| **Value summed** | signed masked weights × input (`route.py:29-31`) | **signed** gathered values (`:43-44`) | **absolute** values (`:34-35`) | base: **YES**; `_enh`: correct on this axis |
| Bias | `+ self.bias` (`route.py:31`) | `+ bias` (`:44`) | `+ bias` (`:35`) | No |
| Base score | energy `+logsumexp` (`util/score.py:93`) | energy `−logsumexp` (`:45`) | **MSP** `1 − softmax_max` (`:36`) | `_enh`: sign flip only, rank-invariant. base: **YES** |
| Orientation | higher = ID | higher = OOD (`:45`) | higher = OOD (`:36`) | No — project convention |
| `fit()` on ID data | required (precompute stage) | **not implemented** | **not implemented** | **YES — no ID statistics used** |
| Temperature | none | exposed, default 1.0 (`:28`) | n/a | No at default |
| Sparsity parameter | `p = 90` (percentile) | `top_k = 20` (count) | `top_k = 20` | Different parameterisation |

## 3. What `CHANGES.md` gets right, and what it misses

`dice_enh/CHANGES.md` identifies two inconsistencies and fixes both:

| Claim | Assessment |
|---|---|
| "Sign dropped … DICE sums the **signed** contributions" | **Correct**, and well evidenced — `route.py:18` shows the authors explicitly disabled the absolute-value variant. `dice_enh:43` restores signed summation. |
| "Score family … DICE uses the **energy** score" | **Correct** — `util/score.py:93`. `dice_enh:45` uses energy. |

But `CHANGES.md` then states the correction *"selects the top-k units by contribution magnitude"* —
describing, without flagging, the third and most consequential divergence. Two things follow:

- Selecting by **magnitude** contradicts the official rule in the very line the notes rely on for the
  sign argument. `route.py` ranks signed contribution *and* thresholds one-sidedly; magnitude ranking
  keeps large-negative units the official discards.
- Neither the notes nor either implementation mentions the **precomputed ID-mean mask**, which is the
  method. `CHANGES.md` should not claim `dice_enh` "matches the paper".

## 4. Mechanism demonstration

Controlled comparison (D=64, C=4, N=400, `p=90`, `k=20`, identical weights and ID features; the
official rule versus `dice_enh`'s rule applied to the same data):

| Property | Official static mask | `dice_enh` per-sample abs-topk |
|---|---|---|
| Weight entries kept | 26 / 256 (**10.2%**) | 20 per class (**31.3%**) |
| Units per class | **unequal**: [8, 7, 4, 7] | **always 20**, equal by construction |
| Negative contributions retained | **0** | **49.2% of all selections** |
| Input-dependent? | **No** — one mask for all inputs | **Yes** — recomputed per sample |
| Spearman correlation of resulting energy scores | — | **0.4302** |

A Spearman of 0.43 between the two scores on identical inputs settles it: these are different
algorithms, not two parameterisations of one. Notably, the official rule keeps roughly a third as many
weights as `dice_enh` selects, and none of the negative ones.

## 5. Measured results

`dice_enh` (the benchmark variant):

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 40 | 0.2857 | — |
| TSB-U (univariate) | 21 | **0.2583** | 16/21 |

Paired against the base variant on the **4** shared datasets:

| Dataset | `dice_enh` | `dice` | Δ |
|---|---|---|---|
| TSB-M-DRIFT003 | 0.576 | 0.554 | +0.022 |
| TSB-U-DRIFT024 | 0.620 | 0.620 | −0.000 |
| TSB-U-OOD009 | 0.000 | 0.000 | +0.000 |
| TSB-U-STABLE001 | 0.837 | 0.857 | −0.020 |
| **mean** | **0.5083** | **0.5080** | **+0.0004** |

`dice_enh` is better on only **1 of 4**.

**The correction's claimed benefit is not observable on real data.** `CHANGES.md` reports a synthetic
improvement (0.826 → 0.961) and states *"The decisive comparison is the Phase 2 sweep on real data."*
That sweep gives Δ = **+0.0004** across the 4 shared datasets — indistinguishable. The synthetic gain
did not transfer, and the base variant's coverage (4 of 40) is too thin to conclude anything either
way. The claim should be withdrawn or the base variant run across all 40 datasets.

Structural checks: `dice_enh` scores span `[−1.042e6, −1.342]`, 100% negative, with the maximum just
above `−log 4 = −1.386` — consistent with `−logsumexp` and K=4. Base `dice` spans `[0.3555, 0.7413]`
⊂ `[0, 0.75]`, consistent with `1 − max softmax` at K=4; the elevated minimum (0.356) is the expected
signature of absolute-value summation inflating all class logits toward similar large values, flattening
the softmax.

The `−1.042e6` extreme is again the same order as the pathological logits recorded in
`methods/energy_ebo/VERIFICATION.md` §3.

## 6. What a faithful implementation requires

The real method is straightforward to add — it needs a `fit()`:

```python
def fit(self, x_id, y_id=None):
    feats = self._forward_features(self._to_tensor(x_id))        # (N, D)
    info = feats.mean(dim=0).cpu().numpy()                       # ID-mean feature vector
    W = self._classifier().weight.detach().cpu().numpy()         # (C, D)
    contrib = info[None, :] * W                                  # signed, input-independent
    thresh = np.percentile(contrib, self.p)                      # global percentile, p=90
    self.mask = torch.from_numpy((contrib > thresh).astype(np.float32))

def score(self, x):
    feats  = self._forward_features(self._to_tensor(x))
    W      = self._classifier().weight * self.mask.to(...)       # sparsified WEIGHTS
    logits = feats @ W.T + bias
    return self._energy(logits).cpu().numpy()                    # higher = OOD
```

Note the parameter changes from `top_k=20` to `p=90` (a percentile), matching `route.py:8`.

## 7. Recommendations

1. **Report both variants as NOT-THE-METHOD** in the fidelity table. Neither implements directed
   sparsification; both are per-sample top-k logit recomputations. `dice_enh` should be described as
   "per-sample top-k contribution + energy", not as DICE.
2. **Correct `CHANGES.md`** — remove the implication that `dice_enh` matches the paper; add the
   precomputed-ID-mask and magnitude-vs-signed-threshold divergences.
3. **Implement the real method** (§6) as `dice_enh2` or replace `dice_enh`, and re-run. Until then the
   DICE row in the results is not a DICE result.
4. **Withdraw or substantiate the ablation claim** — Δ = +0.0004 on n=4 does not support "the abs-sum
   was harmful" on real data, whatever the synthetic task showed.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded in
`methods/msp/VERIFICATION.md` §5, `methods/odin/VERIFICATION.md` §7,
`methods/energy_ebo/VERIFICATION.md` §6, `methods/mahalanobis_mds/VERIFICATION.md` §8,
`methods/dfm/VERIFICATION.md` §7, `methods/srs/VERIFICATION.md` §6 and
`methods/react/VERIFICATION.md` §6 remains unresolved. `dice_enh` covers 21 univariate datasets.

## 8. Conclusion

DICE's defining contribution — ranking classifier **weights** by their mean contribution over ID data
and statically sparsifying them — is absent from both implementations. Both compute contributions per
test sample and take a per-class top-k by absolute magnitude, which is input-dependent, retains the
negative contributions the official mask excludes by construction, and correlates only ρ ≈ 0.43 with
the official score on identical inputs. `dice_enh` is a genuine partial correction: it restores signed
summation and the energy base score, both correctly evidenced against the official code. But it is not
DICE, and its measured advantage over the base variant on real data is +0.0004 across four datasets.

---

## FIX APPLIED (2026-08-20)

**Verdict change: NOT-THE-METHOD → FAITHFUL** (for `dice_enh`; base `dice.py` untouched and
remains NOT-THE-METHOD).

### What changed

`methods/dice/dice_enh/dice_enh.py` (`DICEEnhDetector`) was rewritten to implement DICE's defining
mechanism — *directed, static sparsification of the classification-head weights* — instead of the
previous per-sample absolute-value top-k logit recomputation. The class name and the
`BaseOODDetector` interface are unchanged. No other file was edited.

**New `fit(x_id, y_id)`** (auto-called by the runner before `score()`), mirroring the official
`RouteDICE.calculate_mask_weight()` (`reference/models/route.py:16-24`) plus the precompute stage
(`reference/precompute.py:88`):

1. `info = feats.mean(dim=0)` — a single **ID-mean feature vector** over the ID training data
   (input-independent; the precompute stage).
2. `contrib = info[None, :] * W` — the **signed** contribution matrix over the head weights
   `W` (C, D). The official `np.abs` is commented out (`route.py:18`); we rank the signed values.
3. `thresh = np.percentile(contrib, p)` with `p = 90` — one **global** percentile threshold over the
   entire `(C, D)` matrix (so classes keep unequal numbers of units).
4. `mask = contrib > thresh` — one-sided; with `p = 90` the threshold is positive, so all negative
   contributions are masked out (exactly the units the old abs-topk wrongly kept).
5. `masked_w = W * mask` — the **weights** are sparsified once and stored as a static head.

**New `score(x)`** applies the *same* static sparsified head to every input:
`logits = feats @ masked_w.T + bias`, then returns the **energy** score via `self._energy`
(`-logsumexp`, higher = OOD — a rank-invariant sign flip of the paper's `+logsumexp`,
`util/score.py:93`). No per-sample selection occurs.

**Parameterisation.** Sparsity is now the global percentile `p` (default **90**, matching
`route.py:8`), not a per-class count. The runner still passes `{"top_k": 20}`; `top_k` is parsed for
backward compatibility but is **not used** by the faithful path, so no runner change was required.
Fallbacks preserved: no classification head → energy on raw features; `fit()` not run but head
present → lazy one-time mask build from the batch mean (safety only, not the intended path).

### How each prior divergence (§2) is now resolved

| Component | Prior `dice_enh` | Now | Matches official? |
|---|---|---|---|
| Contribution source | per-test-sample features | **ID-mean feature vector** built in `fit()` | ✅ |
| What is sparsified | nothing (logits recomputed per sample) | **head WEIGHTS** → static `masked_w` | ✅ |
| Ranking criterion | `contrib.abs().topk` (magnitude) | **signed** `contrib > thresh` | ✅ |
| Selection rule | fixed `k=20` per class | **global percentile `p=90`** over `(C, D)` | ✅ |
| `fit()` on ID data | not implemented | **implemented** (precompute stage) | ✅ |
| Base score | energy (`-logsumexp`) | energy (`-logsumexp`) | ✅ (sign = project convention) |

### Smoke test (venv `C:\THESIS\.venv\Scripts\python.exe`)

Dummy linear backbone (32→D=16) + `nn.Linear(16, 4)` head passed via `config['classifier']`;
`fit()` on N=200 random ID samples, `score()` on B=37 random inputs. Result:

```
masked_w shape: (4, 16)  (expect (4, 16))
fraction of head weights zeroed by mask: 0.891  (p=90 -> ~0.90)
score shape: (37,)  (expect (37,))
all finite: True
score range: [-1.6609, -1.3228]
SMOKE TEST PASSED
```

Assertions verified: sparsified head built in `fit()`; ~90% of weights zeroed (one-sided p=90 mask);
output finite and length-correct; `score()` deterministic; single-sample score equals its value
inside a batch (**confirming the mask is static / input-independent** — the property the previous
implementation lacked); no-head fallback stays finite. The full benchmark was not run.

### Faithfulness confirmation vs Sun & Li (2022)

The implementation now reproduces all four decisive properties of the official `RouteDICE`
(§1): (1) contribution from the precomputed ID-mean feature vector, so the mask is computed once and
is input-independent; (2) signed ranking with `np.abs` disabled; (3) a single global percentile
threshold over the whole `(C, D)` matrix, yielding unequal units per class; (4) a one-sided
`contrib > thresh` selection that discards negative contributions — followed by energy scoring on
the statically sparsified weights. **Verdict: FAITHFUL** to DICE (ECCV 2022) within the frozen-
backbone / linear-head / per-sample post-hoc protocol. Re-run the benchmark to regenerate the
`dice_enh` results, which will now be genuine DICE numbers rather than a per-sample top-k artefact.
