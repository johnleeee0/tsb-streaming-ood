# Outlier Exposure Faithfulness Verification — NOT-THE-METHOD (it is an Energy baseline; no training of any kind occurs)

**Method id:** `outlier_exposure` / `oe` · **Paper:** Hendrycks, Mazeika & Dietterich, *Deep Anomaly
Detection with Outlier Exposure*, ICLR 2019 (arXiv:1812.04606)
**Official code:** `https://github.com/hendrycks/outlier-exposure`
**Implementation:** `benchmark1/models/ood_methods/outlier_exposure.py` (`OutlierExposureDetector`,
run with `score_type="energy"`)
**Verified:** 2026-08-20

---

## Verdict

**NOT-THE-METHOD.** Outlier Exposure's defining mechanism is **fine-tuning the classifier against an
auxiliary outlier dataset** with a cross-entropy-to-uniform penalty. This implementation performs **no
training whatsoever** — it computes an energy score on the shared ID-trained backbone.

A static scan of the source confirms the absence is total:

| Token | Present in `outlier_exposure.py`? |
|---|---|
| `def fit` | **No** — inherits `BaseOODDetector.fit`, whose body is `return None` (`base_ood.py:79-80`) |
| `backward()` | **No** |
| `optimizer` / `Adam` / `SGD` | **No** |
| `train()` | **No** |
| `requires_grad` | **No** |
| `ood_loader` / auxiliary outlier data | **No** |

The only occurrence of "outlier" in the file is the class and registry name. There is no auxiliary
dataset, no loss, no gradient step, and no parameter update.

**What it actually computes** (`:19-25`): `−logsumexp(logits / T)` when `score_type="energy"` (the
default, `:17`), or `1 − max softmax` when `score_type="msp"`. Verified from the saved scores: 100%
negative, ceiling at **−1.3297** against `−log 4 = −1.3863` — the signature of `−logsumexp` with K=4
(§3C).

**Two further points that matter for the thesis:**

1. **There is no docstring.** Unlike every Tier C adaptation reviewed in this audit, this file carries
   **no module or class docstring** — no "(Adapted)" note, no statement that training is omitted, no
   pointer to the paper. It is registered under the paper's name (`@register_ood("outlier_exposure")`,
   `@register_ood("oe")`, `:11-12`) and nothing in the source tells a reader it is not OE. This is the
   weakest labelling of any method in the audit.
2. **These runs are also the EBO results.** `energy_ebo` and `energy` have **0** result directories;
   `outlier_exposure` has 40 loadable. The figures here (ALL 0.2949 / TSB-U 0.2770) are exactly those
   reported in `methods/energy_ebo/VERIFICATION.md` §5. **The same 40 score files are being reported
   under two method names.** Presenting both as independent rows double-counts one detector.

---

## 1. Source accessibility

| Source | Status |
|---|---|
| `methods/outlier_exposure/reference/` | Present and intact: `origin = https://github.com/hendrycks/outlier-exposure`, commit `e6ede98a5474a0620d9befa50b38eaf584df4401` (2021-10-08), `HEAD → refs/heads/master`. Contains `CIFAR/oe_tune.py`, `oe_scratch.py`, `baseline.py`, `test.py`. |
| `raw.githubusercontent.com/.../CIFAR/oe_tune.py` | Fetched live; **matches the local clone verbatim**. |
| Paper (arXiv:1812.04606) | **Not fetched.** The mechanism question is settled unambiguously by the official training code, which is the right authority for an implementation diff. Stated rather than implied. |
| Local corroboration | `methods/divoe/reference/src/train.py:193` contains the same OE loss line — an independent second copy in this repo. |

### The official OE training loss

`reference/CIFAR/oe_tune.py:153-182`, confirmed by live fetch:

```python
def train():
    net.train()  # enter train mode
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in zip(train_loader_in, train_loader_out):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]
        x = net(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(x[:len(in_set[0])], target)
        loss += 0.5 * -(x[len(in_set[0]):].mean(1) -
                        torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
        loss.backward()
        optimizer.step()
```

- `train_loader_out` supplies an **auxiliary outlier set** (80 Million TinyImages), `--oe_batch_size`
  default **256** (`:37`, `:95`).
- `:172` — standard cross-entropy on the ID half of the batch.
- `:174` — the OE penalty on the **outlier half**, weighted **0.5**.
- `:176-177` — `loss.backward()`, `optimizer.step()`: **the network weights are updated.**

**The OE penalty is exactly cross-entropy to the uniform distribution.** Verified numerically over 2000
random logit vectors (K=4): `max |−(mean(x) − logsumexp(x)) − CE(uniform, softmax(x))| = 9.54e-07`
(float32 noise). Algebraically,
`−(1/K)Σₖ log pₖ = −(1/K)Σₖ[xₖ − logsumexp(x)] = −(mean(x) − logsumexp(x))`.

## 2. Divergence table

| Component | Official | Mine (`outlier_exposure.py:line`) | Changes results? |
|---|---|---|---|
| **Auxiliary outlier dataset** | 80M TinyImages, `train_loader_out` (`oe_tune.py:95`) | **absent** | **YES — defining mechanism** |
| **OE loss (CE to uniform)** | `0.5 * -(mean − logsumexp)` on the outlier half (`:174`) | **absent** | **YES** |
| **ID cross-entropy term** | `F.cross_entropy(x[:len(in)], target)` (`:172`) | **absent** | **YES** |
| **Parameter updates** | `net.train()`, `loss.backward()`, `optimizer.step()` (`:154`, `:176-177`) | **none** — `fit()` inherited as `return None` | **YES** |
| **What is scored** | the *fine-tuned* classifier | the shared **ID-only** backbone (`:20`) | **YES** |
| Score function | `test.py` offers MSP / energy on the tuned net | `−logsumexp(logits/T)` (`:24`), or `1 − max softmax` if `score_type="msp"` (`:22`) | The score family is a config flag, not the method |
| Temperature | n/a for the loss | exposed, default 1.0 (`:16`) | Inert at default |
| Orientation | higher = more anomalous | higher = OOD (`:24-25`) | **No** |
| Labelling | — | **no docstring at all**; registered as `outlier_exposure` / `oe` (`:11-12`) | See §4 |

## 3. Empirical findings

**(A) Static confirmation** — table in the Verdict above. Nothing in the file trains, and the inherited
`fit` is a no-op.

**(B) The OE penalty identity** — `max |oe_term − CE_uniform| = 9.54e-07`, i.e. exact. This confirms the
paper's description ("cross-entropy to uniform on auxiliary outliers") maps precisely onto
`oe_tune.py:174`, and that no analogue of it exists locally.

**(C) The saved scores are an energy score.**

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All | 40 | **0.2949** | — |
| TSB-U (univariate) | 21 | **0.2770** | 15/21 |

Global range `[−1.75e+06, −1.3297]`, **100% negative**, with the maximum just above `−log 4 = −1.3863`
— the structural signature of `−logsumexp` at K=4. The `−1.75e6` extreme is the
pathological-feature-magnitude family recorded in `methods/energy_ebo/VERIFICATION.md` §3.

**(D) Duplicate reporting.** `energy_ebo`: **0** result dirs. `energy`: **0** result dirs.
`outlier_exposure`: **40**. So the EBO row and the Outlier Exposure row in the results table are backed
by the *same* score files, and the numbers match `methods/energy_ebo/VERIFICATION.md` §5 exactly
(0.2949 / 0.2770).

## 4. Labelling

`methods/energy_ebo/VERIFICATION.md` §0 already states that *"EBO is evaluated under the
`outlier_exposure` label (which, per the Phase 0 audit, performs no outlier training and is therefore
exactly the energy baseline)"* — so the situation **is** documented, but in the *other* method's report,
not here. Within `outlier_exposure.py` itself there is nothing: no docstring, no comment, no note that
the paper's mechanism is omitted.

That matters because the registry name is what reaches the results table. A reader of the thesis seeing
a row labelled "Outlier Exposure" would reasonably assume auxiliary-outlier training took place.

## 5. Recommendations

1. **Rename the registry entry to `energy`** (or `energy_ebo`) and remove the `outlier_exposure` / `oe`
   aliases, or keep the aliases but report the row as **"Energy (EBO)"**. The detector is the energy
   score; that is a legitimate, well-cited baseline — it simply is not Outlier Exposure.
2. **Report one row, not two.** EBO and Outlier Exposure are currently the same 40 score files under two
   names. Collapse them, and state the EBO result once.
3. **Add a docstring** to `outlier_exposure.py` recording that OE's auxiliary-outlier fine-tuning is
   not implemented and that the class computes an energy (or MSP) score on the ID backbone. This is the
   only method in the audit with no documentation at all.
4. **If OE proper is wanted**, it requires an auxiliary outlier corpus and a fine-tuning stage
   (`oe_tune.py:172-177`) — a training change, not a scoring change. That is out of scope for a
   post-hoc benchmark, which is itself a defensible reason to drop the method rather than mislabel it.
5. **Note in the thesis** that OE was evaluated as an energy baseline, so no conclusion about
   auxiliary-outlier training can be drawn from these numbers.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded
across `methods/*/VERIFICATION.md`. `outlier_exposure` covers **21** univariate datasets; see
`methods/diversify/VERIFICATION.md` §4 and `methods/catsight/VERIFICATION.md` §4 for the candidate
explanation of the "18" figure.

## 6. Conclusion

Outlier Exposure is defined by what it does at training time: concatenate an auxiliary outlier batch to
each ID batch and add a 0.5-weighted cross-entropy-to-uniform penalty on the outlier half, backpropagated
into the classifier (`oe_tune.py:172-177`, verified against the live repository and a second local copy
in the DivOE reference). None of that exists here. `outlier_exposure.py` contains no `fit`, no
optimizer, no backward pass, no auxiliary data, and no docstring; it returns `−logsumexp(logits/T)` on
the shared ID-trained backbone, which the saved scores confirm (100% negative, ceiling at `−log 4`).
The defining mechanism is therefore absent rather than simplified, and the runs additionally serve as
the benchmark's EBO results — so the two rows are one detector. It should be reported as an Energy
baseline under a name that says so.

## CLASS-D BUILD (2026-08-21)

**Faithful OE-family build — appendix study, both arms.** The gap identified above
(no `fit`, no auxiliary data, no OE loss) is now closed in a *separate* Class-D
appendix build that never touches production `ood_methods` or the 17-method
leaderboard.

- **Module:** `methods/outlier_exposure/classd/outlier_exposure_classd.py`
  (`OutlierExposureClassD`) — a thin ENERGY scorer over an OE-fine-tuned
  `(backbone, head)`; `score(x) = −logsumexp(head(model(x))/T)`, higher = OOD.
- **Fine-tuning:** driven by `experiments/run_class_d.py::finetune()` with the
  faithful OE objective `L = CE(id) + 0.5·CE_to_uniform(aux)` (λ=0.5, 10 epochs,
  lr=1e-3, bs=64), matching `reference/CIFAR/oe_tune.py:172-177`.
- **Aux corpus:** REAL hold-out TSB windows, channel-matched, drawn via
  `benchmark1/datasets/aux_outliers.py::get_aux_windows` from files disjoint from the
  eval set (persisted `aux_manifest.json` — no leakage). Synthetic fallback only if
  no channel-matched hold-out file exists.
- **Both arms** are run and reported per `CLASS_D_DECISIONS.md` §3:
  `head_only` (ResNet frozen) and `full_net` (paper-faithful). Each arm deep-copies
  the shared backbone before any update — the frozen anchor the 17 methods use is
  never mutated (verified: head_only vs full_net produce different energy scores on
  every eval file).
- **Verification (U split, 2 files/cell, seed 42):** all 12 (arm × file) OE runs
  produced FINITE per-sample AUROCs → `results/class_d_group1.csv`. Example rows:
  OOD_001 head_only 0.1504 / full_net 0.6256; OOD_002 head_only 0.8343 /
  full_net 0.7041.
- **Caveats:** BREAKS the frozen-backbone fair comparison (updates weights the 17
  do not have) → appendix only, never a leaderboard row; the clean Energy (EBO) row
  stays in the main table. Tiny ID/test sets (down to 2 windows on some STABLE
  files) make individual AUROCs high-variance; report aggregated over the full
  eval partition. On the smallest STABLE files fine-tuning can overfit — early
  stopping and few epochs mitigate but do not eliminate this.
