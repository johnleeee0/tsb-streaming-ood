# DFM-PCA Faithfulness Verification — FAITHFUL (to the TS-OOD DFM-PCA target; differs from Ahuja 2019 original)

**Method id:** `dfm` / `dfm_pca` · **Cited paper:** Ahuja, Ndiour, Kalyanpur & Tickoo, *Probabilistic
Modeling of Deep Features for Out-of-Distribution and Adversarial Detection*, 2019 (arXiv:1909.11786)
**Actual target:** Gungor, Rios, Ahuja & Rosing, *TS-OOD*, AAAI-25 AI4TS (arXiv:2502.15901) — "DFM-PCA"
**Implementation under test:** `benchmark1/models/ood_methods/dfm_pca.py` (`DFMPcaDetector`)
**Verified:** 2026-08-20

---

## Verdict

**FAITHFUL to the DFM-PCA definition in TS-OOD** — per-ID-class PCA on pre-logit features, scored by
the minimum feature reconstruction error across class models. That is exactly what TS-OOD prescribes
and exactly what the code does.

**ADAPTATION relative to Ahuja et al. (2019).** The 2019 paper's score is **not** a reconstruction
error: it is the **likelihood/NLL under a fitted Gaussian or Gaussian-mixture density**, with PCA used
only as a dimensionality-reduction preprocessing step. The feature-reconstruction-error (FRE) score
originates in the *later* Intel Labs work (Ndiour et al., ICIP 2022 / BMVC 2023), which TS-OOD
re-labels "DFM-PCA". The citation in `discrepancy_report.md` and in the tracker should be supplemented
accordingly — reproducing FRE is correct for this thesis, but it is not the 2019 paper's method.

**One undocumented deviation:** `n_components=32` is a fixed count. It is **not** documented by TS-OOD
(which does not specify a component count) and it differs in *policy* — not merely in value — from both
reference implementations, which hold **explained variance** constant (0.995 in Ahuja et al., 0.97 in
anomalib). See §5.

---

## 1. Source accessibility

| Source | Status |
|---|---|
| `https://github.com/MehrtashHarandi/DFM` (URL in the tracker) | **UNREACHABLE — HTTP 404.** The URL is wrong. |
| `https://intellabs.github.io/dfm` | Reachable. Confirms FRE is the Intel Labs ICIP-2022 / BMVC-2023 method and points to `IntelLabs/dfm` (branches `ICIP2022`, `main`). |
| `https://github.com/IntelLabs/dfm` | **UNREACHABLE — HTTP 404**, both the bare repo URL and the branch links from the project page. No official source code could be read. `gh` CLI is not installed, so no authenticated retry was possible. |
| Local clone `methods/dfm/reference/` | Present, but it is **anomalib**, not the official DFM repo: `origin = https://github.com/openvinotoolkit/anomalib`, commit `9e962ffda2919a86d6b186931d978b345038d69f` (2026-06-21), 1646 files / 51.9 MB. This is the *cross-check* implementation named in the tracker, not the primary reference. |
| Ahuja et al. 2019 full text | Obtained (PDF text extraction, 10 pages / 31,105 chars). |
| TS-OOD full text (arXiv:2502.15901) | Fetched live via ar5iv. |

**Net effect:** no official DFM/FRE source code was readable. The code comparison below therefore rests
on (a) the anomalib implementation in the local clone, and (b) the two papers' text. This is stated
rather than papered over.

### What the sources say

**Ahuja et al. 2019** (extracted verbatim):

- *"modeling the outputs of the various layers (deep features) with parametric probability distributions once training is completed. At inference, the likelihoods of the deep features w.r.t the previously learnt distributions are calculated"*
- *"two classes of multivariate distributions for modeling the deep features — Gaussian and Gaussian mixture"*, with *"Gaussian (with separate covariances for each class)"*
- PCA is preprocessing only: *"we follow a two-fold approach: average pooling of very high-dimensional layers and applying PCA for projecting onto a lower dimensional subspace"*
- Component policy: *"While applying PCA, one can specify the fraction of the variance of the original data that should be retained… We choose a high value of 0.995"*
- GMM component count chosen by **BIC**.

There is **no feature-reconstruction-error score anywhere in the 2019 paper.**

**TS-OOD (arXiv:2502.15901):**

- *"The OOD score is then computed as the 'Feature Reconstruction Error' of a sample projected to the low dimension and then re-projected back via an **ID class'** inverse PCA transform."* → per-class, confirmed.
- *"We experimented with extracting features from several depths of the backbones tested and obtained on average superior results using the pre-logit layer."*
- Variants evaluated: **DFM-PCA**, DFM-IF (Isolation Forest), DFM-OCSVM — all per ID class.
- **Component count: not specified.**

**anomalib** (`methods/dfm/reference/src/anomalib/models/image/dfm/torch_model.py`):

- Single **global** PCA (`:146`, `:161`), `n_comps: float = 0.97` variance ratio (`:139`).
- FRE branch (`:191`): `score = torch.sum(torch.square(features - feats_reconstructed), dim=1)` — squared L2.
- NLL branch (`:186`) via `SingleClassGaussian` (`:38-90`), SVD-based.

## 2. Divergence table

| Component | Reference | Mine (`file:line`) | Changes results? |
|---|---|---|---|
| Feature source | pre-logit (TS-OOD); named timm layer + avg-pool (anomalib `:209-212`) | `_forward_features` pre-logit (`dfm_pca.py:46, 98`) | **No** — matches TS-OOD |
| Modelling granularity | **per ID class** (TS-OOD); global (anomalib `:161`) | one `PCA` per class (`dfm_pca.py:57-79`) | **No** vs TS-OOD (prescribed); differs from anomalib by design |
| Score type | **FRE** (TS-OOD, anomalib `fre`); **Gaussian/GMM NLL** (Ahuja 2019) | FRE (`dfm_pca.py:113-117`) | **No** vs TS-OOD; **YES** vs Ahuja 2019 — different mechanism |
| FRE norm | squared L2 `Σ(x−x̂)²` (anomalib `:191`) | L2 norm `‖x−x̂‖` (`dfm_pca.py:117`) | **No** — monotone (`√` of the same quantity), rank-preserving |
| Class reduction | min over ID class models (TS-OOD) | `min(reconstruction_errors)` (`dfm_pca.py:123`) | **No** |
| Orientation | large error ⇒ OOD | higher = OOD (`dfm_pca.py:123`) | **No** |
| **Component policy** | **variance ratio**: 0.995 (Ahuja), 0.97 (anomalib); unspecified (TS-OOD) | **fixed count 32**, capped `min(32, d, n_c−1)` (`dfm_pca.py:25, 68-72`) | **YES — see §5** |
| NLL branch | offered by anomalib (`:186`) and central to Ahuja 2019 | absent | **No** vs TS-OOD's DFM-PCA (FRE is the named variant) |
| Label requirement | per-class ⇒ labels needed | raises if `y_id is None` (`dfm_pca.py:37-42`) | **No** — correct |
| Small-class guard | n/a | requires ≥2 samples (`dfm_pca.py:61-65`) | **No** — sensible |

## 3. What is faithful

Against the stated TS-OOD target, every defining element checks out:

- One independent PCA per ID class (`:57-79`) — matching *"via an ID class' inverse PCA transform"*.
- Score is the reconstruction residual after project-then-inverse-project (`:113-117`).
- Reduction is the minimum across class models (`:123`), so a sample near any ID class scores low.
- Pre-logit features (`:46`, `:98`), the layer TS-OOD reports as best.
- Orientation higher = OOD, consistent with *"Samples exhibiting large reconstruction errors are flagged as out-of-distribution."*
- The L2-vs-squared-L2 difference is `√`-monotone and therefore invariant under AUROC/AUPR.

## 4. What is an adaptation, not a reproduction

The tracker and `discrepancy_report.md` both cite **Ahuja et al. 2019** as the paper being reproduced.
That paper's method is density-based:

| | Ahuja et al. 2019 | This implementation |
|---|---|---|
| Model | Gaussian / GMM density per class | PCA subspace per class |
| Role of PCA | preprocessing (dim. reduction) before density fitting | **is** the model |
| Score | likelihood / NLL of features | reconstruction residual |
| Component policy | retain 0.995 variance; GMM order by BIC | fixed 32 |

Reproducing FRE is the right choice for this thesis — TS-OOD is the benchmark being extended and it
names the variant "DFM-PCA". But FRE traces to the later Intel Labs papers (ICIP 2022 / BMVC 2023,
Ndiour et al.), not to arXiv:1909.11786. The thesis should cite both, and should not describe this
detector as a reproduction of the 2019 method.

## 5. `n_components=32` is an undocumented policy deviation

The prompt asked to confirm that `n_components=32` is a documented choice. **It is not.**

- **TS-OOD does not specify a component count at all** (verified against the full text).
- Both reference implementations use a *variance-retention* policy, not a fixed count: **0.995**
  (Ahuja et al.) and **0.97** (anomalib `torch_model.py:139`).

This is a difference in *kind*, not just value. A fixed count lets the retained variance drift with
feature dimensionality, whereas the reference policy pins it. Synthetic demonstration (rank-40 data):

| feat_dim | components used | variance retained | comps needed for 0.97 |
|---|---|---|---|
| 32 | 32 | 1.0000 | 22 |
| 64 | 32 | 0.9766 | 31 |
| 128 | 32 | 0.9415 | 36 |
| 256 | 32 | 0.9211 | 37 |
| 512 | 32 | 0.9006 | 38 |

So the *effective model capacity varies across datasets* under the fixed-count policy, in a way the
reference policy is specifically designed to prevent.

**Moreover, 32 is frequently not the operative value.** The cap at `:68-72` is
`min(32, feat_dim, n_class_samples − 1)`. The pipeline trains on ~82 windows split into
`n_pseudo_classes = 4` equal temporal bins (`datasets/tsb_loader.py:161, :41`), i.e. ~20 samples per
class, so the binding term is `n_class_samples − 1 ≈ 19`. The nominal 32 is never reached on such
datasets, and the true component count is set implicitly by dataset size. This should be reported as
the operative behaviour rather than "n_components=32".

### A related bias — real but mild here

Because components are capped per class, classes with fewer samples get fewer components and therefore
systematically larger reconstruction error. Since the score is the **minimum** across classes, such a
class can be excluded from the score entirely. Synthetic demonstration with three classes drawn from an
**identical** distribution, differing only in sample count:

| class n_train | components | mean FRE | chosen as argmin |
|---|---|---|---|
| 200 | 32 | 29.95 | 47.7% |
| 40 | 32 | 30.06 | 52.3% |
| 12 | 11 | 40.17 | **0.0%** |

The undersampled class is never the minimiser despite being distributionally identical.

**Qualification, stated honestly:** the loader divides training windows into *equal* temporal bins
(`tsb_loader.py:41`), so in this project the class sizes differ by at most one sample and hence the
component counts by at most one. The bias is therefore **latent rather than active** under the current
loader — but it would activate immediately under class-imbalanced training data, and it is worth a
guard (use a common component count `min over classes` of the caps).

## 6. Empirical status

Real-data impact of the component-policy deviation **could not be measured**: `experiments/*/*/dfm_pca/`
contains only `scores.npy`, `labels.npy`, `results.json`, and no cached features or backbone
checkpoints exist anywhere in the repo. The tables in §5 are synthetic demonstrations of mechanism and
direction only; no magnitude is extrapolated to the benchmark.

## 7. Recommendations

1. **Cite both papers.** Add Ndiour et al. (ICIP 2022 / BMVC 2023) as the origin of the FRE score, and
   stop describing this detector as a reproduction of Ahuja et al. (2019). Update
   `discrepancy_report.md:4` and the class docstring (`dfm_pca.py:15-22`).
2. **Fix the tracker URL.** `MehrtashHarandi/DFM` is wrong (404). The project page is
   `https://intellabs.github.io/dfm`; note that `IntelLabs/dfm` itself currently 404s.
3. **Document the component policy** as `min(32, d, n_c − 1)` and report the operative value per
   dataset, or switch to a variance-retention policy (e.g. 0.97) to match the references and remove the
   dataset-dependent capacity drift.
4. **Optionally** use a shared component count across classes to close the latent min-over-classes bias.

**Open item shared with the other verifications:** the 18-vs-21 univariate dataset-count discrepancy
recorded in `methods/msp/VERIFICATION.md` §5, `methods/odin/VERIFICATION.md` §7,
`methods/energy_ebo/VERIFICATION.md` §6 and `methods/mahalanobis_mds/VERIFICATION.md` §8 remains
unresolved.

## 8. Conclusion

As a reproduction of TS-OOD's DFM-PCA the implementation is faithful: per-class PCA on pre-logit
features, feature reconstruction error, minimum across class models, higher = OOD. Relative to the
cited 2019 paper it is an adaptation, because that paper scores by Gaussian/GMM likelihood and uses PCA
only as preprocessing. The one substantive deviation from both references is the fixed component count
— an undocumented policy choice that is, in practice, usually overridden by the per-class sample cap.
