# DIVERSIFY Faithfulness Verification — ADAPTATION (label honest in source; three corrections needed)

**Method id:** `diversify` · **Paper:** Lu et al., *Out-of-Distribution Representation Learning for
Time Series Classification*, ICLR 2023 (arXiv:2209.07027); TPAMI extension doi:10.1109/TPAMI.2024.3355212
**Implementation:** `benchmark1/models/ood_methods/diversify.py` (`DIVERSIFYDetector`)
**Verified:** 2026-08-20

---

## Verdict

**ADAPTATION.** The label is honest where it matters — in the source. The docstring says *"adapted for
frozen backbones"* (`:2`), describes accurately what the code computes (`:8-10`), and names the class
**DIVERSIFY-Lite** (`:37`). Nothing claims fidelity.

But the gap is **total, not partial**, and there is a category shift the tracker does not record:

- Official DIVERSIFY adversarially **retrains the feature extractor** via a gradient-reversal layer
  across three update pathways training seven networks. This implementation trains **only K centroid
  vectors** (`:80`) on a permanently frozen backbone (`:69-70`).
- **The original defines no OOD score at all.** It is a classification / domain-generalisation method.
  So this does not simplify DIVERSIFY's scoring — it **invents** a score the paper never defines.

**Three corrections needed:** the "adversarial" misnomer, `validation_status.json`'s zero-discrepancy
PASS, and a citation/venue mismatch (§3).

---

## 1. Source accessibility

| Source | Status |
|---|---|
| `methods/diversify/reference/` | **ABSENT — directory does not exist.** `methods/diversify/` holds only `validation_status.json`. A scan of all `methods/*/reference/` found no `microsoft/robustlearn` clone. No local reference exists. |
| `robustlearn/main/diversify/README.md` | Fetched live |
| `robustlearn/main/diversify/alg/algs/diversify.py` | Fetched live |
| `arxiv.org/abs/2209.07027` | Fetched live (abstract only) |

The comparison rests entirely on live-fetched official sources. The paper's full method section was not
retrieved — only the abstract — so the training description comes from the official algorithm source,
which is the more precise authority for an implementation comparison.

**What the official code does** (`alg/algs/diversify.py`):

| Pathway | Networks trained |
|---|---|
| `update_d` | featurizer, `dbottleneck`, `dclassifier`, `ddiscriminator` (adversarial) |
| `update` | featurizer, `bottleneck`, `classifier`, `discriminator` (adversarial) |
| `update_a` | featurizer, `abottleneck`, `aclassifier` |

- Gradient reversal: `disc_input = Adver_network.ReverseLayerF.apply(disc_input, self.args.alpha)`
- Domain assignment (`set_dlabel`): **cosine** distance on **L2-normalised** features
  (`(all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()`), centres from classifier outputs, refined via
  `cdist(all_fea, initc, 'cosine')`
- Losses: `F.cross_entropy` (class), `F.cross_entropy(disc_out, disc_labels)`, `Entropylogits(cd1)*lam`
- **The featurizer is retrained in all three pathways. No OOD/anomaly score is computed anywhere.**

README: *"a representation-learning and classification method for time series, not an OOD detection
approach."* Abstract: *"first obtains the worst-case distribution scenario via adversarial training,
then matches the distributions of the obtained sub-domains."*

## 2. Divergence table

| Component | Official | Mine (`diversify.py:line`) | Changes results? |
|---|---|---|---|
| **Feature extractor** | retrained adversarially, all 3 pathways | **frozen** (`:69-70`, `:167-168`) | **YES — mechanism absent** |
| **Adversarial training** | gradient reversal w/ `alpha` | **none**; `alpha` weights centroid repulsion `1/(dᵢⱼ+1e-6)` (`:104`) | **YES — not adversarial** |
| **Trainable params** | featurizer + 3 bottlenecks + 3 classifiers + 2 discriminators | **K centroids only** (`:77`, `:80`) | **YES** |
| Discriminators | `discriminator`, `ddiscriminator` | absent | **YES** |
| Entropy / classification loss | `Entropylogits(...)*lam`; `F.cross_entropy` | absent; `y_id` ignored (`:64`) | **YES** |
| Domain assignment geometry | **cosine** on **L2-normalised** features | **Euclidean** on **unnormalised** (`:84`, `:174`) | **YES** |
| **OOD score** | **none — not an OOD method** | min centroid distance (`:174-175`) | **YES — invented** |
| k-means++ init | n/a | unsquared distances (`:138`); standard uses squared | Minor |
| Orientation | n/a | higher distance = OOD (`:177-178`) | Sound for a distance |

## 3. Is the ADAPTATION label honest?

**In the source, yes** — three independent disclosures (`:2`, `:8-10`, `:37`) mean no reader would
mistake it for the original. That is the Tier C standard, and it is met.

**Three things weaken it:**

| Issue | Detail |
|---|---|
| "Adversarial" misnomer | Used at `:9`, `:32`, `:41`, `:79`, `:99`. The actual objective is `cluster_loss + α·Σ 1/dᵢⱼ` (`:109`) — both terms *minimised together*, no opposing objective, no gradient reversal. Recommend "diversity-regularised k-means". |
| Zero-discrepancy record | `validation_status.json`: `CRITICAL: 0, MODERATE: 0, MINOR: 0`, `status: "PASS"`, `notes: ""`. Machine-readable, and it contradicts the docstring. |
| Citation mismatch | `:4-5` pairs the **TPAMI** DOI with the label **"(ICLR 2023)"**. Cite arXiv:2209.07027 / ICLR 2023, or the TPAMI version by its own venue. |

Also: `@register_ood("diversify")` (`:35`) makes results tables read as "diversify". Since the original
is not an OOD method at all, a suffix such as `diversify_lite` would prevent that implication.

**Note:** the phrase the tracker expects — *"feature-space distance adaptation"* — does **not** appear
in the code. The wording used ("lightweight adaptation", "DIVERSIFY-Lite") is honest but vaguer.

## 4. Measured results

| Subset | n | mean AUROC | below chance |
|---|---|---|---|
| All loadable | 37 | **0.6993** | — |
| TSB-U (univariate) | **18** | **0.6598** | 6/18 |

Third-strongest detector verified so far, behind SRS (0.841) and DiMMAD (0.790) — reinforcing that
feature-space distance methods dominate this benchmark while logit-space methods invert.

Scores span `[0.088, 5.818e6]`, all non-negative (consistent with Euclidean distance). **No degenerate
datasets** (0/37 with range < 1e-12) — a favourable contrast with CODiT's 19/40. The `5.8e6` extreme on
`TSB-M-STABLE_015` is the pathological-feature dataset recorded in
`methods/energy_ebo/VERIFICATION.md` §3.

### Three datasets produced no output

`TSB-U-STABLE_062`, `TSB-U-STABLE_070`, `TSB-U-STABLE_080` each have a result directory but **no
`scores.npy` / `labels.npy`** — 40 directories, 37 usable. All three are small YAHOO series, plausibly a
silent failure when `N < latent_domain_num = 10`: k-means++ (`:135-140`) samples 10 centroids from N
points, and `torch.multinomial` on a degenerate `probs` vector can fail. Should fail loudly instead.

**This bears directly on the long-running dataset-count question.** DIVERSIFY is the **first method
verified whose univariate count is 18** — matching the figure the validation notes cite throughout —
precisely because these three TSB-U datasets are missing. Every other method covers 21 (SRS: 20). That
makes it plausible the "18 univariate datasets" figure originated from a run with missing outputs rather
than a defined subset, which would explain why no such subset definition exists in any config.

## 5. Recommendations

1. **Replace "adversarial" with "diversity-regularised"** throughout — the only place the labelling
   overstates fidelity.
2. **Fix `validation_status.json`** — record the mechanism replacement, or add an explicit
   `adaptation: true`. A zero-discrepancy PASS is indefensible here.
3. **Rename the registry key** to `diversify_lite`, and state in the thesis that **the original defines
   no OOD score** — a stronger caveat than "adaptation" alone conveys.
4. **Fix the citation** venue/DOI pairing.
5. **Clone a reference** (`microsoft/robustlearn`, `diversify/`) — the folder has none.
6. **Investigate the three empty result directories**; fail loudly when `N < latent_domain_num`.

**Open item shared with the other verifications:** the univariate dataset-count discrepancy recorded in
`methods/msp`, `odin`, `energy_ebo`, `mahalanobis_mds`, `dfm`, `srs`, `react`, `dice`, `scale`,
`gradnorm`, `dimmad` and `codit` VERIFICATION.md files — **§4 above offers a candidate explanation.**

## 6. Conclusion

The implementation is a frozen-feature clustering detector: k-means++ init, centroids refined under a
clustering-plus-repulsion objective, scored by distance to the nearest centroid. That is accurately
described in its own docstring and correctly labelled an adaptation, so the Tier C label stands. But the
gap from the original is total rather than partial — DIVERSIFY retrains the feature extractor
adversarially through gradient reversal across seven networks, assigns domains by cosine distance on
normalised features, and defines **no OOD score whatsoever**. The right framing for the thesis is not
"a simplified DIVERSIFY" but "a feature-space distance detector inspired by DIVERSIFY's latent-domain
idea".

---

## CLASS-D BUILD (2026-08-21)

This section records the **faithful-as-possible Class-D appendix build** of DIVERSIFY, built to answer the
§6 critique above: unlike the production `diversify.py` (frozen backbone, K centroids only, no adversary),
this build reproduces DIVERSIFY's defining mechanism — an adversarial feature extractor trained through a
Gradient Reversal Layer — and adds an honestly-captioned OOD score.

### Files created / edited
- **Created** `methods/diversify/classd/diversify_classd.py` — `DiversifyClassD`: a minimal from-scratch
  DANN-style DIVERSIFY adapted to 1-D TSB windows.
- **Edited** `experiments/run_class_d.py` — registered `diversify` (eval_mode `per_sample_selftrain`),
  added `GROUP3_MODES`, `_selftrain_and_score`, `_process_file_group3`, `run_group3`, and gated it in
  `main()` under `TSB_GROUP=3` (also included in `all`). Group I/II code paths are untouched.

### What the build does (GRL adversarial representation learning)
- **Trains its OWN extractor from scratch** on the ID train windows — it does **not** reuse the shared
  frozen ResNet backbone (the whole point of DIVERSIFY is the retraining). Seven-network upstream is
  distilled to four adapted components: a 1-D CNN featurizer `F`, a pseudo-class head `C`, a latent-domain
  characterizer, and a domain-adversarial head `D` through a GRL.
- **Alternating objective per epoch:** (a) re-assign latent-domain labels by k-means on **cosine** distance
  over **L2-normalised** features (the paper's geometry); (b) minimise `CE_class(C(F(x)), y_pseudo) +
  CE_domain(D(GRL_alpha(F(x))), d)`. The GRL negates the gradient into `F` (scaled by `alpha`), so `D`
  maximises latent-domain **diversity** while `F` is pushed to be domain-invariant given the class — the
  paper's worst-case-distribution / distribution-matching intuition.

### Invented OOD score — the paper defines NONE
DIVERSIFY is a domain-generalisation classifier and defines no anomaly/OOD score. Per
`methods/_validation/CLASS_D_DECISIONS.md` this build **adds** one and says so plainly:
- **primary = energy** `-logsumexp(C(F(x))/T)` (higher = OOD);
- **secondary = cosine-centroid** = cosine distance to the nearest class centroid on L2-normalised learned
  features (higher = OOD).
Both variants are reported (carried in the CSV `arm` column as `energy` / `cosine_centroid`).

### N < latent_domain_num crash — FIXED
The production version left three empty TSB-U result dirs (VERIFICATION §4) — a silent failure when
`N < latent_domain_num = 10`. This build clamps `latent_domain_num` to the data
(`k = min(latent_domain_num, N // min_per_domain)`, `min_per_domain=2`) and **skips the adversarial branch
entirely** when the stream cannot form ≥ 2 domains (single-domain fallback). Verified on a tiny file
(`STABLE_003`, train = 4 windows): clamped to `eff_domains=2`, ran to completion, produced finite AUROCs —
no crash.

### Verification sweep (TSB_GROUP=3, TSB_N_PER_CELL=2, U only)
Result: **12/12 (method, score-variant) runs produced finite per-sample AUROCs** →
`results/class_d_group3.csv`. Rows:

| dataset (U) | cat | energy AUROC | cosine-centroid AUROC | n_eval | eff_domains |
|---|---|---|---|---|---|
| DRIFT_002_…NAB_id_19 | DRIFT | 1.000 | 0.694 | 12 | 5 |
| DRIFT_004_…WSD_id_77 | DRIFT | 0.319 | 0.743 | 24 | 5 |
| OOD_001_…WSD_id_90 | OOD | 0.216 | 0.746 | 50 | 5 |
| OOD_002_…NAB_id_24 | OOD | 0.296 | 0.314 | 26 | 5 |
| STABLE_002_…WSD_id_20 | STABLE | 0.222 | 0.444 | 6 | 5 |
| STABLE_003_…YAHOO_id_30 (tiny, train=4) | STABLE | 1.000 | 1.000 | 2 | **2 (clamped)** |

### Caveats (honest framing for the thesis)
- **The score is an addition, not the method.** There is no ground-truth "correct" DIVERSIFY OOD number to
  validate against; results are **exploratory**. Caption: *"score added; paper defines none."*
- On these tiny ID sets (4–260 windows) the **energy** score is frequently below chance while
  **cosine-centroid** is the more reliable of the two invented variants on the larger files (0.69–0.75) —
  consistent with the broader finding that feature-space distance beats logit-space on this benchmark. The
  extreme AUROCs (1.0 / 0.222) on `n_eval ∈ {2, 6}` files are statistically meaningless and shown only to
  demonstrate the edge case does not crash.
- **Fair-comparison BREAKING:** this learns its own representation, so it can never join the 17-method
  frozen-backbone leaderboard — appendix study only.

### Integrity confirmation (md5 unchanged)
- `experiments/run_experiments.py` `c175f519c1f6cbd2c1d8b6f3ef1b1ae2` — unchanged
- `experiments/tsb_benchmark.py` `f3ebbcc926dd97ec66978f4e62ca73d3` — unchanged
- `benchmark1/models/ood_methods/diversify.py` (production) `e0398a3a28f5fbab9a746e26b0324b8d` — unchanged
- `results/class_d_group1.csv` `570f2b4b40f47766f47f71d3796ab497` — unchanged (37 lines)
- `results/class_d_group2.csv` `de9783bf35e425c67ac55c321c13ad12` — unchanged (19 lines)
