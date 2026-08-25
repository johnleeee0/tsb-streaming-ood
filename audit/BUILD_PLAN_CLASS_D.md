# BUILD PLAN — Class D (currently-excluded) OOD detectors

**Author target:** S. Giannoulis (AUTH MSc) · **Date:** 2026-08-21 · **Status:** PLAN ONLY — no code changed.

Goal: make the 7 Class-D detectors run *faithfully* (each as the method its paper describes), instead of
the current unfaithful stand-ins that got them excluded (see `CLASS_D_EXCLUSIONS.md` and each
`methods/<id>/VERIFICATION.md`). None of the 7 can join the main 17-method leaderboard without breaking
the fair-comparison design (shared frozen ResNet backbone, shuffled per-window eval, no auxiliary
outliers). Every faithful build below is therefore a **separate appendix study**.

The 7 fall into three infrastructure groups:

- **(I) auxiliary-outlier corpus + a training/fine-tuning loop** — `outlier_exposure`, `divoe`, `diversemix`
- **(II) an ordered / window-level evaluation path** — `driftlens`, `tdivdm`, `ae_adwin_lstm`
- **(III) DIVERSIFY-specific adversarial representation training + an invented scoring rule** — `diversify`

---

## How the current harness works (facts this plan builds on)

- `datasets/tsb_loader.py::load_tsb` — sliding windows (`_sliding_windows`), boundary split (train =
  Source-1 normals only), builds a **balanced** val/test pool and **shuffles it**:
  `_balance_binary` (`:148`) calls `rng.permutation`. So temporal order in val/test is destroyed. ID
  windows = held-out Source-1 normals; OOD windows = every window containing ≥1 anomalous timestep. Labels
  are per-window (0 = all-normal, 1 = any-anomaly).
- `experiments/run_experiments.py::train_backbone` — one **shared** `ResNetBackbone` (1-D ResNet,
  128-d embedding) + a separate `nn.Linear(128, n_classes)` head, trained with CE on temporal
  pseudo-classes, seed 42, CPU, 40 epochs. The trained `bb, head` are reused by **every** detector in a
  file — this is the fair-comparison anchor.
- `run_one` — builds the detector with `config["classifier"]=head`, calls `fit(train.x[,y])` then
  `score(val.x)` and `score(test.x)`; computes per-sample AUROC/AUPR/FPR95 from `test.y` and the returned
  per-window scores. Higher score must mean "more OOD".
- `base_ood.BaseOODDetector` — `_forward_features(x)` returns the 128-d **embedding** (skips head);
  `_forward_logits(x)` applies the head → class logits; `_energy = -logsumexp(logits/T)`.
- `tsb_benchmark.py::method_set` — the production 17; the 7 here are excluded.
- Reference clones present: `outlier_exposure` (full Hendrycks repo, `CIFAR/oe_tune.py`), `divoe`
  (`src/train_DivOE.py`), `driftlens` (full pip package incl. `driftlens/`, `distribution_distances/`).
  Reference clones **absent** (no public code): `diversemix`, `tdivdm`, `ae_adwin_lstm`, `diversify`
  (diversify's upstream is `microsoft/robustlearn`, fetchable but not cloned).

---

## Summary table

| method | group | infra needed | eval approach (how per-sample AUROC is defined) | effort | key design decision |
|--------|-------|--------------|--------------------------------------------------|--------|---------------------|
| outlier_exposure | I | aux-outlier corpus + fine-tune (CE + 0.5·CE-to-uniform) | per-sample energy/MSP on the **fine-tuned** net — identical eval to now, AUROC well-defined | M | which aux corpus (hold-out TSB files vs synthetic) |
| divoe | I | aux corpus + **feature-space PGD synthesis** + OE fine-tune | per-sample energy on fine-tuned net — as now | L | PGD in input vs frozen-feature space |
| diversemix | I | **real** aux corpus (replace ID-mixing) + energy-head training w/ score-adaptive mixup | per-sample −logsumexp of energy head — as now | M | aux corpus; keep −logsumexp orientation (official) |
| driftlens | II | ordered eval + **batch/window-level** Fréchet path | **batch-level AUROC**: batches of B consecutive windows, batch label = (frac OOD ≥ 0.5), score = Fréchet(batch, baseline). Per-sample AUROC not native → report batch metric | M | monitoring batch size B; batch-label rule |
| tdivdm | II | ordered multi-scale windows (+ TS2Vec branch) | ordered stream; per-window −logdensity aggregated across scales → per-sample AUROC works | XL | paper is paywalled + no code → how "faithful" is even attainable |
| ae_adwin_lstm | II | ordered stream + real ADWIN + drift-triggered incremental update | per-window score in **time order** → per-sample AUROC directly (flip orientation) | L | keep per-window score vs report stream drift-delay metric |
| diversify | III | adversarial extractor retraining (GRL, 7 nets) + **invented** OOD score | per-sample score from retrained net → AUROC works | XL | which invented score (energy vs cosine-centroid) |

---

## Shared infrastructure

Both shared pieces are added as **new, self-contained files plus one optional flag**, so the production
harness (`run_experiments.py`, `tsb_benchmark.py`) and its resumable results are untouched. Class-D runs
live in a dedicated appendix runner.

### (I) Auxiliary-outlier corpus + training/fine-tuning loop

**New file `benchmark1/datasets/aux_outliers.py`:**
```
load_aux_outliers(split, in_channels, window_size, stride,
                  exclude_file, n, seed, source="tsb_holdout") -> np.ndarray  # (n, C, T)
```
- `source="tsb_holdout"` (recommended default): draw windows from a **fixed pool of held-out TSB files**
  of the same split that are *never* used as evaluation files, and never the current `exclude_file`. Filter
  to files whose channel count == `in_channels` (trivially true for U; for M require exact match, else fall
  back to synthetic). Normalise with the *same* rule as the eval file (`per_series`/`global`). Mix normal
  and anomalous windows — OE's corpus need not be "labelled OOD", only disjoint from ID/test.
  **Leakage guard:** the pool of aux files is disjoint from the sweep's evaluation files (partition the 600
  corpus once into `eval_files` and `aux_files`, persisted to a JSON manifest).
- `source="synthetic"` (fallback/ablation): generate outliers from the ID train windows via (a) Gaussian
  noise injection, (b) FFT phase-shuffle, (c) cross-window feature mixup. Cheap, zero leakage, but a weaker
  "faithful OE corpus" claim — OE's paper permits any broad outlier set, so this is defensible as an
  ablation arm.

**New training helpers in the appendix runner** (`experiments/run_class_d.py`, new file):
```
finetune_oe(bb, head, dataset, aux_x, epochs, oe_weight=0.5, lr, seed)      -> (bb2, head2)
finetune_divoe(bb, head, dataset, aux_x, pgd_steps, eps, ...)               -> (bb2, head2)
```
- **Critical for fairness:** these operate on a **`copy.deepcopy` of the shared `bb`/`head`**, so each
  group-I method gets its own fine-tuned model and the shared frozen backbone that the 17 production methods
  use is never mutated. The scoring detector (`OutlierExposureDetector` etc.) is then built against the
  fine-tuned copy, and its `score()` path is unchanged — so the existing per-sample AUROC machinery in
  `run_one` applies verbatim.
- Which weights to update is a design decision (see DECISIONS): **head-only** (keeps the ResNet frozen, so
  it stays *closer* to the shared-backbone design) vs **full net** (the paper's actual behaviour, maximally
  faithful, maximally unfair). Recommended default: run **both** as two arms and report the pair.

**Files changed:** none in production; new `datasets/aux_outliers.py`, new `experiments/run_class_d.py`, new
`experiments/aux_file_manifest.json`.

### (II) Ordered / window-level evaluation path

**One flag in `datasets/tsb_loader.py::load_tsb`:** add `ordered_eval: bool = False`. When True:
- skip the `rng.permutation` shuffle in `_balance_binary` (add `shuffle=` param, or bypass balancing);
- additionally return `dataset["stream"] = {"x": x_eval_ordered, "y": y_eval_ordered, "t": start_rows}` —
  the **full held-out window sequence in original temporal order** (Source-1 held-out normals followed by
  Source-2/anomaly region as they occur in the file), with each window's start row. This is the ordered
  stream the temporal methods need. Train pool and boundary logic are unchanged.

**New helper in the appendix runner:**
```
make_monitoring_batches(stream_x, stream_y, batch_size B, batch_stride)
    -> batches (M, B, C, T), batch_labels (M,)   # batch_label = 1 if mean(window OOD) >= tau, else 0
```
Used by `driftlens` (batch-level Fréchet) and available to `tdivdm`.

**New metric path:** the appendix runner computes **batch-level** AUROC/AUPR for driftlens (and any
window-level method) using `eval/metrics.py` unchanged (it is metric-agnostic; just feed batch scores +
batch labels). Per-window methods on the ordered stream (`ae_adwin_lstm`, `tdivdm`) feed per-window scores
+ per-window labels → the existing per-sample AUROC applies directly.

**Files changed:** `tsb_loader.py` (one optional flag + ordered return, fully backward-compatible — default
`False` reproduces today's behaviour byte-for-byte); everything else new in `run_class_d.py`.

### (III) DIVERSIFY adversarial training + scoring rule

Not shareable — specified under the `diversify` method below. Needs a from-scratch extractor + GRL training
module (`experiments/diversify_train.py`, new) that does **not** reuse the shared backbone.

---

## Per-method detail

### 1. outlier_exposure — group I — effort M

1. **Faithful implementation.** OE fine-tunes the classifier with
   `L = CE(f(x_id), y_id) + λ · CE_uniform(f(x_out))`, where the OE term is
   `0.5 · −(mean_k z_k − logsumexp_k z_k)` on the auxiliary-outlier half of each batch
   (`reference/CIFAR/oe_tune.py:172-177`; λ=0.5). Train for a few epochs, then score with energy
   `−logsumexp(z/T)` (or MSP) on the **fine-tuned** net. Orientation: higher = OOD (unchanged).
2. **Infrastructure:** group (I). Aux corpus from `load_aux_outliers`; `finetune_oe` on a deepcopy of
   `bb`/`head`. The existing `OutlierExposureDetector.score` is reused as-is on the fine-tuned model.
3. **Evaluation:** identical to the current per-window protocol — per-sample energy scores over the shuffled
   balanced test set; AUROC/AUPR/FPR95 well-defined with no change to `run_one`'s math.
4. **Fair comparison:** BREAKS it — OE updates weights the 17 methods do not have. Report as an appendix
   "OE / DivOE / DiverseMix trained-detector study", never in the 17-method table. (Keep the current clean
   **Energy (EBO)** row in the main table, as today.)
5. **Effort M. Risks:** aux-corpus leakage (mitigated by the eval/aux file partition manifest); with tiny
   ID sets (~60–80 windows) fine-tuning can overfit — use early stopping on the val split and few epochs;
   head-only vs full-net changes results a lot.
6. **Design decisions:** aux corpus source (hold-out TSB vs synthetic); update head-only vs full net; λ,
   epochs, lr, oe_batch_size.

### 2. divoe — group I — effort L

1. **Faithful implementation.** DivOE = OE **plus** synthesising diversified outliers by multi-step
   projected-gradient ascent on the OE objective ("informative extrapolation",
   `reference/src/train_DivOE.py:177-197`): start from an aux outlier, take `num_steps` sign-gradient steps
   (default 5, ε=0.01, step ε/4) that *increase* the OE loss, project back into the ε-ball; replace a
   fraction (`extrapolation_ratio`=0.5) of the aux batch with these, then apply the OE fine-tune loss.
2. **Infrastructure:** group (I). Needs `load_aux_outliers` + `finetune_divoe` (adds the PGD inner loop).
   Because the shared backbone is frozen and the harness passes features/inputs, the design decision is
   whether PGD runs in **raw input space** (paper-faithful; requires the aux windows as differentiable
   inputs through the ResNet) or in **frozen-feature space** (cheaper; extrapolate on 128-d embeddings).
   Recommended default: input-space PGD on the fine-tuned copy for fidelity.
3. **Evaluation:** per-sample energy on the fine-tuned net — same as OE, AUROC well-defined.
4. **Fair comparison:** BREAKS it (training + synthesis). Same appendix as OE.
5. **Effort L. Risks:** PGD through a 1-D ResNet on CPU is slow; ε in normalised-window units is not the
   paper's [0,1]-image ε, so ε must be recalibrated per normalisation mode; synthesis on ~80 ID windows may
   add little over plain OE (report the OE-vs-DivOE delta honestly).
6. **Design decisions:** PGD space (input vs feature); ε, num_steps, step size, extrapolation_ratio,
   extrapolation_score (MSP vs energy).

### 3. diversemix — group I — effort M

1. **Faithful implementation.** Train an energy head on features with CE on ID + auxiliary loss
   `relu(logsumexp+1)` pushing **real** aux-outlier energy down, using the score-adaptive mixup
   `λ ~ Beta(ŝ_i·α, ŝ_j·α)` **applied to the real aux corpus** (the paper "diversifies a *collected*
   auxiliary outlier set"). The existing `diversemix.py` already implements the energy head, the auxiliary
   loss, and the Beta-mixup — the only faithful change is replacing `_generate_auxiliary_outliers`
   (cross-class ID mixing, which lands 100% inside the ID distribution per the VERIFICATION) with
   `load_aux_outliers`. Score = **−logsumexp** (official `eval_ood_detection.py` orientation; the current
   `+logsumexp` at `:306` is wrong).
2. **Infrastructure:** group (I). Aux corpus only; the head trains on frozen features so this one can
   *optionally* keep the backbone frozen (closest to fair design of the three group-I methods).
3. **Evaluation:** per-sample −logsumexp of the energy head — AUROC well-defined, no `run_one` change.
4. **Fair comparison:** BREAKS it (trains an extra energy head on an external corpus). Appendix. If the
   backbone stays frozen and only the small head trains on an external corpus, note it is *less* unfair than
   full OE but still not one of the 17.
5. **Effort M. Risks:** with the current fabricated corpus it is at chance (0.52) — a real corpus is the
   whole point; if `load_aux_outliers` falls back to synthetic ID-interpolants the method collapses again,
   so the corpus quality gates the result. Keep the negative-result framing available if a real corpus is
   unavailable for a split.
6. **Design decisions:** aux corpus source; α, temperature, ω, epochs; backbone frozen vs head+backbone.

### 4. driftlens — group II — effort M

1. **Faithful implementation.** Official DriftLens is **window/batch-level**: per monitoring window it
   computes a distribution-to-distribution **Fréchet (Wasserstein-2)** distance to the ID baseline in
   PCA space (`reference/driftlens/driftlens.py:267`; also KL/Bhattacharyya/JS/dist-Mahalanobis), plus a
   per-label decomposition. There is **no per-sample score** in the paper. The repo's `score_batch()`
   (`driftlens.py:242-279`) already computes exactly this batch Fréchet — wire it into an ordered,
   batched eval instead of the dead per-sample Mahalanobis `score()`.
2. **Infrastructure:** group (II). Ordered stream from `load_tsb(ordered_eval=True)` +
   `make_monitoring_batches`. Optionally add per-label PCA budgets for full fidelity (secondary).
3. **Evaluation — how batch scores map to AUROC (critical).** Native granularity is the **batch**, so
   define a **batch-level AUROC**: form batches of B consecutive windows; a batch's label is 1 iff the
   fraction of OOD windows ≥ τ (default τ=0.5); its score is the batch Fréchet distance; AUROC over all
   batches. This is a *well-defined, honest* metric at the method's true granularity. A per-sample AUROC is
   **not** natively definable (broadcasting the batch score to its windows creates massive ties and simply
   reproduces the `mahalanobis` detector, ρ median 0.999 — see VERIFICATION §3A). **Recommendation:** report
   driftlens only on the batch-level leaderboard; state explicitly that its per-sample proxy is redundant
   with `mahalanobis`.
4. **Fair comparison:** the *scoring* uses the shared frozen backbone (no training), so it does not break
   the backbone anchor — but it uses a **different eval granularity**, so it cannot share the 17-method
   per-sample table. Put it in a small "window-level drift" appendix table (driftlens vs a batch-Fréchet /
   batch-Mahalanobis baseline).
5. **Effort M. Risks:** few OOD windows per file → few positive *batches* → noisy batch AUROC; `sqrtm` on
   near-singular covariances is slow/unstable (regularise, use `n_components` ≪ batch size); choice of B and
   τ strongly affects the number of batches.
6. **Design decisions:** monitoring batch size B; batch stride; batch-label threshold τ; distance metric
   (Fréchet default) and PCA budget; whether to include the per-label decomposition.

### 5. tdivdm — group II — effort XL (fidelity partly unattainable)

1. **Faithful implementation.** TD-IVDM = "Time Dependency – Inter Variable Dependency": an improved
   **TS2Vec** representation branch for time dependencies **plus** multi-dimensional KDE for inter-variable
   dependencies, applied **multi-scale** (smaller time frames + variable subsets), in a
   preprocess→detect→postprocess streaming workflow. The current impl is a single global Gaussian KDE on
   20-d PCA of frozen features — one of the two pillars, no multi-scale, no TS2Vec, no stream.
2. **Infrastructure:** group (II) ordered multi-scale windows, **plus** a TS2Vec-style temporal encoder
   (new, `experiments/ts2vec_lite.py`) — which itself is a representation-learning stage that would not use
   the shared backbone.
3. **Evaluation:** on the ordered stream, aggregate per-scale −logdensity into a per-window score →
   per-sample AUROC works directly (each window keeps a score). Multi-scale drift could alternatively be
   reported as a batch/segment drift-delay metric.
4. **Fair comparison:** BREAKS it (adds a learned TS2Vec representation; multi-scale windows differ from the
   fixed window/stride). Appendix only.
5. **Effort XL. Risks — the dominant one is fidelity itself:** the paper is **paywalled** (ScienceDirect &
   ResearchGate both 403) and **no public code exists**, so a truly faithful reproduction cannot be verified
   against the source. Any build is an educated reconstruction. Recommended honest path: **do not claim a
   faithful TD-IVDM.** Either (a) keep it reported as "KDE-density" (already accurate to the KDE pillar) in
   the main study, or (b) build a *multi-scale ordered KDE + TS2Vec-lite* variant and label it
   "TD-IVDM-inspired (unverifiable against source)". This is the weakest-fidelity method of the seven.
6. **Design decisions:** attempt reconstruction at all vs keep as KDE-density; if reconstructing — the set
   of temporal scales, TS2Vec-lite architecture/epochs, how variable-subset KDE is defined for univariate
   series (degenerate → scales over channels only exist for M).

### 6. ae_adwin_lstm — group II — effort L

1. **Faithful implementation.** Autoencoder (spatial) + LSTM next-step predictor (temporal) + **real ADWIN**
   monitoring the LSTM residual stream, with the paper's defining step: **drift-triggered incremental
   update** of the model when ADWIN fires. Requires (a) an **ordered** stream so LSTM history and ADWIN are
   meaningful, (b) a real ADWIN (exponential-histogram buckets, all cut points, Hoeffding bound δ/n) instead
   of the current midpoint two-sample test, (c) resetting ADWIN at the start of scoring, (d) fixing the
   positional artefact (first `seq_len−1` windows scored differently), and (e) **flipping orientation**
   (0.25→0.75 per VERIFICATION §3D; the code currently negates on a false premise).
2. **Infrastructure:** group (II). Ordered stream from `load_tsb(ordered_eval=True)`; a real ADWIN class
   (new, `experiments/adwin.py` or `river.drift.ADWIN` if the dependency is acceptable).
3. **Evaluation — per-sample AUROC (well-defined once ordered).** The native output is already a
   **per-window scalar** in stream order, so feeding per-window scores + per-window OOD labels gives a
   per-sample AUROC directly — no batch mapping needed. The incremental-update loop makes the score
   history-dependent, which is the point; report it on the ordered stream. Optionally also report a
   stream-level **drift-detection-delay** metric (windows between true drift onset and first ADWIN alarm) as
   the paper's native output.
4. **Fair comparison:** BREAKS it — incremental updates mutate the model during scoring, and the ordered
   eval differs from the shuffled protocol. Appendix "ordered-stream temporal detectors" table (alongside
   ordered driftlens/tdivdm).
5. **Effort L. Risks:** incremental update on ~40 ordered eval windows may barely trigger; ADWIN
   sensitivity δ dominates behaviour; ordered eval has class imbalance (ID then OOD) — AUROC still valid but
   det-accuracy threshold selection on val must also be ordered.
6. **Design decisions:** real ADWIN implementation (custom vs `river`); δ; whether to enable the incremental
   update (faithful) or report AE-reconstruction-only as an order-invariant control; per-window AUROC vs
   drift-delay as the headline metric.

### 7. diversify — group III — effort XL

1. **Faithful implementation.** Official DIVERSIFY (`microsoft/robustlearn`) adversarially **retrains the
   feature extractor** via a gradient-reversal layer across three update pathways (`update_d`, `update`,
   `update_a`) training seven networks (featurizer + 3 bottlenecks + 3 classifiers + 2 discriminators);
   domains are assigned by **cosine** distance on **L2-normalised** features. Crucially the paper **defines
   no OOD score** — it is a domain-generalisation classifier. A faithful build must (a) reproduce the
   adversarial latent-domain training on the ID data, then (b) **invent** an OOD scoring rule.
2. **Infrastructure:** group (III). A from-scratch GRL training module
   (`experiments/diversify_train.py`) that trains its **own** featurizer (cannot be the shared frozen
   backbone — the whole method *is* the retraining). This is fully outside the shared-backbone design.
3. **Evaluation — the invented score must yield a per-sample number.** Candidate scoring rules (all give a
   per-window scalar → per-sample AUROC works): (a) **energy** `−logsumexp` of the trained classifier;
   (b) **MSP** `1−max softmax`; (c) **min cosine distance** to the learned L2-normalised domain centroids
   (closest to the current impl but on retrained, normalised features); (d) domain-discriminator confidence.
   Recommended default: report (a) energy as primary and (c) cosine-centroid as a secondary, and be explicit
   in the thesis that the score is an addition not present in the paper.
4. **Fair comparison:** BREAKS it comprehensively (retrains the representation; the 17 share a frozen one).
   Appendix "representation-learning detectors" study, clearly captioned "score invented; original defines
   none".
5. **Effort XL. Risks:** reproducing 7-network adversarial training on ~60–80 ID windows on CPU is fragile
   and may not converge; upstream code must be adapted from image/HAR inputs to 1-D TSB windows; the invented
   score means there is no ground-truth "correct" DIVERSIFY OOD number to validate against — frame as
   exploratory. Also fix the empty-output crash when `N < latent_domain_num` (VERIFICATION §4).
6. **Design decisions:** invented scoring rule (energy vs cosine-centroid vs discriminator); `latent_domain_num`,
   `alpha` (GRL weight), epochs; adapt upstream code vs re-implement GRL minimally; how to caption a method
   whose OOD score the paper never defines.

---

## DECISIONS NEEDED FROM AUTHOR

Each item lists a recommended default. These are the choices the plan cannot make on the author's behalf.

**Shared / cross-cutting**
1. **Auxiliary-outlier corpus (I).** Hold-out TSB files (real, disjoint from eval) vs synthetic
   (noise/FFT-shuffle/mixup of ID) vs both. → **Default: hold-out TSB files as primary, synthetic as an
   ablation arm.** Requires partitioning the 600-file corpus into `eval_files`/`aux_files` (persisted
   manifest) to prevent leakage.
2. **Multivariate aux matching.** M-split aux windows must match channel count. Require exact-C match, else
   fall back to synthetic for that file. → **Default: exact-C match, synthetic fallback.**
3. **What weights fine-tuning updates (I).** Head-only (keeps ResNet frozen, closer to fair design) vs full
   net (paper-faithful). → **Default: run both arms; report the pair.**
4. **Where Class-D runs live.** A separate `experiments/run_class_d.py` producing an appendix table, leaving
   `run_experiments.py`/`tsb_benchmark.py` and their resumable results untouched. → **Default: yes, separate
   runner + separate appendix tables; never merge into the 17-method leaderboard.**
5. **Ordered-eval flag (II).** Add `ordered_eval=False` to `load_tsb` (default reproduces today's shuffle
   exactly). → **Default: add the flag, backward-compatible.**

**Per-method**
6. **OE hyperparameters.** λ (OE weight), epochs, lr, oe_batch_size. → **Default: λ=0.5, 10 epochs, lr=1e-3,
   oe_batch_size=64 (scaled to the tiny ID sets), early-stop on val.**
7. **DivOE PGD space.** Input-space (faithful) vs frozen-feature-space (cheap). → **Default: input-space PGD
   on the fine-tuned copy; ε recalibrated to normalised-window units; num_steps=5, ratio=0.5.**
8. **DiverseMix orientation & backbone.** → **Default: −logsumexp (official), backbone frozen + head-only
   training (least unfair arm), α/T/ω as in the current file.**
9. **DriftLens monitoring batch.** Batch size B, stride, batch-label threshold τ, distance metric. →
   **Default: B=32 (U) / 16 (M) consecutive windows, non-overlapping, τ=0.5, Fréchet distance; report
   batch-level AUROC only.**
10. **DriftLens redundancy note.** Report per-sample proxy at all? → **Default: no — state it duplicates
    `mahalanobis` (median ρ 0.999) and report batch-level only.**
11. **TD-IVDM: reconstruct or not.** Attempt a multi-scale TS2Vec+KDE reconstruction (XL, unverifiable) vs
    keep "KDE-density". → **Default: keep KDE-density in the main study; if an appendix reconstruction is
    built, label it "TD-IVDM-inspired (unverifiable — paper paywalled, no public code)".**
12. **AE-ADWIN-LSTM: ADWIN impl & incremental update.** Custom ADWIN vs `river.drift.ADWIN`; enable the
    drift-triggered incremental update (faithful) or not. → **Default: use `river` ADWIN if the dependency
    is acceptable else a correct custom ADWIN; enable incremental update; flip orientation; headline
    per-window AUROC on the ordered stream + report drift-delay as secondary.**
13. **DIVERSIFY invented score.** energy vs cosine-centroid vs discriminator confidence. → **Default: energy
    `−logsumexp` primary, cosine-centroid secondary; caption explicitly "score added; paper defines none".**
14. **DIVERSIFY code path.** Adapt `microsoft/robustlearn` upstream vs minimal from-scratch GRL. → **Default:
    minimal from-scratch GRL adapted to 1-D TSB windows; fix the `N < latent_domain_num` crash.**
15. **Reporting frame for the three "can't be made faithful here" cases.** driftlens (redundant per-sample),
    tdivdm (unverifiable), diversemix-with-synthetic (at chance): keep as explicit **negative/'
    protocol-incompatible' results** in the appendix rather than headline rows. → **Default: yes, keep the
    honest negative framing already established in `CLASS_D_EXCLUSIONS.md`.**

---

*This is a planning document. Implementing any item above is a code change to be done as separate,
clearly-captioned appendix studies, none of which enters the 17-method fair-comparison leaderboard.*
