# Core Reference Paper — Structured Summary

**Author of this thesis project:** Stylianos Giannoulis
**Institution:** Aristotle University of Thessaloniki (AUTH), MSc in Data and Web Science
**Supervisor:** John Paparrizos

> This file summarises `C:\THESIS\PAPER.pdf`, the core reference against which the
> thesis benchmark is built and extended. It is the canonical source for notation,
> protocol, and terminology to reuse throughout the thesis.

---

## 1. Bibliographic details

- **Title:** TS-OOD: Evaluating Time-Series Out-of-Distribution Detection and Prospective Directions for Progress
- **Authors:** Onat Gungor, Amanda Rios, Nilesh Ahuja, Tajana Rosing
- **Affiliations:** University of California, San Diego (UCSD); Intel Labs
- **Venue:** AAAI 2025 (Association for the Advancement of Artificial Intelligence). Preprint: arXiv:2502.15901v1, 21 Feb 2025.
- **Citation key:** `Gungor2025`

## 2. Problem framing and core claims

OOD detection identifies inputs that deviate from the training distribution and
should be flagged as "unknown". The paper distinguishes two shift types:

- **Covariate shift** — same classes as ID train but altered noise / acquisition factors.
- **Semantic shift** — entirely novel, unseen classes (the harder regime, and the one studied here).

**Central claims the thesis extends or challenges:**

1. The majority of state-of-the-art modality-agnostic OOD methods (MSP, ODIN, EBO, ReAct, DICE, GradNorm) perform **poorly** on time-series data.
2. OOD methods based on **deep feature modeling (DFM/MDS)** are more effective for time series and represent the most promising direction.
3. Semantic OOD evaluated *across different datasets* is too easy for multivariate TS (origins differ drastically). The realistic, harder protocol draws **ID and OOD classes from the same dataset** — the convention this work adopts.
4. ID-vs-OOD performance correlation (strong in vision) is **weaker** for classification-based TS OOD methods; distance/density methods (DFM, MDS, EBO) correlate more strongly with ID accuracy.

## 3. Evaluation datasets

Thirteen multivariate datasets from the **UCR/UEA** repository (Dau et al. 2019).
Half of the classes (first half) = **ID**, second half = **OOD**.

| Dataset | Abbr | Train | Test | Dims | Length | Classes |
|---|---|---|---|---|---|---|
| ArticularyWordRecognition | AWR | 275 | 300 | 9 | 144 | 25 |
| Epilepsy | EP | 137 | 138 | 3 | 206 | 4 |
| EthanolConcentration | EC | 261 | 263 | 3 | 1751 | 4 |
| HandMovementDirection | HMD | 160 | 74 | 10 | 400 | 4 |
| Handwriting | HW | 150 | 850 | 3 | 152 | 26 |
| Libras | LIB | 180 | 180 | 2 | 45 | 15 |
| LSST | LSST | 2459 | 2466 | 6 | 36 | 14 |
| NATOPS | NATO | 180 | 180 | 24 | 51 | 6 |
| PEMS-SF | PEMS | 267 | 173 | 963 | 144 | 7 |
| PenDigits | PD | 7494 | 3498 | 2 | 8 | 10 |
| PhonemeSpectra | PS | 3315 | 3353 | 11 | 217 | 39 |
| RacketSports | RS | 151 | 152 | 6 | 30 | 4 |
| UWaveGestureLibrary | UW | 120 | 320 | 3 | 315 | 8 |

## 4. OOD scoring functions evaluated

Eight post-hoc, modality-agnostic detectors spanning three families:

- **Classification-based:** MSP (max softmax prob; low = OOD), ODIN (temperature scaling + input perturbation), ReAct (rectified/clipped activations), DICE (activation sparsification).
- **Density / energy-based:** EBO (energy from logits; high energy = OOD).
- **Distance / feature-based:** Mahalanobis / MDS (class-conditional Gaussians, tied covariance), DFM (feature reconstruction error via PCA; variants DFM-PCA, DFM-IF, DFM-OCSVM). Features are taken from the **pre-logit layer** (empirically best).
- **Gradient-based:** GradNorm (L1 norm of gradients of KL(softmax ‖ uniform)).

For multivariate TS (no large pretrained backbones), MDS and DFM backbones are
trained **on ID-class training data only**.

### Detection metrics

After binarising test labels (ID vs OOD), the paper reports:

- **AUROC** — Area Under the ROC curve (primary).
- **AUPR** — Area Under the Precision-Recall curve.

Both are chosen for **threshold independence**. (The benchmark1 extension additionally
reports **FPR@95** — false-positive rate at 95% TPR, threshold tuned on a validation split.)

## 5. Augmentation strategies

Augmentations matter because they shape the contrastive backbone whose features feed
OOD scoring. Seven are compared: **Jittering, Permutation, Magnitude Warping,
Window Warping, Resizing (crop+resize), Flipping, Time Masking**. On average
**Magnitude Warping** is best (AUROC 0.634), then Permutation (0.619); ranking is
highly dataset-dependent.

## 6. Backbones and loss functions

- **Backbones:** 1-D ResNet (3 residual blocks × 3 conv layers + GAP + softmax) — *best on average*; Transformer/TST (3 encoder layers + classification head); LSTM (2 layers + FC) — *clearly worst*.
- **Losses:** Cross-Entropy (CE) and **Multi-Positive Contrastive (MPC)** loss. MPC outperforms CE by **+4.3% AUROC / +4.1% AUPR** on average, with the largest gains on activation-based methods (ReAct, DICE).

## 7. Train / val / test split conventions

- Classes split: first half = ID, second half = OOD.
- Backbones and OOD models trained **only on ID training data**; OOD models never see/tune on OOD data.
- Test set = **50/50 mixture** of unseen ID-class samples and unseen OOD-class samples.
- OOD models are evaluated, not tuned, on OOD.

## 8. Notation and terminology to reuse in the thesis

- `D = {(X_0, y_0), …, (X_{2n}, y_{2n})}` — full labelled dataset; `D_ID`, `D_OOD` derived halves.
- `ID` / `OOD`, "semantic shift" vs "covariate shift", "modality-agnostic OOD detection", "backbone-agnostic loss".
- "Pre-logit layer features", "feature reconstruction error", "class-conditional Gaussians with tied covariance".
- Method abbreviations: MSP, ODIN, EBO, GradNorm, ReAct, DICE, MDS (Mahalanobis), DFM (DFM-PCA / DFM-IF / DFM-OCSVM).
- Loss abbreviations: CE, MPC (temperature `τ`, ℓ2-normalised anchors/candidates).

## 9. How the thesis extends this work

The benchmark1 codebase (this thesis) extends `Gungor2025` along several axes documented
in `THESIS_FINDINGS.md`:

1. **New data regime:** TSB-StreamingAD (300 univariate + 300 multivariate streaming files) where each file concatenates two source recordings — a *streaming distributional-shift* setting beyond the static UCR/UEA semantic-shift protocol.
2. **Wider method pool:** ~24 implemented detectors vs the paper's 8, adding TS-specific and drift/changepoint methods (CODiT, SRS, DIVERSIFY, InvAD, M2N2, DiMMAD, DEEDEE, TD-IVDM, CatSight, AE-ADWIN-LSTM, DriftLens, DiffAD).
3. **Protocol contributions:** source-boundary training split (avoid Source-2 contamination); per-series vs global normalisation as a function of the OOD signal type (morphological vs level shift); empirical demonstration of the **softmax overconfidence dichotomy** at scale (183 TSB-U datasets).
4. **Confirms the paper's central thesis:** distribution/feature-based methods (Mahalanobis, DFM-PCA, DriftLens) dominate softmax/logit methods on streaming TS — consistent with `Gungor2025`'s recommendation of deep feature modeling.
