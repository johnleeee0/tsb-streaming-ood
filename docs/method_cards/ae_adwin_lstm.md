# AE-ADWIN-LSTM (`ae_adwin_lstm`)

- **Category:** class-D appendix
- **Paper:** A Novel Concept Drift Detection Model for Handling Evolving Patterns in Multivariate Time Series — IEEE APCI 2025 (doi:10.1109/APCI65531.2025.11136854). (The registry title is wrong; the code title is correct.)
- **Official code:** none public (no repository found; paper paywalled on IEEE Xplore)
- **Fidelity verdict:** NOT-THE-METHOD — the production variant's temporal components are non-functional on shuffled windows and its orientation is inverted. A faithful ordered-stream class-D build fixes every defect.
- **Core idea:** A hybrid autoencoder + LSTM predictor + ADWIN drift detector on an **ordered** stream: ADWIN monitors LSTM prediction residuals and, when drift fires, the model is incrementally updated. The paper's contribution is that drift-triggered update loop.
- **Key parameters (class-D build):** ordered per-window scoring on `load_tsb(ordered_eval=True)`; real ADWIN (exponential-histogram buckets, all cut points, Hoeffding bound); drift-triggered light gradient step; ADWIN reset per stream; score composition `0.4·recon + 0.4·pred + 0.2·drift`, un-negated, higher = OOD. Secondary metric: drift-detection delay.
- **Divergences from original / caveats:**
  - Production `ae_adwin_lstm.py` runs on a **randomly permuted** eval set, so the LSTM history and ADWIN error stream are noise (two shuffles agree on only ~80% of the ranking); "ADWIN" is a single midpoint two-sample mean test, not ADWIN; the drift-triggered incremental update (the paper's contribution) is absent; and the score is negated on a premise the docstring itself contradicts. Mean AUROC 0.253 as implemented vs 0.747 flipped; a ~22% positional artefact affects the first `seq_len−1` windows.
  - The class-D build fixes all six: ordered stream, real ADWIN, incremental update, ADWIN reset, positional artefact, and orientation. Still on frozen features (disclosed); per-window AUROC noisy on short streams.
  - Low verification confidence: no public code and a paywalled paper.
- **Where it runs:** `models/detectors/class_d/ae_adwin_lstm.py` (`--group class_d`)
