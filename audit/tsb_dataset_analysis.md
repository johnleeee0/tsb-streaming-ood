# TSB-StreamingAD Dataset Analysis

*Quantitative characterization of the two target benchmark families for the MSc thesis "Benchmarking OOD Methods for Time Series" (S. Giannoulis, AUTh; supervisor J. Paparrizos).*

> All statistics below are computed directly from the CSV files. **Every one of the 600 files was processed (no sampling):** 300 files in TSB-StreamingAD-U and 300 in TSB-StreamingAD-M. The per-file raw table is saved at `C:/THESIS/experiments/tsb_file_stats.csv`.

## 1. What the two families contain

**TSB-StreamingAD-U (univariate)** and **TSB-StreamingAD-M (multivariate)** are families of long *streaming* time-series files derived from the TSB-StreamingAD anomaly-detection benchmark. Each CSV has one or more numeric feature columns followed by a final binary `Label` column (0 = normal, 1 = anomalous).

**Two-source concatenation.** Each file is built by concatenating **two source recordings**. Source 1 (S1) is a normal recording; Source 2 (S2) introduces a distribution shift and is labelled anomalous. The **first anomalous timestep marks the S1/S2 boundary** — all S1 timesteps are normal by construction. The filename encodes the construction, e.g. `DRIFT_003_SEG_..._<SRC1>_..._AND_<SRC2>_...`.

**Window-level OOD definition.** A sliding window of fixed size/stride (default 64/32; configs also use 128/64) is swept over the series. A window is **ID (label 0)** if *all* its timesteps are normal and **OOD (label 1)** if it contains *any* anomalous timestep. Normal S1 windows are the in-distribution data; OOD windows come from the shifted S2 region. (See `datasets/tsb_loader.py`.)

**Source-boundary training split.** With `boundary_split=True` (default) the backbone is trained **only on Source-1 normal windows** — those that end strictly before the boundary row — using `train_frac` (default 0.70) of that pool. Held-out S1 normals form the ID evaluation windows and all anomalous windows form the OOD evaluation windows; the evaluation pool is balanced 1:1 and split 50/50 into val/test. This prevents S2 signal from leaking into training, which would otherwise collapse AUROC toward 0.5.

**Streaming characteristics.** Series are long (thousands to hundreds of thousands of timesteps), with a single shift event per file rather than scattered point anomalies; the anomalous (S2) region is a contiguous tail. Normalisation follows the rule *Medical/HumanActivity -> per_series, else global* per the registry.

## 2. Counts

### 2.1 Files per split

| split | n_files |
| --- | --- |
| M | 300 |
| U | 300 |

### 2.2 Files per split x category

| split | DRIFT | OOD | STABLE |
| --- | --- | --- | --- |
| M | 100 | 100 | 100 |
| U | 100 | 100 | 100 |

### 2.3 Files per split x domain

| split | Environment | Facility | HumanActivity | Medical | Sensor | Synthetic | Unknown | WebService |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M | 0 | 116 | 2 | 135 | 47 | 0 | 0 | 0 |
| U | 3 | 31 | 13 | 74 | 2 | 26 | 1 | 150 |

### 2.4 Files per category x domain

| category | Environment | Facility | HumanActivity | Medical | Sensor | Synthetic | Unknown | WebService |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DRIFT | 0 | 15 | 4 | 111 | 0 | 8 | 0 | 62 |
| OOD | 2 | 99 | 8 | 52 | 1 | 2 | 0 | 36 |
| STABLE | 1 | 33 | 3 | 46 | 48 | 16 | 1 | 52 |

## 3. Distribution summaries (min / median / max)

Anomaly rate = fraction of timesteps labelled 1 (i.e. relative size of the S2 tail). Boundary fraction = position of the first anomalous timestep as a fraction of series length (higher = larger S1 training region).

### 3.1 By split

| split | n_files | length_min | length_med | length_max | channels_min | channels_med | channels_max | anomaly_rate_min | anomaly_rate_med | anomaly_rate_max | boundary_frac_min | boundary_frac_med | boundary_frac_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M | 300 | 3399 | 329198 | 880400 | 2 | 19 | 248 | 0.29% | 5.23% | 15.03% | 0.000 | 0.181 | 0.624 |
| U | 300 | 2842 | 20060 | 1030400 | 1 | 1 | 1 | 0.00% | 0.53% | 31.12% | 0.000 | 0.169 | 0.928 |

### 3.2 By split x category

| split | category | n_files | length_min | length_med | length_max | channels_min | channels_med | channels_max | anomaly_rate_min | anomaly_rate_med | anomaly_rate_max | boundary_frac_min | boundary_frac_med | boundary_frac_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M | DRIFT | 100 | 180400 | 430400 | 880400 | 2 | 2 | 2 | 0.29% | 4.47% | 15.03% | 0.000 | 0.101 | 0.624 |
| M | OOD | 100 | 59286 | 329198 | 329198 | 3 | 19 | 25 | 2.27% | 5.48% | 14.79% | 0.002 | 0.298 | 0.585 |
| M | STABLE | 100 | 3399 | 89718 | 600000 | 2 | 22 | 248 | 0.52% | 2.65% | 12.62% | 0.000 | 0.167 | 0.595 |
| U | DRIFT | 100 | 6421 | 19916 | 689999 | 1 | 1 | 1 | 0.07% | 0.50% | 15.04% | 0.000 | 0.156 | 0.822 |
| U | OOD | 100 | 2848 | 31646 | 1030400 | 1 | 1 | 1 | 0.00% | 0.92% | 31.12% | 0.000 | 0.169 | 0.928 |
| U | STABLE | 100 | 2842 | 10000 | 338340 | 1 | 1 | 1 | 0.05% | 0.45% | 12.61% | 0.000 | 0.188 | 0.923 |

## 4. Per-category notes (DRIFT / OOD / STABLE)

- **DRIFT** — Source 2 is a *related but drifted* recording (e.g. same modality, gradually changed statistics). The shift is subtle: window representations of S2 overlap substantially with S1, so detectors must separate near-distribution drift. *Expected difficulty: medium-high.*

- **OOD** — Source 2 is a *clearly different* recording / regime (strong out-of-distribution shift). Window features should be more separable from the S1 manifold. *Expected difficulty: low-medium (the easier, more clearly separable category).*

- **STABLE** — the two sources are *similar / stationary* with little genuine shift at the boundary. These act largely as **negative / control** cases: a well-calibrated detector should NOT over-flag them, so high false-positive methods are penalised here. *Expected difficulty: high for distinguishing real OOD, useful as a specificity check.*

## 5. Why these are the right benchmark target

These two families are the thesis's streaming-time-series OOD target because they (i) provide **labelled, window-level OOD ground truth** under a single, reproducible protocol shared across all registered detectors; (ii) span a controlled difficulty axis — STABLE (no/low shift, specificity control), DRIFT (subtle near-distribution shift), and OOD (strong shift) — letting us measure not just average AUROC but *where* a method's separation power and calibration break down; and (iii) cover **both univariate (-U) and multivariate (-M)** regimes across multiple application domains, testing whether OOD scores generalise across channel counts and modalities. The source-boundary split makes the task honest by forbidding S2 leakage, so AUROC reflects true generalisation rather than memorisation. Expected difficulty is graded: OOD files should yield the highest AUROC, DRIFT the most informative spread, and STABLE the strongest test of false-positive control — collectively a demanding and discriminative benchmark for OOD detectors on time series.
