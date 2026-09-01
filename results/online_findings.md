# Online / incremental-covariance Mahalanobis under true streaming

_Generated 2026-09-01T14:36:16 - seed 42, splits U, 10 files/category, online decay=0.999._

Per-window AUROC/AUPR/FPR95 on the temporally-ordered evaluation stream (`load_tsb(ordered_eval=True)`). **batch** = tied covariance fit once on the ID training windows then frozen; **online** = same warm-start, then mean+covariance updated incrementally after each streamed window (score-then-update, unsupervised, exponential forgetting). Every number below is traceable to `results/online_incremental.csv`.

## Mean AUROC: batch vs online

| Scope | n datasets | batch AUROC | online AUROC | delta (online-batch) |
|---|---|---|---|---|
| **Overall** | 29 | 0.6938 | 0.4278 | -0.2660 |
| DRIFT | 10 | 0.6814 | 0.4476 | -0.2338 |
| OOD | 10 | 0.7658 | 0.4789 | -0.2869 |
| STABLE | 9 | 0.6278 | 0.3492 | -0.2786 |

## Mean AUPR / FPR95 (overall)

| Metric | batch | online |
|---|---|---|
| AUPR (higher better) | 0.6978 | 0.5398 |
| FPR@95 (lower better) | 0.6832 | 0.8754 |

## Online vs batch, per-dataset AUROC (tolerance +/-0.005)

- online **wins**: 1
- **ties**: 2
- online **loses**: 26

**Verdict (streaming-deployment claim):** online incremental Mahalanobis **trails** the fit-once batch detector on mean AUROC (delta -0.2660).

_Total wall-clock runtime: 18.4 min (29 datasets x 2 variants)._

