# Orientation-stability findings

Per-detector orientation over all completed datasets (inverted = AUROC < 0.5).

| detector | family | N | mean AUROC | inverted frac | flip AUROC | inv frac (global) | inv frac (per_series) |
|---|---|---|---|---|---|---|---|
| mahalanobis | feature-manifold | 527 | 0.826 | 0.095 | 0.174 | 0.105 | 0.078 |
| dfm_pca | feature-manifold | 515 | 0.812 | 0.099 | 0.188 | 0.114 | 0.073 |
| dimmad | feature-manifold | 527 | 0.747 | 0.192 | 0.253 | 0.147 | 0.269 |
| invad | feature-manifold | 527 | 0.740 | 0.211 | 0.260 | 0.159 | 0.301 |
| catsight | feature-manifold | 515 | 0.735 | 0.217 | 0.265 | 0.176 | 0.288 |
| m2n2 | feature-manifold | 527 | 0.733 | 0.226 | 0.267 | 0.186 | 0.295 |
| deedee | feature-manifold | 527 | 0.700 | 0.220 | 0.300 | 0.281 | 0.114 |
| codit | ts-specific | 527 | 0.684 | 0.190 | 0.316 | 0.183 | 0.202 |
| diffad | ts-specific | 527 | 0.531 | 0.493 | 0.469 | 0.476 | 0.523 |
| msp | post-hoc | 527 | 0.370 | 0.584 | 0.630 | 0.632 | 0.503 |
| srs | ts-specific | 260 | 0.352 | 0.669 | 0.648 | 0.713 | 0.570 |
| dice | post-hoc | 527 | 0.313 | 0.731 | 0.687 | 0.793 | 0.622 |
| scale | post-hoc | 527 | 0.308 | 0.681 | 0.692 | 0.775 | 0.518 |
| odin | post-hoc | 527 | 0.303 | 0.694 | 0.697 | 0.787 | 0.534 |
| react | post-hoc | 527 | 0.303 | 0.702 | 0.697 | 0.799 | 0.534 |
| energy | post-hoc | 527 | 0.301 | 0.696 | 0.699 | 0.793 | 0.528 |
| gradnorm | post-hoc | 527 | 0.294 | 0.721 | 0.706 | 0.796 | 0.591 |

## Systematic inversion of the post-hoc cluster

- msp: inverted on 58.4% of datasets; a global sign flip would lift mean AUROC 0.370 -> 0.630.
- dice: inverted on 73.1% of datasets; a global sign flip would lift mean AUROC 0.313 -> 0.687.
- scale: inverted on 68.1% of datasets; a global sign flip would lift mean AUROC 0.308 -> 0.692.
- odin: inverted on 69.4% of datasets; a global sign flip would lift mean AUROC 0.303 -> 0.697.
- react: inverted on 70.2% of datasets; a global sign flip would lift mean AUROC 0.303 -> 0.697.
- energy: inverted on 69.6% of datasets; a global sign flip would lift mean AUROC 0.301 -> 0.699.
- gradnorm: inverted on 72.1% of datasets; a global sign flip would lift mean AUROC 0.294 -> 0.706.

## Regime dependence (global vs per-series inverted fraction)

- msp: 63.2% inverted under global vs 50.3% under per-series (worse under global).
- dice: 79.3% inverted under global vs 62.2% under per-series (worse under global).
- scale: 77.5% inverted under global vs 51.8% under per-series (worse under global).
- odin: 78.7% inverted under global vs 53.4% under per-series (worse under global).
- react: 79.9% inverted under global vs 53.4% under per-series (worse under global).
- energy: 79.3% inverted under global vs 52.8% under per-series (worse under global).
- gradnorm: 79.6% inverted under global vs 59.1% under per-series (worse under global).
