# Datasets

## TSB-StreamingAD (primary)
- `TSB-StreamingAD-U` (univariate) and `TSB-StreamingAD-M` (multivariate), 300 files each,
  stratified across **DRIFT / OOD / STABLE** categories.
- The headline benchmark uses the **U** split at full coverage (`--scale full`, ~183 of 300 load;
  the rest are skipped for size/errors), matching the prior full-budget run.

## UCR / UEA (secondary)
- The general archive loaders live in `data/ucr_loader.py` (the original TS-OOD replication path).

## Not committed
Raw data is **not** in the repo. Fetch it with the scripts in `data/download/`, then point the
loaders at it:
```bash
python runners/run.py --scale full --data-root /path/to/datasets   # or set TSB_DATA_ROOT
```
Expected layout: `<data-root>/TSB-StreamingAD-U/*.csv`, `<data-root>/TSB-StreamingAD-M/*.csv`.

## Auxiliary-outlier corpus (class-D only)
The OE-family appendix detectors need outliers. `data/aux_outliers.py` partitions the corpus into
disjoint **eval** vs **aux** sets via a persisted manifest (`aux_manifest.json`, seed 42) so no aux
file is ever an eval file — see [`../audit/CLASS_D_DECISIONS.md`](../audit/CLASS_D_DECISIONS.md).
