# Extended analysis findings

## Nemenyi / critical-difference
- Complete-case Friedman: chi2=2336.59, p=0.000e+00 (k=16, N=515).
- Critical difference (alpha=0.05): CD=1.016 mean-rank units.
- Mean ranks (1=best):
    - mahalanobis: 4.353
    - dfm_pca: 4.794
    - invad: 6.491
    - dimmad: 6.537
    - catsight: 6.641
    - deedee: 6.657
    - m2n2: 6.714
    - codit: 7.336
    - diffad: 9.205
    - msp: 10.380
    - scale: 10.996
    - react: 11.091
    - dice: 11.095
    - odin: 11.173
    - energy: 11.211
    - gradnorm: 11.326
- Top feature methods vs post-hoc cluster (Nemenyi p-values):
    - mahalanobis: msp=1.11e-16*, odin=1.11e-16*, energy=1.11e-16*, react=1.11e-16*, dice=1.11e-16*, scale=1.11e-16*, gradnorm=1.11e-16*
    - dfm_pca: msp=1.11e-16*, odin=1.11e-16*, energy=1.11e-16*, react=1.11e-16*, dice=1.11e-16*, scale=1.11e-16*, gradnorm=1.11e-16*
    - invad: msp=1.11e-16*, odin=1.11e-16*, energy=1.11e-16*, react=1.11e-16*, dice=1.11e-16*, scale=1.11e-16*, gradnorm=1.11e-16*
    - dimmad: msp=1.11e-16*, odin=1.11e-16*, energy=1.11e-16*, react=1.11e-16*, dice=1.11e-16*, scale=1.11e-16*, gradnorm=1.11e-16*
    - catsight: msp=1.11e-16*, odin=1.11e-16*, energy=1.11e-16*, react=1.11e-16*, dice=1.11e-16*, scale=1.11e-16*, gradnorm=1.11e-16*

## Per-domain mean AUROC by family

| domain | N | feature-manifold | post-hoc | ts-specific |
|---|---|---|---|---|
| WSD | 97 | 0.897 | 0.165 | 0.562 |
| MITDB | 82 | 0.654 | 0.488 | 0.566 |
| GHL | 76 | 0.735 | 0.258 | 0.534 |
| UCR | 58 | 0.769 | 0.316 | 0.579 |
| YAHOO | 43 | 0.762 | 0.350 | 0.473 |
| SMD | 33 | 0.937 | 0.063 | 0.651 |
| SVDB | 27 | 0.612 | 0.507 | 0.549 |
| SMAP | 23 | 0.770 | 0.259 | 0.663 |
| LTDB | 22 | 0.644 | 0.463 | 0.498 |
| PSM | 18 | 0.294 | 0.787 | 0.339 |
| NAB | 10 | 0.838 | 0.269 | 0.534 |
| MSL | 9 | 0.878 | 0.130 | 0.578 |
| Exathlon | 7 | 0.921 | 0.091 | 0.697 |
| Genesis | 6 | 1.000 | 0.000 | 0.996 |
| IOPS | 6 | 0.812 | 0.273 | 0.467 |
| MGAB | 5 | 0.665 | 0.360 | 0.569 |
| Other | 5 | 0.591 | 0.408 | 0.579 |

## Failure cases

- Datasets where even the best feature-manifold detector is below chance (AUROC < 0.5): 23.
    - TSB-U-STABLE_098_SEG_1680_686_YAHOO_id_136_Web: best feature AUROC 0.250 (src=YAHOO)
    - TSB-M-STABLE_086_SEG_8640_147_SMAP_id_4_Sensor: best feature AUROC 0.375 (src=SMAP)
    - TSB-U-STABLE_046_SEG_1421_552_YAHOO_id_2_Synth: best feature AUROC 0.375 (src=YAHOO)
    - TSB-M-OOD_017_SEG_217624_115_PSM_id_1_Facility: best feature AUROC 0.452 (src=PSM)
    - TSB-M-OOD_034_SEG_217624_115_PSM_id_1_Facility: best feature AUROC 0.452 (src=PSM)
    - TSB-M-OOD_025_SEG_217624_115_PSM_id_1_Facility: best feature AUROC 0.452 (src=PSM)
    - TSB-M-OOD_026_SEG_217624_115_PSM_id_1_Facility: best feature AUROC 0.452 (src=PSM)
    - TSB-M-OOD_028_SEG_217624_115_PSM_id_1_Facility: best feature AUROC 0.452 (src=PSM)
    - TSB-M-OOD_064_SEG_217624_115_PSM_id_1_Facility: best feature AUROC 0.452 (src=PSM)
    - TSB-M-OOD_048_SEG_217624_115_PSM_id_1_Facility: best feature AUROC 0.452 (src=PSM)
    - TSB-M-OOD_056_SEG_217624_115_PSM_id_1_Facility: best feature AUROC 0.452 (src=PSM)
    - TSB-M-OOD_062_SEG_217624_115_PSM_id_1_Facility: best feature AUROC 0.452 (src=PSM)
    - TSB-M-OOD_075_SEG_217624_115_PSM_id_1_Facility: best feature AUROC 0.452 (src=PSM)
    - TSB-M-OOD_032_SEG_217624_115_PSM_id_1_Facility: best feature AUROC 0.452 (src=PSM)
    - TSB-M-OOD_039_SEG_217624_115_PSM_id_1_Facility: best feature AUROC 0.452 (src=PSM)
- Datasets where a post-hoc detector unexpectedly exceeds 0.7: 106.
    - TSB-M-OOD_007_SEG_200001_036_GHL_id_5_Sensor_t: best post-hoc AUROC 1.000 (src=GHL)
    - TSB-M-OOD_001_SEG_200001_036_GHL_id_5_Sensor_t: best post-hoc AUROC 1.000 (src=GHL)
    - TSB-M-OOD_013_SEG_200001_036_GHL_id_5_Sensor_t: best post-hoc AUROC 1.000 (src=GHL)
    - TSB-M-OOD_019_SEG_200001_036_GHL_id_5_Sensor_t: best post-hoc AUROC 1.000 (src=GHL)
    - TSB-M-STABLE_002_SEG_200001_036_GHL_id_5_Senso: best post-hoc AUROC 1.000 (src=GHL)
    - TSB-U-OOD_041_SEG_18236_084_WSD_id_56_WebServi: best post-hoc AUROC 1.000 (src=WSD)
    - TSB-U-OOD_051_SEG_18236_107_WSD_id_79_WebServi: best post-hoc AUROC 1.000 (src=WSD)
    - TSB-U-DRIFT_034_SEG_18236_137_WSD_id_109_WebSe: best post-hoc AUROC 1.000 (src=WSD)
    - TSB-U-DRIFT_014_SEG_18236_050_WSD_id_22_WebSer: best post-hoc AUROC 1.000 (src=WSD)
    - TSB-M-STABLE_028_SEG_8294_152_SMAP_id_9_Sensor: best post-hoc AUROC 1.000 (src=SMAP)
    - TSB-U-STABLE_085_SEG_1680_561_YAHOO_id_11_WebS: best post-hoc AUROC 1.000 (src=YAHOO)
    - TSB-U-STABLE_004_SEG_1680_561_YAHOO_id_11_WebS: best post-hoc AUROC 1.000 (src=YAHOO)
    - TSB-U-STABLE_003_SEG_1680_580_YAHOO_id_30_WebS: best post-hoc AUROC 1.000 (src=YAHOO)
    - TSB-U-OOD_100_SEG_18236_067_WSD_id_39_WebServi: best post-hoc AUROC 1.000 (src=WSD)
    - TSB-U-STABLE_042_SEG_1680_637_YAHOO_id_87_WebS: best post-hoc AUROC 1.000 (src=YAHOO)
