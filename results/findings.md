# TSB-StreamingAD Benchmark — Findings

Datasets evaluated: 527 (U: 260, M: 267). Detectors: 17. Completed runs: 8668.

## Overall ranking (mean AUROC across all datasets)

             mean_auroc  std_auroc  mean_aupr  mean_fpr95  mean_inference_ms    n
method                                                                           
mahalanobis       0.826      0.228      0.844       0.366              1.871  527
dfm_pca           0.812      0.232      0.826       0.390              3.502  515
dimmad            0.747      0.260      0.776       0.492              3.166  527
invad             0.740      0.259      0.766       0.490              2.088  527
catsight          0.735      0.295      0.779       0.495              1.675  515
m2n2              0.733      0.276      0.776       0.512              2.171  527
deedee            0.700      0.271      0.749       0.512              1.036  527
codit             0.684      0.242      0.731       0.676             27.843  527
diffad            0.531      0.173      0.597       0.842             21.663  527
msp               0.370      0.284      0.540       0.877              1.878  527
srs               0.352      0.304      0.527       0.821             15.893  260
dice              0.313      0.305      0.494       0.877              1.756  527
scale             0.308      0.303      0.487       0.878              1.843  527
odin              0.303      0.299      0.495       0.884              8.016  527
react             0.303      0.299      0.487       0.886              1.810  527
energy            0.301      0.301      0.486       0.886              1.949  527
gradnorm          0.294      0.301      0.483       0.886             11.790  527

## TSB-StreamingAD-U ranking

method
mahalanobis    0.865
dfm_pca        0.852
deedee         0.810
dimmad         0.798
catsight       0.787
invad          0.781
m2n2           0.773
codit          0.751
diffad         0.523
srs            0.352
msp            0.305
dice           0.291
scale          0.256
odin           0.256
energy         0.253
react          0.253
gradnorm       0.241

## TSB-StreamingAD-M ranking

method
mahalanobis    0.788
dfm_pca        0.774
invad          0.700
dimmad         0.697
m2n2           0.695
catsight       0.686
codit          0.618
deedee         0.592
diffad         0.539
msp            0.433
scale          0.359
react          0.351
odin           0.349
energy         0.348
gradnorm       0.345
dice           0.333

## Mean AUROC by category (DRIFT / OOD / STABLE)

category     DRIFT    OOD  STABLE
method                           
catsight     0.749  0.690   0.767
codit        0.725  0.635   0.688
deedee       0.771  0.598   0.724
dfm_pca      0.825  0.762   0.849
dice         0.347  0.295   0.294
diffad       0.510  0.519   0.565
dimmad       0.735  0.704   0.801
energy       0.325  0.318   0.260
gradnorm     0.313  0.312   0.255
invad        0.733  0.709   0.777
m2n2         0.749  0.698   0.751
mahalanobis  0.858  0.761   0.855
msp          0.357  0.403   0.350
odin         0.326  0.317   0.266
react        0.329  0.317   0.261
scale        0.332  0.316   0.276
srs          0.282  0.406   0.372

## Mean AUROC by normalization (global vs per_series) — dichotomy test

normalize    global  per_series
method                         
catsight      0.797       0.630
codit         0.723       0.616
deedee        0.704       0.691
dfm_pca       0.863       0.726
dice          0.227       0.461
diffad        0.540       0.516
dimmad        0.832       0.600
energy        0.214       0.452
gradnorm      0.213       0.433
invad         0.827       0.588
m2n2          0.787       0.641
mahalanobis   0.867       0.755
msp           0.309       0.475
odin          0.218       0.451
react         0.214       0.457
scale         0.218       0.464
srs           0.296       0.480

## Methods failing under streaming (mean AUROC <= 0.5, i.e. at/below chance or inverted)

msp (0.370), srs (0.352), dice (0.313), scale (0.308), odin (0.303), react (0.303), energy (0.301), gradnorm (0.294)

## Statistical test

- Friedman: chi2=1674.97, p=0 (k=17 methods, N=250 datasets)

## Class-D appendix detectors (mean AUROC by arm)

Rows: 1731 across 7 detectors (ae_adwin_lstm, diversemix, diversify, divoe, driftlens, outlier_exposure, tdivdm).

                                  mean_auroc  std_auroc    n
method           arm                                        
tdivdm           none                  0.662      0.181  161
diversemix       head_only             0.620      0.275  158
diversify        cosine_centroid       0.587      0.266  158
driftlens        none                  0.573      0.368  146
divoe            full_net              0.572      0.233  158
outlier_exposure full_net              0.564      0.232  158
ae_adwin_lstm    none                  0.556      0.211  160
diversemix       full_net              0.550      0.246  158
outlier_exposure head_only             0.442      0.259  158
divoe            head_only             0.438      0.258  158
diversify        energy                0.398      0.269  158
