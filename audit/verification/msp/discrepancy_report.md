# Discrepancy Report — `msp` (Maximum Softmax Probability)

**Author:** Stylianos Giannoulis · Aristotle University of Thessaloniki, MSc Data and Web Science · Supervisor: John Paparrizos
**Method id:** `msp` · **Citation:** Hendrycks & Gimpel, 2017 (arXiv:1610.02136)
**Local implementation:** `benchmark1/models/ood_methods/msp.py` (`MSPDetector`)
**Reference:** `methods/msp/reference/` (hendrycks/error-detection — original MSP baseline)

## Component comparison

| Component | Finding | Severity | Recommendation |
|---|---|---|---|
| Model architecture | MSP is backbone-agnostic and post-hoc; it consumes classifier logits only. Local implementation routes through the shared ResNet/TST/LSTM backbone + classification head via `BaseOODDetector._logits_and_input`. Matches. | — | None |
| OOD scoring function | Reference defines confidence as `max_c softmax(logits)_c` (higher = more ID). Local computes `1 − max_c softmax(logits/T)_c` (higher = more OOD). The negation is an orientation convention to keep "higher = more anomalous" project-wide; it is a monotone-decreasing transform of the reference score and therefore **AUROC/AUPR-invariant**. Matches. | MINOR | Keep; document the orientation convention once in the thesis methods section. |
| Temperature | Reference vanilla MSP uses no temperature (T=1). Local exposes `temperature` (default 1.0). With the default, behaviour is identical to vanilla MSP; T≠1 would make it a temperature-scaled variant (closer to ODIN without input perturbation). Matches at default. | MINOR | Keep default T=1 for the canonical MSP baseline. |
| Training procedure | MSP requires no training of its own; it reuses the ID-trained backbone. Matches the reference (post-hoc). | — | None |
| Data preprocessing | Inherited from the shared pipeline (windowing/normalisation per dataset config); reference applies dataset-specific image normalisation. Domain-appropriate difference. | — | None |
| Evaluation protocol | AUROC/AUPR via scikit-learn (`eval/metrics.py`); FPR@95 custom. Reference uses its own `calculate_log` routines but the same threshold-free metrics. Matches. | — | None |

## Summary

The local `MSPDetector` is a faithful, minimal reproduction of the maximum-softmax-probability
baseline of Hendrycks & Gimpel (2017). The only deviation from the reference is a sign
convention — the score is reported as one minus the maximum softmax probability so that larger
values denote greater anomalousness — which is a strictly monotone transformation and leaves
all threshold-free detection metrics unchanged. The optional temperature parameter defaults to
unity, recovering the canonical baseline exactly. No corrective `_enh` variant is required.
