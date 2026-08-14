# Test-set sample-size metamorphic consistency equivalence

## Research question

If only 50% of the test dataset were used for metamorphic assessment, would the observed parameter-dependent consistency patterns be equivalent to using the full test dataset?

## Design

The analysis covers 5 models, 6 rules, and 51 settings, drawn entirely from the `test` dataset (363 images).

The `sample` arm uses a single fixed, reproducibly-drawn subsample containing 182 images (50% of 363, rounded to the nearest image). The same subsample is reused across every model, rule, and setting.

AP and AR are COCO original-vs-transformed consistency metrics: each model's original predictions act as the reference, rather than labelled ground truth. Every rule is normalised against the same gamma=1.0 control (`imgDarkBright_1`).

Each arm is independently resampled 2,000 times with replacement from within its own population: the `full` arm resamples from all 363 images, while the `sample` arm resamples only from its 182-image subsample. Results use percentile 90% confidence intervals and a symmetric +/-0.05 equivalence margin. No p-values or multiple-testing corrections are used.

Classification rules:

- **Equivalent:** the complete 90% interval lies within [-0.05, +0.05].
- **Meaningfully different:** the complete interval lies beyond one equivalence boundary.
- **Inconclusive:** the interval overlaps an equivalence boundary.

A positive difference means full-dataset degradation is greater than the 50% subsample; a negative difference means the subsample degradation is greater.

## Validation

The implementation recomputed 510 point AP/AR values for the `full` arm over every declared image and compared them with the existing aggregate JSON for the source dataset. 510 values agreed within 1e-09; 0 differed because the stored evaluation omitted images with zero transformed detections.

The maximum stored-versus-recomputed discrepancy was 3.33e-16 at `yolov8/imgDarkBright_3/AR`. The bootstrap uses the recomputed all-image values; it does not reproduce the biased detection-positive-only subset.

## Key findings

- At parameter level, 418 of 510 comparisons (82.0%) were equivalent, 92 were inconclusive, and 0 were meaningfully different.
- At rule-curve level, 50 of 50 AUC comparisons were equivalent; 0 were inconclusive and 0 were meaningfully different.
- Every AUC comparison was equivalent for YOLOv5, YOLOv8, YOLOv12, YOLOv26, YOLOv26_v2. Their overall parameter-response patterns therefore transfer from the full test set to the 50% subsample within the +/-0.05 margin.

## Parameter-level overview

| Comparison | Metric | Settings | Equivalent | Inconclusive | Different | Equivalent % |
| --- | --- | --- | --- | --- | --- | --- |
| Test subsample | AP | 255 | 210 | 45 | 0 | 82.4% |
| Test subsample | AR | 255 | 208 | 47 | 0 | 81.6% |

## Rule-level AUC results

AUC uses the complete configured parameter range with its axis scaled to 0-1. Because the response is degradation, a higher AUC means lower consistency across the rule. Grayscale has one setting and therefore has no AUC result.

### AP: full test set vs 50% subsample

| Model | Rule | Full AUC | Sample AUC | Difference | 90% CI | Result |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | Hue rotation | 0.710 | 0.714 | -0.004 | [-0.018, +0.014] | Equivalent |
| YOLOv8 | Hue rotation | 0.712 | 0.717 | -0.005 | [-0.019, +0.012] | Equivalent |
| YOLOv12 | Hue rotation | 0.707 | 0.712 | -0.004 | [-0.018, +0.013] | Equivalent |
| YOLOv26 | Hue rotation | 0.694 | 0.695 | -0.001 | [-0.019, +0.013] | Equivalent |
| YOLOv26_v2 | Hue rotation | 0.707 | 0.706 | +0.001 | [-0.012, +0.016] | Equivalent |
| YOLOv5 | Resolution | 0.096 | 0.092 | +0.004 | [-0.007, +0.016] | Equivalent |
| YOLOv8 | Resolution | 0.100 | 0.100 | +0.000 | [-0.010, +0.014] | Equivalent |
| YOLOv12 | Resolution | 0.101 | 0.099 | +0.002 | [-0.009, +0.016] | Equivalent |
| YOLOv26 | Resolution | 0.158 | 0.157 | +0.001 | [-0.017, +0.016] | Equivalent |
| YOLOv26_v2 | Resolution | 0.142 | 0.138 | +0.004 | [-0.011, +0.018] | Equivalent |
| YOLOv5 | Gamma brightness | 0.228 | 0.219 | +0.009 | [-0.017, +0.036] | Equivalent |
| YOLOv8 | Gamma brightness | 0.227 | 0.218 | +0.009 | [-0.014, +0.035] | Equivalent |
| YOLOv12 | Gamma brightness | 0.223 | 0.212 | +0.011 | [-0.013, +0.037] | Equivalent |
| YOLOv26 | Gamma brightness | 0.253 | 0.242 | +0.011 | [-0.016, +0.033] | Equivalent |
| YOLOv26_v2 | Gamma brightness | 0.264 | 0.249 | +0.015 | [-0.010, +0.040] | Equivalent |
| YOLOv5 | Motion blur | 0.183 | 0.179 | +0.004 | [-0.018, +0.029] | Equivalent |
| YOLOv8 | Motion blur | 0.198 | 0.195 | +0.003 | [-0.021, +0.028] | Equivalent |
| YOLOv12 | Motion blur | 0.198 | 0.192 | +0.006 | [-0.015, +0.030] | Equivalent |
| YOLOv26 | Motion blur | 0.320 | 0.320 | +0.000 | [-0.030, +0.028] | Equivalent |
| YOLOv26_v2 | Motion blur | 0.280 | 0.274 | +0.005 | [-0.022, +0.033] | Equivalent |
| YOLOv5 | Horizontal stretch | 0.693 | 0.696 | -0.003 | [-0.023, +0.020] | Equivalent |
| YOLOv8 | Horizontal stretch | 0.680 | 0.688 | -0.008 | [-0.030, +0.019] | Equivalent |
| YOLOv12 | Horizontal stretch | 0.652 | 0.661 | -0.009 | [-0.033, +0.020] | Equivalent |
| YOLOv26 | Horizontal stretch | 0.682 | 0.686 | -0.004 | [-0.032, +0.020] | Equivalent |
| YOLOv26_v2 | Horizontal stretch | 0.693 | 0.698 | -0.005 | [-0.024, +0.016] | Equivalent |

### AR: full test set vs 50% subsample

| Model | Rule | Full AUC | Sample AUC | Difference | 90% CI | Result |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | Hue rotation | 0.659 | 0.661 | -0.002 | [-0.019, +0.015] | Equivalent |
| YOLOv8 | Hue rotation | 0.667 | 0.670 | -0.002 | [-0.017, +0.014] | Equivalent |
| YOLOv12 | Hue rotation | 0.659 | 0.662 | -0.003 | [-0.019, +0.014] | Equivalent |
| YOLOv26 | Hue rotation | 0.644 | 0.648 | -0.004 | [-0.020, +0.014] | Equivalent |
| YOLOv26_v2 | Hue rotation | 0.660 | 0.662 | -0.002 | [-0.016, +0.014] | Equivalent |
| YOLOv5 | Resolution | 0.088 | 0.082 | +0.007 | [-0.004, +0.018] | Equivalent |
| YOLOv8 | Resolution | 0.095 | 0.089 | +0.005 | [-0.005, +0.017] | Equivalent |
| YOLOv12 | Resolution | 0.095 | 0.088 | +0.007 | [-0.005, +0.019] | Equivalent |
| YOLOv26 | Resolution | 0.149 | 0.148 | +0.001 | [-0.014, +0.016] | Equivalent |
| YOLOv26_v2 | Resolution | 0.132 | 0.127 | +0.004 | [-0.010, +0.018] | Equivalent |
| YOLOv5 | Gamma brightness | 0.205 | 0.194 | +0.011 | [-0.015, +0.036] | Equivalent |
| YOLOv8 | Gamma brightness | 0.207 | 0.195 | +0.013 | [-0.010, +0.036] | Equivalent |
| YOLOv12 | Gamma brightness | 0.203 | 0.187 | +0.016 | [-0.007, +0.041] | Equivalent |
| YOLOv26 | Gamma brightness | 0.229 | 0.220 | +0.009 | [-0.015, +0.032] | Equivalent |
| YOLOv26_v2 | Gamma brightness | 0.236 | 0.223 | +0.014 | [-0.009, +0.037] | Equivalent |
| YOLOv5 | Motion blur | 0.166 | 0.159 | +0.007 | [-0.015, +0.030] | Equivalent |
| YOLOv8 | Motion blur | 0.181 | 0.174 | +0.008 | [-0.016, +0.031] | Equivalent |
| YOLOv12 | Motion blur | 0.183 | 0.172 | +0.011 | [-0.011, +0.034] | Equivalent |
| YOLOv26 | Motion blur | 0.304 | 0.304 | +0.000 | [-0.029, +0.029] | Equivalent |
| YOLOv26_v2 | Motion blur | 0.263 | 0.256 | +0.006 | [-0.022, +0.033] | Equivalent |
| YOLOv5 | Horizontal stretch | 0.666 | 0.667 | -0.001 | [-0.024, +0.022] | Equivalent |
| YOLOv8 | Horizontal stretch | 0.657 | 0.660 | -0.003 | [-0.029, +0.023] | Equivalent |
| YOLOv12 | Horizontal stretch | 0.630 | 0.635 | -0.005 | [-0.032, +0.022] | Equivalent |
| YOLOv26 | Horizontal stretch | 0.656 | 0.663 | -0.007 | [-0.033, +0.020] | Equivalent |
| YOLOv26_v2 | Horizontal stretch | 0.672 | 0.677 | -0.005 | [-0.027, +0.016] | Equivalent |

## Sample-size equivalence summary

| Model | Sample AP | Sample AR | All equivalent AUCs |
| --- | --- | --- | --- |
| YOLOv5 | 5/5 | 5/5 | 10/10 |
| YOLOv8 | 5/5 | 5/5 | 10/10 |
| YOLOv12 | 5/5 | 5/5 | 10/10 |
| YOLOv26 | 5/5 | 5/5 | 10/10 |
| YOLOv26_v2 | 5/5 | 5/5 | 10/10 |

Counts show equivalent rule-level AUC comparisons out of the five rules with numeric parameter curves.

## Largest parameter-level differences

| Target | Metric | Model | Rule | Setting | Difference | 90% CI | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Test subsample | AR | YOLOv5 | Resolution | imgResolution_0 | +0.051 | [+0.001, +0.105] | Inconclusive |
| Test subsample | AP | YOLOv5 | Resolution | imgResolution_0 | +0.045 | [-0.001, +0.099] | Inconclusive |
| Test subsample | AR | YOLOv12 | Resolution | imgResolution_0 | +0.044 | [-0.005, +0.096] | Inconclusive |
| Test subsample | AR | YOLOv26_v2 | Resolution | imgResolution_0 | +0.042 | [-0.003, +0.088] | Inconclusive |
| Test subsample | AR | YOLOv8 | Resolution | imgResolution_0 | +0.041 | [-0.006, +0.089] | Inconclusive |
| Test subsample | AR | YOLOv26 | Resolution | imgResolution_0 | +0.039 | [-0.002, +0.081] | Inconclusive |
| Test subsample | AR | YOLOv12 | Gamma brightness | imgDarkBright_9 | +0.039 | [-0.009, +0.085] | Inconclusive |
| Test subsample | AP | YOLOv26_v2 | Resolution | imgResolution_0 | +0.038 | [-0.004, +0.084] | Inconclusive |
| Test subsample | AP | YOLOv26 | Resolution | imgResolution_0 | +0.037 | [-0.004, +0.076] | Inconclusive |
| Test subsample | AP | YOLOv12 | Resolution | imgResolution_0 | +0.037 | [-0.009, +0.088] | Inconclusive |
| Test subsample | AR | YOLOv12 | Gamma brightness | imgDarkBright_8 | +0.034 | [-0.011, +0.078] | Inconclusive |
| Test subsample | AR | YOLOv12 | Gamma brightness | imgDarkBright_7 | +0.034 | [-0.005, +0.074] | Inconclusive |
| Test subsample | AP | YOLOv8 | Resolution | imgResolution_0 | +0.031 | [-0.013, +0.081] | Inconclusive |
| Test subsample | AP | YOLOv12 | Gamma brightness | imgDarkBright_9 | +0.031 | [-0.017, +0.080] | Inconclusive |
| Test subsample | AR | YOLOv5 | Resolution | imgResolution_1 | +0.027 | [-0.004, +0.060] | Inconclusive |
| Test subsample | AP | YOLOv12 | Gamma brightness | imgDarkBright_7 | +0.027 | [-0.013, +0.071] | Inconclusive |
| Test subsample | AR | YOLOv8 | Gamma brightness | imgDarkBright_9 | +0.026 | [-0.016, +0.069] | Inconclusive |
| Test subsample | AR | YOLOv12 | Motion blur | imgMotionBlur_9 | +0.026 | [-0.014, +0.066] | Inconclusive |
| Test subsample | AP | YOLOv12 | Gamma brightness | imgDarkBright_8 | +0.026 | [-0.019, +0.074] | Inconclusive |
| Test subsample | AR | YOLOv12 | Gamma brightness | imgDarkBright_6 | +0.026 | [-0.009, +0.062] | Inconclusive |
| Test subsample | AP | YOLOv26_v2 | Gamma brightness | imgDarkBright_6 | +0.025 | [-0.012, +0.062] | Inconclusive |
| Test subsample | AR | YOLOv26_v2 | Gamma brightness | imgDarkBright_9 | +0.025 | [-0.016, +0.066] | Inconclusive |
| Test subsample | AR | YOLOv8 | Gamma brightness | imgDarkBright_8 | +0.024 | [-0.016, +0.066] | Inconclusive |
| Test subsample | AR | YOLOv5 | Gamma brightness | imgDarkBright_7 | +0.024 | [-0.016, +0.063] | Inconclusive |
| Test subsample | AP | YOLOv26_v2 | Gamma brightness | imgDarkBright_5 | +0.023 | [-0.009, +0.057] | Inconclusive |

## Response curves

### Grayscale

![Grayscale AP degradation](plots/imgGrayscale_ap_degradation.png)

![Grayscale AR degradation](plots/imgGrayscale_ar_degradation.png)

### Hue rotation

![Hue rotation AP degradation](plots/imgColorWheel_ap_degradation.png)

![Hue rotation AR degradation](plots/imgColorWheel_ar_degradation.png)

### Resolution

![Resolution AP degradation](plots/imgResolution_ap_degradation.png)

![Resolution AR degradation](plots/imgResolution_ar_degradation.png)

### Gamma brightness

![Gamma brightness AP degradation](plots/imgDarkBright_ap_degradation.png)

![Gamma brightness AR degradation](plots/imgDarkBright_ar_degradation.png)

### Motion blur

![Motion blur AP degradation](plots/imgMotionBlur_ap_degradation.png)

![Motion blur AR degradation](plots/imgMotionBlur_ar_degradation.png)

### Horizontal stretch

![Horizontal stretch AP degradation](plots/imgStretch_ap_degradation.png)

![Horizontal stretch AR degradation](plots/imgStretch_ar_degradation.png)

## Interpretation cautions

- Equivalence concerns metamorphic consistency, not absolute ground-truth detection accuracy.
- Unlike a comparison between independent datasets, the `sample` arm's images are a strict subset of the `full` arm's images, so the two bootstrap distributions are correlated rather than independent. The comparison estimates the added sampling variability/bias from using fewer images, not generalisation across distinct datasets.
- Images are the resampling units. No sequence/rosbag metadata was available to model temporal clustering.
- The shared gamma=1.0 control was requested for every rule; rule-specific identity settings were not substituted.
- Parameter-level intervals are estimation-only and are not adjusted for multiple comparisons.
- AUC can be equivalent even when individual settings are inconclusive or different; consult the parameter records in `results.json`.
