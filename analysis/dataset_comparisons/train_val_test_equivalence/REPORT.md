# Test vs validation/training metamorphic consistency equivalence

## Research question

To what extent are the parameter-dependent metamorphic consistency patterns observed on the test dataset reproduced in the validation and training-sample datasets?

## Design

The analysis covers 5 models, 6 rules, and 51 settings. Each dataset contains 363 images.

AP and AR are COCO original-vs-transformed consistency metrics: each model's original predictions act as the reference, rather than labelled ground truth. Every rule is normalised against the same gamma=1.0 control (`imgDarkBright_1`).

Images were independently resampled 2,000 times within each dataset. Results use percentile 90% confidence intervals and a symmetric +/-0.05 equivalence margin. No p-values or multiple-testing corrections are used.

Classification rules:

- **Equivalent:** the complete 90% interval lies within [-0.05, +0.05].
- **Meaningfully different:** the complete interval lies beyond one equivalence boundary.
- **Inconclusive:** the interval overlaps an equivalence boundary.

A positive difference means test degradation is greater than the alternative dataset; a negative difference means validation/training degradation is greater.

## Validation

The implementation recomputed 1,530 point AP/AR values over every declared image and compared them with the existing aggregate JSONs. 1,140 values agreed within 1e-09; 390 differed because the stored evaluation omitted images with zero transformed detections.

The maximum stored-versus-recomputed discrepancy was 0.0428 at `train_samples/yolov8/imgDarkBright_9/AR`. The bootstrap uses the recomputed all-image values; it does not reproduce the biased detection-positive-only subset.

## Key findings

- At parameter level, 787 of 1,020 comparisons (77.2%) were equivalent, 156 were inconclusive, and 77 were meaningfully different.
- At rule-curve level, 84 of 100 AUC comparisons were equivalent; 9 were inconclusive and 7 were meaningfully different.
- Every AUC comparison was equivalent for YOLOv5, YOLOv8, YOLOv12, YOLOv26. Their overall parameter-response patterns therefore transfer from test to both validation and training samples within the +/-0.05 margin.
- All non-equivalent AUC outcomes were confined to YOLOv26_v2. Its test degradation was generally larger, especially for hue rotation and horizontal stretch.

## Parameter-level overview

| Comparison | Metric | Settings | Equivalent | Inconclusive | Different | Equivalent % |
| --- | --- | --- | --- | --- | --- | --- |
| Training samples | AP | 255 | 197 | 31 | 27 | 77.3% |
| Training samples | AR | 255 | 207 | 34 | 14 | 81.2% |
| Validation | AP | 255 | 190 | 44 | 21 | 74.5% |
| Validation | AR | 255 | 193 | 47 | 15 | 75.7% |

## Rule-level AUC results

AUC uses the complete configured parameter range with its axis scaled to 0-1. Because the response is degradation, a higher AUC means lower consistency across the rule. Grayscale has one setting and therefore has no AUC result.

### AP: test vs validation

| Model | Rule | Test AUC | Alternative AUC | Difference | 90% CI | Result |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | Hue rotation | 0.710 | 0.716 | -0.006 | [-0.018, +0.007] | Equivalent |
| YOLOv8 | Hue rotation | 0.712 | 0.721 | -0.009 | [-0.020, +0.004] | Equivalent |
| YOLOv12 | Hue rotation | 0.707 | 0.711 | -0.003 | [-0.016, +0.011] | Equivalent |
| YOLOv26 | Hue rotation | 0.694 | 0.698 | -0.004 | [-0.019, +0.009] | Equivalent |
| YOLOv26_v2 | Hue rotation | 0.707 | 0.619 | +0.087 | [+0.071, +0.104] | Different |
| YOLOv5 | Resolution | 0.096 | 0.099 | -0.003 | [-0.014, +0.008] | Equivalent |
| YOLOv8 | Resolution | 0.100 | 0.099 | +0.002 | [-0.009, +0.013] | Equivalent |
| YOLOv12 | Resolution | 0.101 | 0.105 | -0.004 | [-0.015, +0.008] | Equivalent |
| YOLOv26 | Resolution | 0.158 | 0.161 | -0.002 | [-0.017, +0.010] | Equivalent |
| YOLOv26_v2 | Resolution | 0.142 | 0.104 | +0.038 | [+0.028, +0.048] | Equivalent |
| YOLOv5 | Gamma brightness | 0.228 | 0.233 | -0.005 | [-0.030, +0.020] | Equivalent |
| YOLOv8 | Gamma brightness | 0.227 | 0.235 | -0.008 | [-0.033, +0.017] | Equivalent |
| YOLOv12 | Gamma brightness | 0.223 | 0.230 | -0.007 | [-0.032, +0.018] | Equivalent |
| YOLOv26 | Gamma brightness | 0.253 | 0.261 | -0.008 | [-0.034, +0.015] | Equivalent |
| YOLOv26_v2 | Gamma brightness | 0.264 | 0.208 | +0.056 | [+0.035, +0.078] | Inconclusive |
| YOLOv5 | Motion blur | 0.183 | 0.170 | +0.014 | [-0.006, +0.032] | Equivalent |
| YOLOv8 | Motion blur | 0.198 | 0.181 | +0.016 | [-0.005, +0.036] | Equivalent |
| YOLOv12 | Motion blur | 0.198 | 0.191 | +0.006 | [-0.015, +0.026] | Equivalent |
| YOLOv26 | Motion blur | 0.320 | 0.316 | +0.004 | [-0.022, +0.027] | Equivalent |
| YOLOv26_v2 | Motion blur | 0.280 | 0.217 | +0.063 | [+0.041, +0.084] | Inconclusive |
| YOLOv5 | Horizontal stretch | 0.693 | 0.682 | +0.012 | [-0.005, +0.029] | Equivalent |
| YOLOv8 | Horizontal stretch | 0.680 | 0.671 | +0.009 | [-0.009, +0.029] | Equivalent |
| YOLOv12 | Horizontal stretch | 0.652 | 0.642 | +0.011 | [-0.009, +0.033] | Equivalent |
| YOLOv26 | Horizontal stretch | 0.682 | 0.669 | +0.013 | [-0.008, +0.034] | Equivalent |
| YOLOv26_v2 | Horizontal stretch | 0.693 | 0.595 | +0.098 | [+0.077, +0.121] | Different |

### AR: test vs validation

| Model | Rule | Test AUC | Alternative AUC | Difference | 90% CI | Result |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | Hue rotation | 0.659 | 0.662 | -0.003 | [-0.016, +0.012] | Equivalent |
| YOLOv8 | Hue rotation | 0.667 | 0.670 | -0.003 | [-0.015, +0.010] | Equivalent |
| YOLOv12 | Hue rotation | 0.659 | 0.657 | +0.002 | [-0.012, +0.016] | Equivalent |
| YOLOv26 | Hue rotation | 0.644 | 0.647 | -0.003 | [-0.017, +0.011] | Equivalent |
| YOLOv26_v2 | Hue rotation | 0.660 | 0.591 | +0.070 | [+0.054, +0.085] | Different |
| YOLOv5 | Resolution | 0.088 | 0.093 | -0.005 | [-0.015, +0.005] | Equivalent |
| YOLOv8 | Resolution | 0.095 | 0.093 | +0.001 | [-0.009, +0.011] | Equivalent |
| YOLOv12 | Resolution | 0.095 | 0.099 | -0.004 | [-0.015, +0.007] | Equivalent |
| YOLOv26 | Resolution | 0.149 | 0.152 | -0.004 | [-0.017, +0.009] | Equivalent |
| YOLOv26_v2 | Resolution | 0.132 | 0.109 | +0.023 | [+0.013, +0.033] | Equivalent |
| YOLOv5 | Gamma brightness | 0.205 | 0.212 | -0.007 | [-0.031, +0.017] | Equivalent |
| YOLOv8 | Gamma brightness | 0.207 | 0.216 | -0.009 | [-0.033, +0.015] | Equivalent |
| YOLOv12 | Gamma brightness | 0.203 | 0.212 | -0.009 | [-0.034, +0.016] | Equivalent |
| YOLOv26 | Gamma brightness | 0.229 | 0.239 | -0.010 | [-0.033, +0.014] | Equivalent |
| YOLOv26_v2 | Gamma brightness | 0.236 | 0.201 | +0.036 | [+0.015, +0.057] | Inconclusive |
| YOLOv5 | Motion blur | 0.166 | 0.156 | +0.010 | [-0.009, +0.029] | Equivalent |
| YOLOv8 | Motion blur | 0.181 | 0.169 | +0.013 | [-0.007, +0.032] | Equivalent |
| YOLOv12 | Motion blur | 0.183 | 0.179 | +0.004 | [-0.017, +0.023] | Equivalent |
| YOLOv26 | Motion blur | 0.304 | 0.302 | +0.002 | [-0.023, +0.025] | Equivalent |
| YOLOv26_v2 | Motion blur | 0.263 | 0.223 | +0.040 | [+0.019, +0.061] | Inconclusive |
| YOLOv5 | Horizontal stretch | 0.666 | 0.657 | +0.009 | [-0.008, +0.029] | Equivalent |
| YOLOv8 | Horizontal stretch | 0.657 | 0.647 | +0.010 | [-0.010, +0.031] | Equivalent |
| YOLOv12 | Horizontal stretch | 0.630 | 0.620 | +0.010 | [-0.010, +0.033] | Equivalent |
| YOLOv26 | Horizontal stretch | 0.656 | 0.645 | +0.012 | [-0.009, +0.035] | Equivalent |
| YOLOv26_v2 | Horizontal stretch | 0.672 | 0.584 | +0.087 | [+0.066, +0.110] | Different |

### AP: test vs training samples

| Model | Rule | Test AUC | Alternative AUC | Difference | 90% CI | Result |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | Hue rotation | 0.710 | 0.713 | -0.003 | [-0.014, +0.012] | Equivalent |
| YOLOv8 | Hue rotation | 0.712 | 0.718 | -0.006 | [-0.018, +0.007] | Equivalent |
| YOLOv12 | Hue rotation | 0.707 | 0.710 | -0.003 | [-0.016, +0.011] | Equivalent |
| YOLOv26 | Hue rotation | 0.694 | 0.703 | -0.008 | [-0.022, +0.004] | Equivalent |
| YOLOv26_v2 | Hue rotation | 0.707 | 0.624 | +0.083 | [+0.067, +0.100] | Different |
| YOLOv5 | Resolution | 0.096 | 0.098 | -0.001 | [-0.012, +0.010] | Equivalent |
| YOLOv8 | Resolution | 0.100 | 0.099 | +0.001 | [-0.009, +0.014] | Equivalent |
| YOLOv12 | Resolution | 0.101 | 0.097 | +0.004 | [-0.007, +0.016] | Equivalent |
| YOLOv26 | Resolution | 0.158 | 0.158 | +0.000 | [-0.014, +0.013] | Equivalent |
| YOLOv26_v2 | Resolution | 0.142 | 0.105 | +0.037 | [+0.026, +0.049] | Equivalent |
| YOLOv5 | Gamma brightness | 0.228 | 0.227 | +0.001 | [-0.022, +0.026] | Equivalent |
| YOLOv8 | Gamma brightness | 0.227 | 0.233 | -0.006 | [-0.031, +0.018] | Equivalent |
| YOLOv12 | Gamma brightness | 0.223 | 0.228 | -0.004 | [-0.029, +0.019] | Equivalent |
| YOLOv26 | Gamma brightness | 0.253 | 0.253 | +0.000 | [-0.025, +0.021] | Equivalent |
| YOLOv26_v2 | Gamma brightness | 0.264 | 0.197 | +0.067 | [+0.045, +0.088] | Inconclusive |
| YOLOv5 | Motion blur | 0.183 | 0.160 | +0.023 | [+0.005, +0.044] | Equivalent |
| YOLOv8 | Motion blur | 0.198 | 0.177 | +0.021 | [+0.001, +0.042] | Equivalent |
| YOLOv12 | Motion blur | 0.198 | 0.178 | +0.020 | [+0.000, +0.039] | Equivalent |
| YOLOv26 | Motion blur | 0.320 | 0.308 | +0.012 | [-0.012, +0.035] | Equivalent |
| YOLOv26_v2 | Motion blur | 0.280 | 0.209 | +0.071 | [+0.049, +0.093] | Inconclusive |
| YOLOv5 | Horizontal stretch | 0.693 | 0.693 | +0.001 | [-0.015, +0.017] | Equivalent |
| YOLOv8 | Horizontal stretch | 0.680 | 0.682 | -0.001 | [-0.019, +0.018] | Equivalent |
| YOLOv12 | Horizontal stretch | 0.652 | 0.649 | +0.003 | [-0.017, +0.022] | Equivalent |
| YOLOv26 | Horizontal stretch | 0.682 | 0.683 | -0.000 | [-0.021, +0.018] | Equivalent |
| YOLOv26_v2 | Horizontal stretch | 0.693 | 0.604 | +0.089 | [+0.068, +0.110] | Different |

### AR: test vs training samples

| Model | Rule | Test AUC | Alternative AUC | Difference | 90% CI | Result |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | Hue rotation | 0.659 | 0.660 | -0.000 | [-0.014, +0.014] | Equivalent |
| YOLOv8 | Hue rotation | 0.667 | 0.670 | -0.002 | [-0.015, +0.011] | Equivalent |
| YOLOv12 | Hue rotation | 0.659 | 0.660 | -0.001 | [-0.015, +0.013] | Equivalent |
| YOLOv26 | Hue rotation | 0.644 | 0.650 | -0.006 | [-0.019, +0.008] | Equivalent |
| YOLOv26_v2 | Hue rotation | 0.660 | 0.598 | +0.062 | [+0.046, +0.078] | Inconclusive |
| YOLOv5 | Resolution | 0.088 | 0.091 | -0.002 | [-0.013, +0.007] | Equivalent |
| YOLOv8 | Resolution | 0.095 | 0.092 | +0.003 | [-0.008, +0.013] | Equivalent |
| YOLOv12 | Resolution | 0.095 | 0.092 | +0.003 | [-0.007, +0.014] | Equivalent |
| YOLOv26 | Resolution | 0.149 | 0.149 | +0.000 | [-0.013, +0.012] | Equivalent |
| YOLOv26_v2 | Resolution | 0.132 | 0.109 | +0.022 | [+0.011, +0.034] | Equivalent |
| YOLOv5 | Gamma brightness | 0.205 | 0.202 | +0.003 | [-0.020, +0.025] | Equivalent |
| YOLOv8 | Gamma brightness | 0.207 | 0.209 | -0.002 | [-0.025, +0.021] | Equivalent |
| YOLOv12 | Gamma brightness | 0.203 | 0.204 | -0.001 | [-0.025, +0.021] | Equivalent |
| YOLOv26 | Gamma brightness | 0.229 | 0.227 | +0.002 | [-0.021, +0.022] | Equivalent |
| YOLOv26_v2 | Gamma brightness | 0.236 | 0.189 | +0.047 | [+0.027, +0.067] | Inconclusive |
| YOLOv5 | Motion blur | 0.166 | 0.146 | +0.021 | [+0.001, +0.040] | Equivalent |
| YOLOv8 | Motion blur | 0.181 | 0.162 | +0.019 | [-0.000, +0.039] | Equivalent |
| YOLOv12 | Motion blur | 0.183 | 0.165 | +0.018 | [-0.001, +0.037] | Equivalent |
| YOLOv26 | Motion blur | 0.304 | 0.292 | +0.012 | [-0.012, +0.035] | Equivalent |
| YOLOv26_v2 | Motion blur | 0.263 | 0.214 | +0.048 | [+0.025, +0.071] | Inconclusive |
| YOLOv5 | Horizontal stretch | 0.666 | 0.667 | -0.001 | [-0.019, +0.017] | Equivalent |
| YOLOv8 | Horizontal stretch | 0.657 | 0.657 | -0.001 | [-0.020, +0.019] | Equivalent |
| YOLOv12 | Horizontal stretch | 0.630 | 0.628 | +0.002 | [-0.019, +0.022] | Equivalent |
| YOLOv26 | Horizontal stretch | 0.656 | 0.656 | +0.001 | [-0.021, +0.021] | Equivalent |
| YOLOv26_v2 | Horizontal stretch | 0.672 | 0.595 | +0.077 | [+0.055, +0.098] | Different |

## Transferability summary

| Model | Val AP | Val AR | Train AP | Train AR | All equivalent AUCs |
| --- | --- | --- | --- | --- | --- |
| YOLOv5 | 5/5 | 5/5 | 5/5 | 5/5 | 20/20 |
| YOLOv8 | 5/5 | 5/5 | 5/5 | 5/5 | 20/20 |
| YOLOv12 | 5/5 | 5/5 | 5/5 | 5/5 | 20/20 |
| YOLOv26 | 5/5 | 5/5 | 5/5 | 5/5 | 20/20 |
| YOLOv26_v2 | 1/5 | 1/5 | 1/5 | 1/5 | 4/20 |

Counts show equivalent rule-level AUC comparisons out of the five rules with numeric parameter curves.

## Largest parameter-level differences

| Target | Metric | Model | Rule | Setting | Difference | 90% CI | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Validation | AP | YOLOv26_v2 | Horizontal stretch | imgStretch_3 | +0.122 | [+0.095, +0.150] | Different |
| Validation | AP | YOLOv26_v2 | Hue rotation | imgColorWheel_7 | +0.121 | [+0.093, +0.150] | Different |
| Validation | AP | YOLOv26_v2 | Hue rotation | imgColorWheel_6 | +0.117 | [+0.097, +0.137] | Different |
| Training samples | AP | YOLOv26_v2 | Gamma brightness | imgDarkBright_9 | +0.114 | [+0.077, +0.152] | Different |
| Training samples | AP | YOLOv26_v2 | Hue rotation | imgColorWheel_6 | +0.114 | [+0.095, +0.134] | Different |
| Training samples | AP | YOLOv26_v2 | Gamma brightness | imgDarkBright_8 | +0.112 | [+0.075, +0.147] | Different |
| Validation | AR | YOLOv26_v2 | Horizontal stretch | imgStretch_3 | +0.111 | [+0.084, +0.139] | Different |
| Training samples | AP | YOLOv26_v2 | Hue rotation | imgColorWheel_5 | +0.111 | [+0.093, +0.129] | Different |
| Validation | AP | YOLOv26_v2 | Hue rotation | imgColorWheel_5 | +0.110 | [+0.090, +0.131] | Different |
| Training samples | AP | YOLOv26_v2 | Grayscale | imgGrayscale_0 | +0.108 | [+0.089, +0.130] | Different |
| Training samples | AP | YOLOv26_v2 | Gamma brightness | imgDarkBright_7 | +0.108 | [+0.071, +0.143] | Different |
| Validation | AP | YOLOv26_v2 | Horizontal stretch | imgStretch_4 | +0.108 | [+0.082, +0.137] | Different |
| Training samples | AP | YOLOv26_v2 | Hue rotation | imgColorWheel_7 | +0.107 | [+0.080, +0.136] | Different |
| Validation | AP | YOLOv26_v2 | Horizontal stretch | imgStretch_5 | +0.107 | [+0.082, +0.134] | Different |
| Validation | AP | YOLOv26_v2 | Horizontal stretch | imgStretch_2 | +0.105 | [+0.078, +0.134] | Different |
| Validation | AP | YOLOv26_v2 | Hue rotation | imgColorWheel_3 | +0.104 | [+0.081, +0.128] | Different |
| Validation | AP | YOLOv26_v2 | Hue rotation | imgColorWheel_2 | +0.103 | [+0.074, +0.132] | Different |
| Validation | AP | YOLOv26_v2 | Horizontal stretch | imgStretch_9 | +0.103 | [+0.085, +0.125] | Different |
| Validation | AR | YOLOv26_v2 | Hue rotation | imgColorWheel_7 | +0.103 | [+0.072, +0.137] | Different |
| Training samples | AP | YOLOv26_v2 | Hue rotation | imgColorWheel_2 | +0.102 | [+0.075, +0.130] | Different |
| Validation | AR | YOLOv26_v2 | Horizontal stretch | imgStretch_4 | +0.102 | [+0.075, +0.130] | Different |
| Validation | AP | YOLOv26_v2 | Horizontal stretch | imgStretch_8 | +0.102 | [+0.081, +0.123] | Different |
| Validation | AP | YOLOv26_v2 | Grayscale | imgGrayscale_0 | +0.101 | [+0.080, +0.123] | Different |
| Training samples | AP | YOLOv26_v2 | Horizontal stretch | imgStretch_4 | +0.100 | [+0.073, +0.128] | Different |
| Training samples | AP | YOLOv26_v2 | Horizontal stretch | imgStretch_3 | +0.100 | [+0.074, +0.128] | Different |

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
- The datasets contain different images, so reference and target bootstraps are independent.
- Images are the resampling units. No sequence/rosbag metadata was available to model temporal clustering.
- The shared gamma=1.0 control was requested for every rule; rule-specific identity settings were not substituted.
- Parameter-level intervals are estimation-only and are not adjusted for multiple comparisons.
- AUC can be equivalent even when individual settings are inconclusive or different; consult the parameter records in `results.json`.
