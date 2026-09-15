# Test-set sample-size metamorphic consistency equivalence

## Research question

If only 50% of the test dataset were used for metamorphic assessment, would (a) the observed original-vs-transformed consistency and (b) the observed ground-truth accuracy degradation caused by each transformation be equivalent to using the full test dataset?

## Design

The analysis covers 5 models, 6 rules, and 51 settings, drawn entirely from the `test` dataset (363 images).

The `sample` arm uses a single fixed, reproducibly-drawn subsample containing 182 images (50% of 363, rounded to the nearest image). The same subsample is reused across every model, rule, setting, and analysis below.

Two independent statistics are tested for equivalence between the two arms:

- **Original-vs-transformed consistency** (original_vs_transformed_consistency): raw COCO AP/AR with each model's original detections as the reference and the transformed detections as the prediction (no baseline subtraction).
- **Ground-truth accuracy degradation from transformation** (gt_accuracy_degradation): original_vs_gt AP/AR (untransformed detections vs. real ground truth; a per-model baseline constant across rules) minus transformed_vs_gt AP/AR (transformed detections vs. real ground truth).

Each arm is independently resampled 2,000 times with replacement from within its own population: the `full` arm resamples from all 363 images, while the `sample` arm resamples only from its 182-image subsample. Results use percentile 90% confidence intervals and a symmetric +/-0.05 equivalence margin. No p-values or multiple-testing corrections are used.

Point AP/AR values reported for the `full` arm are read directly from the existing `analysis/results/<dataset>/<model>/<variant_id>/<comparison>/stats.json` files (and the per-model `original_vs_gt/stats.json`), not recomputed; only the `sample` arm's point and both arms' bootstrap confidence intervals are computed by this script (see Validation).

Classification rules:

- **Equivalent:** the complete 90% interval lies within [-0.05, +0.05].
- **Meaningfully different:** the complete interval lies beyond one equivalence boundary.
- **Inconclusive:** the interval overlaps an equivalence boundary.

A positive difference means the full-dataset statistic is greater than the 50% subsample; a negative difference means the subsample statistic is greater.

## Validation

The implementation recomputed 1,030 point AP/AR values for the `full` arm over every declared image and compared them with the stored per-variant `stats.json` files. 851 values agreed within 1e-09; 179 differed.

The maximum stored-versus-recomputed discrepancy was 0.0125 at `transformed_vs_gt/yolov26_v2/imgDarkBright_9/AR`. Reported `full` arm points use the stored stats.json values, not this recomputation.

## Original-vs-transformed consistency

Statistic: raw COCO AP/AR with each model's original detections as the reference and the transformed detections as the prediction (no baseline subtraction).

### Key findings

- At parameter level, 421 of 510 comparisons (82.5%) were equivalent, 89 were inconclusive, and 0 were meaningfully different.
- At rule-curve level, 50 of 50 AUC comparisons were equivalent; 0 were inconclusive and 0 were meaningfully different.
- Every AUC comparison was equivalent for YOLOv5, YOLOv8, YOLOv12, YOLOv26, YOLOv26_v2. Their overall parameter-response patterns therefore transfer from the full test set to the 50% subsample within the +/-0.05 margin.

### Parameter-level overview

| Comparison | Metric | Settings | Equivalent | Inconclusive | Different | Equivalent % |
| --- | --- | --- | --- | --- | --- | --- |
| Test subsample | AP | 255 | 212 | 43 | 0 | 83.1% |
| Test subsample | AR | 255 | 209 | 46 | 0 | 82.0% |

### Rule-level AUC results

AUC uses the complete configured parameter range with its axis scaled to 0-1. Rules with a single setting (Grayscale) have no AUC result.

#### AP: full test set vs 50% subsample

| Model | Rule | Full AUC | Sample AUC | Difference | 90% CI | Result |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | Hue rotation | 0.285 | 0.283 | +0.001 | [-0.016, +0.016] | Equivalent |
| YOLOv8 | Hue rotation | 0.281 | 0.280 | +0.000 | [-0.014, +0.014] | Equivalent |
| YOLOv12 | Hue rotation | 0.287 | 0.288 | -0.002 | [-0.018, +0.013] | Equivalent |
| YOLOv26 | Hue rotation | 0.296 | 0.292 | +0.004 | [-0.012, +0.019] | Equivalent |
| YOLOv26_v2 | Hue rotation | 0.286 | 0.287 | -0.001 | [-0.018, +0.013] | Equivalent |
| YOLOv5 | Resolution | 0.899 | 0.905 | -0.007 | [-0.018, +0.005] | Equivalent |
| YOLOv8 | Resolution | 0.892 | 0.897 | -0.005 | [-0.017, +0.007] | Equivalent |
| YOLOv12 | Resolution | 0.893 | 0.901 | -0.008 | [-0.020, +0.004] | Equivalent |
| YOLOv26 | Resolution | 0.832 | 0.830 | +0.002 | [-0.015, +0.017] | Equivalent |
| YOLOv26_v2 | Resolution | 0.850 | 0.854 | -0.004 | [-0.019, +0.012] | Equivalent |
| YOLOv5 | Gamma brightness | 0.767 | 0.778 | -0.012 | [-0.039, +0.017] | Equivalent |
| YOLOv8 | Gamma brightness | 0.766 | 0.780 | -0.014 | [-0.038, +0.010] | Equivalent |
| YOLOv12 | Gamma brightness | 0.771 | 0.788 | -0.017 | [-0.042, +0.008] | Equivalent |
| YOLOv26 | Gamma brightness | 0.737 | 0.745 | -0.008 | [-0.033, +0.017] | Equivalent |
| YOLOv26_v2 | Gamma brightness | 0.729 | 0.744 | -0.015 | [-0.040, +0.011] | Equivalent |
| YOLOv5 | Motion blur | 0.812 | 0.818 | -0.007 | [-0.031, +0.017] | Equivalent |
| YOLOv8 | Motion blur | 0.795 | 0.803 | -0.008 | [-0.031, +0.017] | Equivalent |
| YOLOv12 | Motion blur | 0.796 | 0.808 | -0.012 | [-0.034, +0.011] | Equivalent |
| YOLOv26 | Motion blur | 0.670 | 0.667 | +0.002 | [-0.027, +0.030] | Equivalent |
| YOLOv26_v2 | Motion blur | 0.713 | 0.718 | -0.006 | [-0.034, +0.023] | Equivalent |
| YOLOv5 | Horizontal stretch | 0.302 | 0.301 | +0.001 | [-0.021, +0.021] | Equivalent |
| YOLOv8 | Horizontal stretch | 0.312 | 0.310 | +0.003 | [-0.021, +0.025] | Equivalent |
| YOLOv12 | Horizontal stretch | 0.342 | 0.339 | +0.003 | [-0.024, +0.029] | Equivalent |
| YOLOv26 | Horizontal stretch | 0.308 | 0.301 | +0.007 | [-0.019, +0.030] | Equivalent |
| YOLOv26_v2 | Horizontal stretch | 0.299 | 0.295 | +0.004 | [-0.017, +0.024] | Equivalent |

#### AR: full test set vs 50% subsample

| Model | Rule | Full AUC | Sample AUC | Difference | 90% CI | Result |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | Hue rotation | 0.340 | 0.337 | +0.002 | [-0.015, +0.019] | Equivalent |
| YOLOv8 | Hue rotation | 0.331 | 0.329 | +0.003 | [-0.013, +0.017] | Equivalent |
| YOLOv12 | Hue rotation | 0.340 | 0.338 | +0.002 | [-0.014, +0.018] | Equivalent |
| YOLOv26 | Hue rotation | 0.351 | 0.347 | +0.005 | [-0.012, +0.021] | Equivalent |
| YOLOv26_v2 | Hue rotation | 0.337 | 0.335 | +0.002 | [-0.015, +0.017] | Equivalent |
| YOLOv5 | Resolution | 0.911 | 0.917 | -0.006 | [-0.017, +0.005] | Equivalent |
| YOLOv8 | Resolution | 0.904 | 0.909 | -0.005 | [-0.016, +0.006] | Equivalent |
| YOLOv12 | Resolution | 0.904 | 0.912 | -0.008 | [-0.019, +0.004] | Equivalent |
| YOLOv26 | Resolution | 0.847 | 0.847 | +0.000 | [-0.015, +0.016] | Equivalent |
| YOLOv26_v2 | Resolution | 0.866 | 0.870 | -0.004 | [-0.019, +0.011] | Equivalent |
| YOLOv5 | Gamma brightness | 0.794 | 0.805 | -0.011 | [-0.037, +0.016] | Equivalent |
| YOLOv8 | Gamma brightness | 0.792 | 0.804 | -0.013 | [-0.036, +0.010] | Equivalent |
| YOLOv12 | Gamma brightness | 0.796 | 0.813 | -0.017 | [-0.041, +0.006] | Equivalent |
| YOLOv26 | Gamma brightness | 0.767 | 0.774 | -0.008 | [-0.031, +0.016] | Equivalent |
| YOLOv26_v2 | Gamma brightness | 0.761 | 0.775 | -0.014 | [-0.037, +0.010] | Equivalent |
| YOLOv5 | Motion blur | 0.832 | 0.839 | -0.007 | [-0.029, +0.015] | Equivalent |
| YOLOv8 | Motion blur | 0.817 | 0.825 | -0.007 | [-0.031, +0.015] | Equivalent |
| YOLOv12 | Motion blur | 0.816 | 0.828 | -0.012 | [-0.034, +0.010] | Equivalent |
| YOLOv26 | Motion blur | 0.692 | 0.691 | +0.001 | [-0.027, +0.030] | Equivalent |
| YOLOv26_v2 | Motion blur | 0.735 | 0.741 | -0.006 | [-0.034, +0.022] | Equivalent |
| YOLOv5 | Horizontal stretch | 0.333 | 0.331 | +0.001 | [-0.022, +0.024] | Equivalent |
| YOLOv8 | Horizontal stretch | 0.342 | 0.339 | +0.004 | [-0.023, +0.029] | Equivalent |
| YOLOv12 | Horizontal stretch | 0.369 | 0.365 | +0.005 | [-0.023, +0.031] | Equivalent |
| YOLOv26 | Horizontal stretch | 0.339 | 0.331 | +0.008 | [-0.018, +0.033] | Equivalent |
| YOLOv26_v2 | Horizontal stretch | 0.326 | 0.320 | +0.005 | [-0.017, +0.027] | Equivalent |

### Sample-size equivalence summary

| Model | Sample AP | Sample AR | All equivalent AUCs |
| --- | --- | --- | --- |
| YOLOv5 | 5/5 | 5/5 | 10/10 |
| YOLOv8 | 5/5 | 5/5 | 10/10 |
| YOLOv12 | 5/5 | 5/5 | 10/10 |
| YOLOv26 | 5/5 | 5/5 | 10/10 |
| YOLOv26_v2 | 5/5 | 5/5 | 10/10 |

Counts show equivalent rule-level AUC comparisons out of the rules with numeric parameter curves.

### Largest parameter-level differences

| Target | Metric | Model | Rule | Setting | Difference | 90% CI | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Test subsample | AR | YOLOv5 | Resolution | imgResolution_0 | -0.051 | [-0.105, -0.001] | Inconclusive |
| Test subsample | AP | YOLOv5 | Resolution | imgResolution_0 | -0.047 | [-0.101, +0.002] | Inconclusive |
| Test subsample | AR | YOLOv12 | Resolution | imgResolution_0 | -0.045 | [-0.097, +0.003] | Inconclusive |
| Test subsample | AP | YOLOv12 | Resolution | imgResolution_0 | -0.043 | [-0.093, +0.004] | Inconclusive |
| Test subsample | AR | YOLOv26_v2 | Resolution | imgResolution_0 | -0.042 | [-0.088, +0.003] | Inconclusive |
| Test subsample | AR | YOLOv8 | Resolution | imgResolution_0 | -0.041 | [-0.090, +0.005] | Inconclusive |
| Test subsample | AR | YOLOv12 | Gamma brightness | imgDarkBright_9 | -0.039 | [-0.086, +0.008] | Inconclusive |
| Test subsample | AP | YOLOv26_v2 | Resolution | imgResolution_0 | -0.038 | [-0.084, +0.004] | Inconclusive |
| Test subsample | AR | YOLOv26 | Resolution | imgResolution_0 | -0.037 | [-0.080, +0.004] | Inconclusive |
| Test subsample | AP | YOLOv12 | Gamma brightness | imgDarkBright_9 | -0.037 | [-0.085, +0.013] | Inconclusive |
| Test subsample | AP | YOLOv8 | Resolution | imgResolution_0 | -0.036 | [-0.084, +0.009] | Inconclusive |
| Test subsample | AR | YOLOv12 | Gamma brightness | imgDarkBright_8 | -0.035 | [-0.079, +0.010] | Inconclusive |
| Test subsample | AR | YOLOv12 | Gamma brightness | imgDarkBright_7 | -0.035 | [-0.075, +0.005] | Inconclusive |
| Test subsample | AP | YOLOv26 | Resolution | imgResolution_0 | -0.034 | [-0.075, +0.004] | Inconclusive |
| Test subsample | AP | YOLOv12 | Gamma brightness | imgDarkBright_7 | -0.033 | [-0.075, +0.008] | Inconclusive |
| Test subsample | AP | YOLOv12 | Gamma brightness | imgDarkBright_8 | -0.032 | [-0.079, +0.014] | Inconclusive |
| Test subsample | AR | YOLOv12 | Motion blur | imgMotionBlur_9 | -0.027 | [-0.066, +0.013] | Inconclusive |
| Test subsample | AR | YOLOv5 | Resolution | imgResolution_1 | -0.027 | [-0.059, +0.004] | Inconclusive |
| Test subsample | AR | YOLOv12 | Gamma brightness | imgDarkBright_6 | -0.027 | [-0.063, +0.009] | Inconclusive |
| Test subsample | AR | YOLOv8 | Gamma brightness | imgDarkBright_9 | -0.026 | [-0.069, +0.016] | Inconclusive |
| Test subsample | AP | YOLOv26_v2 | Gamma brightness | imgDarkBright_6 | -0.025 | [-0.063, +0.014] | Inconclusive |
| Test subsample | AP | YOLOv8 | Gamma brightness | imgDarkBright_8 | -0.025 | [-0.068, +0.015] | Inconclusive |
| Test subsample | AP | YOLOv5 | Gamma brightness | imgDarkBright_7 | -0.025 | [-0.067, +0.017] | Inconclusive |
| Test subsample | AR | YOLOv26_v2 | Gamma brightness | imgDarkBright_9 | -0.025 | [-0.067, +0.017] | Inconclusive |
| Test subsample | AP | YOLOv12 | Resolution | imgResolution_1 | -0.025 | [-0.057, +0.008] | Inconclusive |

### Response curves

#### Grayscale

![Grayscale AP original_vs_transformed](plots/imgGrayscale_ap_original_vs_transformed.png)

![Grayscale AR original_vs_transformed](plots/imgGrayscale_ar_original_vs_transformed.png)

#### Hue rotation

![Hue rotation AP original_vs_transformed](plots/imgColorWheel_ap_original_vs_transformed.png)

![Hue rotation AR original_vs_transformed](plots/imgColorWheel_ar_original_vs_transformed.png)

#### Resolution

![Resolution AP original_vs_transformed](plots/imgResolution_ap_original_vs_transformed.png)

![Resolution AR original_vs_transformed](plots/imgResolution_ar_original_vs_transformed.png)

#### Gamma brightness

![Gamma brightness AP original_vs_transformed](plots/imgDarkBright_ap_original_vs_transformed.png)

![Gamma brightness AR original_vs_transformed](plots/imgDarkBright_ar_original_vs_transformed.png)

#### Motion blur

![Motion blur AP original_vs_transformed](plots/imgMotionBlur_ap_original_vs_transformed.png)

![Motion blur AR original_vs_transformed](plots/imgMotionBlur_ar_original_vs_transformed.png)

#### Horizontal stretch

![Horizontal stretch AP original_vs_transformed](plots/imgStretch_ap_original_vs_transformed.png)

![Horizontal stretch AR original_vs_transformed](plots/imgStretch_ar_original_vs_transformed.png)


## Ground-truth accuracy degradation from transformation

Statistic: original_vs_gt AP/AR (untransformed detections vs. real ground truth; a per-model baseline constant across rules) minus transformed_vs_gt AP/AR (transformed detections vs. real ground truth).

### Key findings

- At parameter level, 418 of 510 comparisons (82.0%) were equivalent, 92 were inconclusive, and 0 were meaningfully different.
- At rule-curve level, 50 of 50 AUC comparisons were equivalent; 0 were inconclusive and 0 were meaningfully different.
- Every AUC comparison was equivalent for YOLOv5, YOLOv8, YOLOv12, YOLOv26, YOLOv26_v2. Their overall parameter-response patterns therefore transfer from the full test set to the 50% subsample within the +/-0.05 margin.

### Parameter-level overview

| Comparison | Metric | Settings | Equivalent | Inconclusive | Different | Equivalent % |
| --- | --- | --- | --- | --- | --- | --- |
| Test subsample | AP | 255 | 203 | 52 | 0 | 79.6% |
| Test subsample | AR | 255 | 215 | 40 | 0 | 84.3% |

### Rule-level AUC results

AUC uses the complete configured parameter range with its axis scaled to 0-1. Rules with a single setting (Grayscale) have no AUC result.

#### AP: full test set vs 50% subsample

| Model | Rule | Full AUC | Sample AUC | Difference | 90% CI | Result |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | Hue rotation | 0.305 | 0.293 | +0.012 | [-0.020, +0.043] | Equivalent |
| YOLOv8 | Hue rotation | 0.313 | 0.302 | +0.011 | [-0.020, +0.042] | Equivalent |
| YOLOv12 | Hue rotation | 0.309 | 0.294 | +0.015 | [-0.017, +0.045] | Equivalent |
| YOLOv26 | Hue rotation | 0.374 | 0.359 | +0.015 | [-0.013, +0.043] | Equivalent |
| YOLOv26_v2 | Hue rotation | 0.386 | 0.377 | +0.009 | [-0.020, +0.041] | Equivalent |
| YOLOv5 | Resolution | 0.018 | 0.015 | +0.003 | [-0.003, +0.009] | Equivalent |
| YOLOv8 | Resolution | 0.017 | 0.015 | +0.002 | [-0.004, +0.008] | Equivalent |
| YOLOv12 | Resolution | 0.015 | 0.012 | +0.003 | [-0.003, +0.010] | Equivalent |
| YOLOv26 | Resolution | 0.023 | 0.018 | +0.005 | [-0.002, +0.013] | Equivalent |
| YOLOv26_v2 | Resolution | 0.024 | 0.022 | +0.002 | [-0.004, +0.010] | Equivalent |
| YOLOv5 | Gamma brightness | 0.062 | 0.054 | +0.008 | [-0.002, +0.019] | Equivalent |
| YOLOv8 | Gamma brightness | 0.060 | 0.054 | +0.006 | [-0.002, +0.019] | Equivalent |
| YOLOv12 | Gamma brightness | 0.060 | 0.053 | +0.007 | [-0.002, +0.021] | Equivalent |
| YOLOv26 | Gamma brightness | 0.080 | 0.071 | +0.009 | [-0.002, +0.024] | Equivalent |
| YOLOv26_v2 | Gamma brightness | 0.087 | 0.083 | +0.005 | [-0.007, +0.023] | Equivalent |
| YOLOv5 | Motion blur | 0.040 | 0.037 | +0.003 | [-0.008, +0.013] | Equivalent |
| YOLOv8 | Motion blur | 0.040 | 0.039 | +0.001 | [-0.008, +0.011] | Equivalent |
| YOLOv12 | Motion blur | 0.038 | 0.035 | +0.003 | [-0.006, +0.014] | Equivalent |
| YOLOv26 | Motion blur | 0.092 | 0.083 | +0.009 | [-0.007, +0.024] | Equivalent |
| YOLOv26_v2 | Motion blur | 0.073 | 0.070 | +0.003 | [-0.011, +0.018] | Equivalent |
| YOLOv5 | Horizontal stretch | 0.285 | 0.277 | +0.008 | [-0.020, +0.035] | Equivalent |
| YOLOv8 | Horizontal stretch | 0.281 | 0.275 | +0.006 | [-0.020, +0.033] | Equivalent |
| YOLOv12 | Horizontal stretch | 0.261 | 0.253 | +0.008 | [-0.017, +0.033] | Equivalent |
| YOLOv26 | Horizontal stretch | 0.332 | 0.321 | +0.012 | [-0.015, +0.038] | Equivalent |
| YOLOv26_v2 | Horizontal stretch | 0.352 | 0.349 | +0.003 | [-0.023, +0.030] | Equivalent |

#### AR: full test set vs 50% subsample

| Model | Rule | Full AUC | Sample AUC | Difference | 90% CI | Result |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | Hue rotation | 0.289 | 0.280 | +0.009 | [-0.020, +0.039] | Equivalent |
| YOLOv8 | Hue rotation | 0.301 | 0.293 | +0.008 | [-0.021, +0.037] | Equivalent |
| YOLOv12 | Hue rotation | 0.294 | 0.285 | +0.009 | [-0.020, +0.038] | Equivalent |
| YOLOv26 | Hue rotation | 0.353 | 0.342 | +0.011 | [-0.015, +0.038] | Equivalent |
| YOLOv26_v2 | Hue rotation | 0.369 | 0.364 | +0.005 | [-0.023, +0.034] | Equivalent |
| YOLOv5 | Resolution | 0.018 | 0.015 | +0.003 | [-0.003, +0.010] | Equivalent |
| YOLOv8 | Resolution | 0.018 | 0.015 | +0.002 | [-0.003, +0.008] | Equivalent |
| YOLOv12 | Resolution | 0.016 | 0.013 | +0.003 | [-0.003, +0.010] | Equivalent |
| YOLOv26 | Resolution | 0.021 | 0.016 | +0.005 | [-0.002, +0.014] | Equivalent |
| YOLOv26_v2 | Resolution | 0.023 | 0.020 | +0.003 | [-0.004, +0.011] | Equivalent |
| YOLOv5 | Gamma brightness | 0.059 | 0.051 | +0.007 | [-0.002, +0.018] | Equivalent |
| YOLOv8 | Gamma brightness | 0.058 | 0.053 | +0.006 | [-0.001, +0.018] | Equivalent |
| YOLOv12 | Gamma brightness | 0.058 | 0.051 | +0.006 | [-0.002, +0.020] | Equivalent |
| YOLOv26 | Gamma brightness | 0.073 | 0.065 | +0.008 | [-0.002, +0.023] | Equivalent |
| YOLOv26_v2 | Gamma brightness | 0.081 | 0.077 | +0.004 | [-0.007, +0.020] | Equivalent |
| YOLOv5 | Motion blur | 0.039 | 0.036 | +0.003 | [-0.007, +0.013] | Equivalent |
| YOLOv8 | Motion blur | 0.039 | 0.037 | +0.002 | [-0.007, +0.011] | Equivalent |
| YOLOv12 | Motion blur | 0.038 | 0.035 | +0.003 | [-0.007, +0.014] | Equivalent |
| YOLOv26 | Motion blur | 0.090 | 0.080 | +0.010 | [-0.006, +0.026] | Equivalent |
| YOLOv26_v2 | Motion blur | 0.072 | 0.068 | +0.004 | [-0.010, +0.019] | Equivalent |
| YOLOv5 | Horizontal stretch | 0.280 | 0.274 | +0.006 | [-0.020, +0.032] | Equivalent |
| YOLOv8 | Horizontal stretch | 0.277 | 0.271 | +0.006 | [-0.020, +0.031] | Equivalent |
| YOLOv12 | Horizontal stretch | 0.260 | 0.255 | +0.005 | [-0.020, +0.030] | Equivalent |
| YOLOv26 | Horizontal stretch | 0.327 | 0.318 | +0.009 | [-0.017, +0.035] | Equivalent |
| YOLOv26_v2 | Horizontal stretch | 0.351 | 0.349 | +0.001 | [-0.025, +0.027] | Equivalent |

### Sample-size equivalence summary

| Model | Sample AP | Sample AR | All equivalent AUCs |
| --- | --- | --- | --- |
| YOLOv5 | 5/5 | 5/5 | 10/10 |
| YOLOv8 | 5/5 | 5/5 | 10/10 |
| YOLOv12 | 5/5 | 5/5 | 10/10 |
| YOLOv26 | 5/5 | 5/5 | 10/10 |
| YOLOv26_v2 | 5/5 | 5/5 | 10/10 |

Counts show equivalent rule-level AUC comparisons out of the rules with numeric parameter curves.

### Largest parameter-level differences

| Target | Metric | Model | Rule | Setting | Difference | 90% CI | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Test subsample | AR | YOLOv26 | Resolution | imgResolution_0 | +0.041 | [+0.005, +0.081] | Inconclusive |
| Test subsample | AP | YOLOv26 | Resolution | imgResolution_0 | +0.039 | [+0.003, +0.078] | Inconclusive |
| Test subsample | AR | YOLOv5 | Resolution | imgResolution_0 | +0.031 | [-0.000, +0.073] | Inconclusive |
| Test subsample | AP | YOLOv5 | Resolution | imgResolution_0 | +0.028 | [-0.003, +0.070] | Inconclusive |
| Test subsample | AP | YOLOv12 | Resolution | imgResolution_0 | +0.027 | [-0.005, +0.072] | Inconclusive |
| Test subsample | AR | YOLOv12 | Resolution | imgResolution_0 | +0.026 | [-0.005, +0.072] | Inconclusive |
| Test subsample | AR | YOLOv26 | Motion blur | imgMotionBlur_9 | +0.025 | [-0.004, +0.055] | Inconclusive |
| Test subsample | AR | YOLOv26 | Hue rotation | imgColorWheel_4 | +0.025 | [-0.018, +0.067] | Inconclusive |
| Test subsample | AR | YOLOv8 | Resolution | imgResolution_0 | +0.023 | [-0.008, +0.065] | Inconclusive |
| Test subsample | AP | YOLOv26 | Hue rotation | imgColorWheel_4 | +0.022 | [-0.018, +0.065] | Inconclusive |
| Test subsample | AR | YOLOv26 | Motion blur | imgMotionBlur_8 | +0.022 | [-0.004, +0.050] | Inconclusive |
| Test subsample | AP | YOLOv26 | Motion blur | imgMotionBlur_9 | +0.022 | [-0.005, +0.051] | Inconclusive |
| Test subsample | AP | YOLOv26 | Hue rotation | imgColorWheel_2 | +0.022 | [-0.016, +0.059] | Inconclusive |
| Test subsample | AR | YOLOv26_v2 | Resolution | imgResolution_0 | +0.022 | [-0.009, +0.066] | Inconclusive |
| Test subsample | AP | YOLOv12 | Hue rotation | imgColorWheel_4 | +0.021 | [-0.020, +0.064] | Inconclusive |
| Test subsample | AP | YOLOv8 | Resolution | imgResolution_0 | +0.021 | [-0.011, +0.063] | Inconclusive |
| Test subsample | AP | YOLOv26 | Horizontal stretch | imgStretch_9 | +0.020 | [-0.019, +0.063] | Inconclusive |
| Test subsample | AP | YOLOv26 | Motion blur | imgMotionBlur_8 | +0.020 | [-0.004, +0.047] | Equivalent |
| Test subsample | AR | YOLOv26 | Hue rotation | imgColorWheel_2 | +0.020 | [-0.015, +0.054] | Inconclusive |
| Test subsample | AP | YOLOv26 | Horizontal stretch | imgStretch_7 | +0.019 | [-0.019, +0.058] | Inconclusive |
| Test subsample | AR | YOLOv26 | Motion blur | imgMotionBlur_7 | +0.019 | [-0.006, +0.046] | Equivalent |
| Test subsample | AP | YOLOv5 | Hue rotation | imgColorWheel_3 | +0.019 | [-0.026, +0.061] | Inconclusive |
| Test subsample | AP | YOLOv26_v2 | Resolution | imgResolution_0 | +0.019 | [-0.012, +0.063] | Inconclusive |
| Test subsample | AR | YOLOv12 | Hue rotation | imgColorWheel_5 | +0.019 | [-0.026, +0.062] | Inconclusive |
| Test subsample | AP | YOLOv26 | Hue rotation | imgColorWheel_3 | +0.019 | [-0.021, +0.058] | Inconclusive |

### Response curves

#### Grayscale

![Grayscale AP gt_accuracy_degradation](plots/imgGrayscale_ap_gt_accuracy_degradation.png)

![Grayscale AR gt_accuracy_degradation](plots/imgGrayscale_ar_gt_accuracy_degradation.png)

#### Hue rotation

![Hue rotation AP gt_accuracy_degradation](plots/imgColorWheel_ap_gt_accuracy_degradation.png)

![Hue rotation AR gt_accuracy_degradation](plots/imgColorWheel_ar_gt_accuracy_degradation.png)

#### Resolution

![Resolution AP gt_accuracy_degradation](plots/imgResolution_ap_gt_accuracy_degradation.png)

![Resolution AR gt_accuracy_degradation](plots/imgResolution_ar_gt_accuracy_degradation.png)

#### Gamma brightness

![Gamma brightness AP gt_accuracy_degradation](plots/imgDarkBright_ap_gt_accuracy_degradation.png)

![Gamma brightness AR gt_accuracy_degradation](plots/imgDarkBright_ar_gt_accuracy_degradation.png)

#### Motion blur

![Motion blur AP gt_accuracy_degradation](plots/imgMotionBlur_ap_gt_accuracy_degradation.png)

![Motion blur AR gt_accuracy_degradation](plots/imgMotionBlur_ar_gt_accuracy_degradation.png)

#### Horizontal stretch

![Horizontal stretch AP gt_accuracy_degradation](plots/imgStretch_ap_gt_accuracy_degradation.png)

![Horizontal stretch AR gt_accuracy_degradation](plots/imgStretch_ar_gt_accuracy_degradation.png)


## Interpretation cautions

- Original-vs-transformed consistency concerns metamorphic consistency, not absolute ground-truth detection accuracy.
- Ground-truth accuracy degradation concerns real detection accuracy loss and is directly comparable across rules on the same AP/AR scale.
- Unlike a comparison between independent datasets, the `sample` arm's images are a strict subset of the `full` arm's images, so the two bootstrap distributions are correlated rather than independent. The comparison estimates the added sampling variability/bias from using fewer images, not generalisation across distinct datasets.
- Images are the resampling units. No sequence/rosbag metadata was available to model temporal clustering.
- Parameter-level intervals are estimation-only and are not adjusted for multiple comparisons.
- AUC can be equivalent even when individual settings are inconclusive or different; consult the parameter records in `results.json`.
