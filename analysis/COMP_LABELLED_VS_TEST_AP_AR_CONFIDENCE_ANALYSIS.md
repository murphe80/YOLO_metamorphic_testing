# Comp-labelled vs test: AP, AR, and confidence-change analysis

## Executive summary

The test dataset usually predicts **the shape of model degradation** as a metamorphic parameter becomes more severe, but it often overestimates how much performance remains on `comp_labelled`. The strongest domain gaps are motion blur and gamma darkening. Resolution and horizontal stretch also transfer in shape but become materially harsher on competition images. Hue rotation is the most transferable rule, while grayscale is the only rule whose aggregate competition results are slightly better.

The most important model-level conclusions are:

- **YOLOv26_v2 has the strongest absolute competition baseline** at AP/AR 0.964/0.973, but it is often one of the least self-consistent models under transformations. High accuracy and high metamorphic consistency are therefore not interchangeable.
- **YOLOv5 is the repeatability specialist.** It leads consistency at every blur kernel on both datasets and has the strongest average resolution consistency. On competition blur it also becomes the absolute-accuracy leader from kernel 11 onward.
- **YOLOv26 is the strongest accuracy-oriented model across transformed data.** It leads average transformed-vs-GT accuracy for hue, brightness, and stretch on both datasets. It also leads grayscale accuracy and most test resolution settings.
- **YOLOv12's test-set stretch advantage does not transfer.** It leads test consistency from width ratio 1.4 to 2.8, whereas YOLOv26 leads most moderate competition settings.
- **Competition brightness and blur cause many more correct detections to disappear.** Across non-control settings, retained-TP rate falls from 87.4% to 70.1% for brightness and from 90.1% to 56.6% for blur. Confidence also drops more strongly among the TPs that remain.
- **Hue response transfers unusually well.** Both datasets collapse at 60-120 degrees and recover at 140-160 degrees. The cross-dataset consistency correlation is 0.986 across model-setting cells.
- **Extreme settings are failure regions, not useful model-selection opportunities.** On `comp_labelled`, mean AP/AR is only 0.053/0.070 at gamma 5, 0.125/0.129 at blur kernel 21, and 0.005/0.010 at stretch ratio 2.8.

## Scope, sources, and metric interpretation

The assessment uses the aggregate AP/AR JSONs:

- `analysis/results/test/parameter_variations/metrics_original_vs_transformed.json`
- `analysis/results/test/parameter_variations/metrics_transformed_vs_gt.json`
- `analysis/results/comp_labelled/parameter_variations/metrics_original_vs_transformed.json`
- `analysis/results/comp_labelled/parameter_variations_gt/metrics_transformed_vs_gt.json`

The confidence-change plots do not have a separate summary JSON. Their values were recomputed with the repository's `plot_confidence_change.py` matching logic from the per-model detection JSONs under `results/test` and `results/competition_labelled`, using the corresponding `annotations_coco.json` ground truth and IoU 0.5. Geometric detections use the coordinate-restored `detections_for_gt_transformed.json` files.

The two AP/AR comparisons answer different questions:

- **Consistency AP/AR (`original_vs_transformed`)**: how closely transformed predictions reproduce the model's original predictions. It can reward stable errors.
- **Accuracy AP/AR (`transformed_vs_gt`)**: how accurately transformed predictions match labelled ground truth.

The confidence analysis separates:

- **TP retained**: the percentage of original true positives still detected after transformation.
- **Retained-TP delta confidence**: transformed confidence minus original confidence for the same labelled object. Values near zero are stable; negative values indicate confidence loss.
- **Lost-TP confidence**: the original confidence of correct detections that disappear. A high value means the transformation removes detections that were previously confident.
- **Persistent-FP delta confidence**: confidence change for spatially matched false positives. Negative is generally favourable, but only if true positives are not disappearing at the same time.

AP/AR tables below are equal-weight means over the five models. Confidence means are weighted by the number of matched observations. Model leaders use the mean of AP and AR only as a compact tie-breaker; AP and AR remain reported separately.

Control settings are excluded from non-control averages: hue 0/180, gamma 1.0, resolution 1.0, and stretch 1.0. Hue 180 is an identity because the implementation wraps the OpenCV hue channel modulo 180. Competition resolution 0.5 is also excluded from summary averages because it produces exact 1.000/1.000 consistency for every model; see [Resolution](#resolution).

## Dataset baselines

The datasets are not matched samples: test has 363 images and 7,078 annotations, while `comp_labelled` has 139 images and 3,639 annotations. Absolute cross-dataset AP/AR differences therefore combine domain, scene composition, annotation mix, and transformation sensitivity.

| Model | Test original-vs-GT AP/AR | Comp-labelled original-vs-GT AP/AR | Comp - test AP/AR |
|---|---:|---:|---:|
| YOLOv5 | 0.470 / 0.488 | 0.574 / 0.589 | +0.104 / +0.101 |
| YOLOv8 | 0.477 / 0.494 | 0.571 / 0.587 | +0.094 / +0.093 |
| YOLOv12 | 0.475 / 0.492 | 0.565 / 0.580 | +0.090 / +0.088 |
| YOLOv26 | 0.583 / 0.609 | 0.724 / 0.754 | +0.141 / +0.145 |
| YOLOv26_v2 | **0.584 / 0.611** | **0.964 / 0.973** | +0.380 / +0.362 |

The unusually strong `comp_labelled` YOLOv26_v2 baseline is important context: it can lose more prediction consistency than YOLOv5 yet remain more accurate against ground truth.

## Cross-rule comparison

These are non-control averages over all models and applicable settings. Confidence columns are `retained-TP delta / lost-TP original confidence / persistent-FP delta`.

| Rule | Dataset | Consistency AP/AR | Accuracy AP/AR | TP retained | Confidence values |
|---|---|---:|---:|---:|---:|
| Grayscale | Test | 0.143 / 0.204 | 0.100 / 0.139 | 38.9% | -0.214 / 0.787 / -0.050 |
|  | Comp | 0.144 / 0.225 | 0.122 / 0.181 | **51.7%** | -0.198 / 0.741 / -0.046 |
| Hue rotation | Test | 0.199 / 0.258 | 0.138 / 0.177 | 27.3% | -0.087 / 0.801 / +0.034 |
|  | Comp | 0.194 / 0.241 | **0.177 / 0.202** | 25.8% | -0.081 / 0.763 / -0.044 |
| Resolution | Test | **0.842 / 0.857** | 0.486 / 0.507 | **94.5%** | -0.009 / 0.647 / +0.000 |
|  | Comp | 0.701 / 0.710 | **0.535 / 0.550** | 82.5% | -0.034 / 0.638 / -0.043 |
| Brightness | Test | **0.725 / 0.756** | **0.437 / 0.462** | **87.4%** | -0.021 / 0.705 / -0.012 |
|  | Comp | 0.416 / 0.448 | 0.359 / 0.387 | 70.1% | -0.055 / 0.704 / -0.037 |
| Motion blur | Test | **0.756 / 0.777** | **0.459 / 0.481** | **90.1%** | -0.022 / 0.611 / -0.019 |
|  | Comp | 0.429 / 0.439 | 0.365 / 0.377 | 56.6% | -0.075 / 0.683 / -0.083 |
| Horizontal stretch | Test | **0.260 / 0.290** | **0.189 / 0.213** | **57.1%** | -0.072 / 0.732 / -0.052 |
|  | Comp | 0.195 / 0.214 | 0.178 / 0.196 | 53.2% | -0.090 / 0.709 / -0.102 |

Two effects should be read together. First, competition accuracy can be higher at mild settings because its untransformed baseline is higher, particularly for YOLOv26_v2. Second, competition consistency and TP retention fall faster as brightness, blur, resolution loss, or stretch becomes severe. This shows that baseline difficulty alone does not explain the transformation gap.

## Model selection by rule

| Rule | Test consistency leader | Comp consistency leader | Test accuracy leader | Comp accuracy leader | Confidence observation |
|---|---|---|---|---|---|
| Grayscale | YOLOv8 | YOLOv12 | YOLOv26 | YOLOv26 | YOLOv26_v2 has the smallest competition retained-TP drop (-0.136). |
| Hue rotation | YOLOv26 | YOLOv26 | YOLOv26 | YOLOv26 | Winner varies by angle; YOLOv26 is the best overall compromise. |
| Resolution | YOLOv5 | YOLOv5 | YOLOv26/YOLOv26_v2 | YOLOv26_v2 | YOLOv8 has the smallest average retained-TP confidence loss, but accuracy favours the v26 models. |
| Brightness | YOLOv12 | YOLOv5 | YOLOv26 | YOLOv26 | YOLOv26_v2 preserves confidence best on competition, but has weak consistency. |
| Motion blur | YOLOv5 | YOLOv5 | YOLOv26_v2 | YOLOv26_v2 overall; YOLOv5 at kernel 11-21 | YOLOv5 best preserves competition retained-TP confidence on average. |
| Stretch | YOLOv12 | YOLOv26 | YOLOv26 | YOLOv26 | YOLOv26_v2 preserves retained-TP confidence best, despite weaker consistency. |

This split is the central result: use **accuracy leaders** for deployment selection and **consistency/confidence leaders** to diagnose sensitivity. Do not substitute a high consistency score for labelled accuracy.

## Rule-by-rule parameter findings

In the setting tables, each AP/AR cell is `test -> comp`. The leader cells are `test / comp` and identify the highest combined AP/AR model at that setting.

### Grayscale

| Setting | Consistency AP/AR | Accuracy AP/AR | TP retained | Retained-TP delta confidence | Consistency leader | Accuracy leader |
|---|---:|---:|---:|---:|---|---|
| Grayscale | 0.143/0.204 -> 0.144/0.225 | 0.100/0.139 -> 0.122/0.181 | 38.9% -> 51.7% | -0.214 -> -0.198 | YOLOv8 / YOLOv12 | YOLOv26 / YOLOv26 |

Grayscale is the only family where competition is not harsher on average. It retains more TPs and slightly more AP/AR, although both datasets remain poor. The model ranking does not transfer for consistency: YOLOv8 leads test, while YOLOv12 leads competition. YOLOv26 is the more transferable accuracy choice. YOLOv26_v2 preserves confidence best on retained competition TPs, but this does not translate into the highest AP.

### Hue rotation (`hue_rotation`)

| Degrees | Consistency AP/AR | Accuracy AP/AR | TP retained | Retained-TP delta confidence | Consistency leader | Accuracy leader |
|---:|---:|---:|---:|---:|---|---|
| 0 | 0.990/0.995 -> 0.983/0.986 | 0.518/0.538 -> 0.678/0.695 | 99.8% -> 99.7% | -0.001 -> -0.001 | YOLOv8 / YOLOv5 | YOLOv26_v2 / YOLOv26_v2 |
| 20 | 0.569/0.619 -> 0.663/0.682 | 0.350/0.385 -> 0.586/0.608 | 86.8% -> 93.3% | -0.031 -> -0.028 | YOLOv12 / YOLOv5 | YOLOv26 / YOLOv26_v2 |
| 40 | 0.175/0.239 -> 0.128/0.157 | 0.125/0.170 -> 0.115/0.146 | 36.4% -> 37.6% | -0.166 -> -0.157 | YOLOv26 / YOLOv12 | YOLOv26 / YOLOv26 |
| 60 | 0.071/0.100 -> 0.003/0.011 | 0.065/0.086 -> 0.004/0.011 | 4.9% -> 2.0% | -0.223 -> -0.111 | YOLOv26 / YOLOv26 | YOLOv26 / YOLOv26 |
| 80 | 0.070/0.094 -> 0.019/0.038 | 0.065/0.083 -> 0.021/0.041 | 3.8% -> 1.7% | -0.204 -> -0.133 | YOLOv5 / YOLOv26 | YOLOv26 / YOLOv26 |
| 100 | 0.029/0.036 -> 0.030/0.034 | 0.027/0.031 -> 0.030/0.034 | 2.1% -> 1.3% | -0.239 -> -0.188 | YOLOv5 / YOLOv12 | YOLOv5 / YOLOv12 |
| 120 | 0.018/0.046 -> 0.027/0.070 | 0.016/0.038 -> 0.028/0.072 | 3.2% -> 2.1% | -0.266 -> -0.316 | YOLOv12 / YOLOv26 | YOLOv26 / YOLOv26 |
| 140 | 0.181/0.291 -> 0.236/0.351 | 0.147/0.217 -> 0.236/0.279 | 21.1% -> 22.1% | -0.184 -> -0.203 | YOLOv26 / YOLOv26 | YOLOv26 / YOLOv26_v2 |
| 160 | 0.476/0.640 -> 0.446/0.584 | 0.311/0.407 -> 0.397/0.428 | 60.1% -> 46.3% | -0.053 -> -0.048 | YOLOv26 / YOLOv26 | YOLOv26 / YOLOv26_v2 |
| 180 | 0.990/0.995 -> 0.983/0.986 | 0.518/0.538 -> 0.678/0.695 | 99.8% -> 99.7% | -0.001 -> -0.001 | YOLOv8 / YOLOv5 | YOLOv26_v2 / YOLOv26_v2 |

The wrapped U-shaped curve is almost identical across domains: good at 0/180, degraded at 20/160, near collapse from 60-120, and a partial recovery at 140. Cross-dataset correlations are 0.986 for consistency and 0.965 for accuracy. Retained-confidence patterns also correlate strongly (0.820).

YOLOv26 is the most transferable accuracy model and leads both datasets at 60 degrees and test at 40, 80, 120, 140, and 160. YOLOv26_v2 benefits from its strong competition baseline and leads competition accuracy at 20, 140, and 160 degrees. At 60-120 degrees, however, all models are effectively outside a safe operating region. Lost TPs in this family were originally very confident (mean 0.801 test and 0.763 competition), so these are substantive misses rather than marginal detections.

### Resolution (`size_ratio`)

| Ratio | Consistency AP/AR | Accuracy AP/AR | TP retained | Retained-TP delta confidence | Consistency leader | Accuracy leader |
|---:|---:|---:|---:|---:|---|---|
| 0.1 | 0.433/0.457 -> 0.091/0.095 | 0.299/0.316 -> 0.077/0.080 | 69.4% -> 26.8% | -0.079 -> -0.161 | YOLOv5 / YOLOv5 | YOLOv26_v2 / YOLOv26 |
| 0.2 | 0.715/0.738 -> 0.447/0.461 | 0.461/0.482 -> 0.416/0.432 | 91.2% -> 67.5% | -0.026 -> -0.096 | YOLOv5 / YOLOv12 | YOLOv26_v2 / YOLOv26 |
| 0.3 | 0.816/0.835 -> 0.640/0.655 | 0.501/0.523 -> 0.533/0.549 | 96.0% -> 83.8% | -0.008 -> -0.056 | YOLOv5 / YOLOv5 | YOLOv26_v2 / YOLOv26 |
| 0.4 | 0.880/0.896 -> 0.769/0.785 | 0.515/0.536 -> 0.597/0.615 | 98.0% -> 90.5% | -0.001 -> -0.035 | YOLOv5 / YOLOv5 | YOLOv26 / YOLOv26_v2 |
| 0.5 | 0.922/0.934 -> 1.000/1.000 | 0.520/0.541 -> 0.679/0.697 | 98.8% -> 100.0% | +0.002 -> +0.000 | YOLOv5 / tie | YOLOv26 / YOLOv26_v2 |
| 0.6 | 0.936/0.947 -> 0.880/0.888 | 0.520/0.541 -> 0.653/0.671 | 99.1% -> 96.4% | +0.002 -> -0.015 | YOLOv5 / YOLOv5 | YOLOv26 / YOLOv26_v2 |
| 0.7 | 0.949/0.958 -> 0.902/0.910 | 0.521/0.542 -> 0.662/0.680 | 99.2% -> 97.6% | +0.003 -> -0.010 | YOLOv5 / YOLOv5 | YOLOv26_v2 / YOLOv26_v2 |
| 0.8 | 0.959/0.967 -> 0.925/0.930 | 0.521/0.542 -> 0.668/0.685 | 99.3% -> 98.4% | +0.003 -> -0.008 | YOLOv5 / YOLOv5 | YOLOv26 / YOLOv26_v2 |
| 0.9 | 0.973/0.979 -> 0.953/0.957 | 0.520/0.541 -> 0.672/0.690 | 99.5% -> 98.9% | +0.002 -> -0.004 | YOLOv8 / YOLOv5 | YOLOv26_v2 / YOLOv26_v2 |
| 1.0 | 0.993/0.998 -> 1.000/1.000 | 0.518/0.538 -> 0.679/0.697 | 99.9% -> 100.0% | -0.000 -> +0.000 | YOLOv5 / tie | YOLOv26_v2 / YOLOv26_v2 |

Resolution degradation has a transferable shape (consistency correlation 0.953; accuracy 0.924; retained-confidence change 0.933), but `comp_labelled` is much more vulnerable below ratio 0.4. At ratio 0.1, only 26.8% of competition TPs remain and their confidence falls by another 0.161 on average.

YOLOv5 is the clearest consistency choice, particularly at ratio 0.1 and 0.3-0.8. Absolute accuracy tells a different story: YOLOv26/YOLOv26_v2 lead all settings. For deployment, this supports a v26-family default and YOLOv5 only as a repeatability specialist after checking that its absolute accuracy is adequate.

The exact competition consistency score of 1.000/1.000 at ratio 0.5 is a preprocessing equivalence, not credible evidence that halving resolution is lossless. The 1280x720 competition source becomes 640x360 at ratio 0.5, aligning with the model's effective inference resize; original and transformed detection coordinates/confidences are identical apart from scaling. This point is excluded from averages and should be retested by downscaling and then upscaling back to a fixed input canvas.

### Brightness (`gamma`)

Gamma below 1 brightens midtones; gamma above 1 darkens them.

| Gamma | Consistency AP/AR | Accuracy AP/AR | TP retained | Retained-TP delta confidence | Consistency leader | Accuracy leader |
|---:|---:|---:|---:|---:|---|---|
| 0.5 | 0.898/0.913 -> 0.815/0.835 | 0.507/0.527 -> 0.625/0.643 | 97.5% -> 90.5% | -0.013 -> -0.040 | YOLOv5 / YOLOv12 | YOLOv26_v2 / YOLOv26_v2 |
| 1.0 | 0.993/0.998 -> 1.000/1.000 | 0.518/0.538 -> 0.679/0.697 | 99.9% -> 100.0% | -0.000 -> +0.000 | YOLOv5 / tie | YOLOv26_v2 / YOLOv26_v2 |
| 1.5 | 0.944/0.954 -> 0.843/0.859 | 0.514/0.534 -> 0.659/0.681 | 98.9% -> 98.3% | +0.001 -> +0.010 | YOLOv12 / YOLOv5 | YOLOv26_v2 / YOLOv26_v2 |
| 2.0 | 0.871/0.890 -> 0.693/0.721 | 0.501/0.521 -> 0.599/0.625 | 97.6% -> 95.6% | -0.003 -> -0.005 | YOLOv12 / YOLOv5 | YOLOv26_v2 / YOLOv26_v2 |
| 2.5 | 0.801/0.826 -> 0.525/0.566 | 0.481/0.503 -> 0.490/0.524 | 95.2% -> 91.0% | -0.013 -> -0.031 | YOLOv12 / YOLOv5 | YOLOv26 / YOLOv26 |
| 3.0 | 0.726/0.759 -> 0.354/0.406 | 0.451/0.475 -> 0.341/0.383 | 90.7% -> 81.3% | -0.023 -> -0.066 | YOLOv12 / YOLOv5 | YOLOv26 / YOLOv26 |
| 3.5 | 0.655/0.693 -> 0.227/0.271 | 0.416/0.444 -> 0.223/0.259 | 84.8% -> 66.3% | -0.031 -> -0.097 | YOLOv12 / YOLOv5 | YOLOv26 / YOLOv26 |
| 4.0 | 0.594/0.637 -> 0.147/0.186 | 0.383/0.412 -> 0.147/0.183 | 79.2% -> 50.7% | -0.036 -> -0.137 | YOLOv12 / YOLOv12 | YOLOv26 / YOLOv26 |
| 4.5 | 0.543/0.591 -> 0.090/0.118 | 0.354/0.386 -> 0.091/0.119 | 74.1% -> 34.8% | -0.042 -> -0.167 | YOLOv12 / YOLOv12 | YOLOv26 / YOLOv26 |
| 5.0 | 0.492/0.542 -> 0.053/0.070 | 0.325/0.358 -> 0.053/0.070 | 68.5% -> 22.0% | -0.046 -> -0.200 | YOLOv12 / YOLOv5 | YOLOv26 / YOLOv26 |

Curve shape transfers well (consistency correlation 0.977; accuracy 0.920), but competition performance falls much faster after gamma 2.0. At gamma 5, competition loses 78% of original TPs; even the retained detections lose 0.200 confidence. Persistent competition FPs also fall by 0.190 confidence, but that apparent benefit is dominated by the much larger true-positive loss.

The model role is stable but split by metric: YOLOv12 leads test consistency through most dark settings, YOLOv5 leads competition consistency at most settings, and YOLOv26 leads absolute accuracy from gamma 2.5 onward on both datasets. YOLOv26_v2 leads accuracy at gamma 0.5-2.0 because of its stronger baseline, and generally has the smallest retained-confidence loss on dark competition images. Gamma 4-5 should be treated as an unsafe operating range for every model.

### Motion blur (`kernel_size`)

| Kernel | Consistency AP/AR | Accuracy AP/AR | TP retained | Retained-TP delta confidence | Consistency leader | Accuracy leader |
|---:|---:|---:|---:|---:|---|---|
| 3 | 0.969/0.975 -> 0.897/0.904 | 0.520/0.541 -> 0.658/0.677 | 99.4% -> 96.8% | +0.002 -> -0.014 | YOLOv5 / YOLOv5 | YOLOv26_v2 / YOLOv26_v2 |
| 5 | 0.931/0.942 -> 0.769/0.783 | 0.519/0.540 -> 0.597/0.616 | 98.8% -> 89.9% | +0.001 -> -0.037 | YOLOv5 / YOLOv5 | YOLOv26_v2 / YOLOv26_v2 |
| 7 | 0.890/0.904 -> 0.620/0.634 | 0.513/0.534 -> 0.515/0.533 | 97.7% -> 79.8% | -0.003 -> -0.057 | YOLOv5 / YOLOv5 | YOLOv26_v2 / YOLOv26_v2 |
| 9 | 0.844/0.862 -> 0.492/0.505 | 0.502/0.523 -> 0.446/0.461 | 96.0% -> 68.6% | -0.008 -> -0.075 | YOLOv5 / YOLOv5 | YOLOv26_v2 / YOLOv26_v2 |
| 11 | 0.795/0.817 -> 0.405/0.418 | 0.486/0.508 -> 0.382/0.396 | 93.8% -> 58.0% | -0.016 -> -0.094 | YOLOv5 / YOLOv5 | YOLOv26_v2 / YOLOv5 |
| 13 | 0.739/0.764 -> 0.342/0.354 | 0.465/0.486 -> 0.325/0.336 | 90.7% -> 48.7% | -0.024 -> -0.111 | YOLOv5 / YOLOv5 | YOLOv26_v2 / YOLOv5 |
| 15 | 0.681/0.708 -> 0.259/0.268 | 0.438/0.461 -> 0.248/0.255 | 87.2% -> 40.5% | -0.033 -> -0.129 | YOLOv5 / YOLOv5 | YOLOv26_v2 / YOLOv5 |
| 17 | 0.624/0.652 -> 0.210/0.219 | 0.412/0.434 -> 0.199/0.209 | 83.5% -> 33.1% | -0.043 -> -0.137 | YOLOv5 / YOLOv5 | YOLOv26_v2 / YOLOv5 |
| 19 | 0.569/0.598 -> 0.169/0.175 | 0.382/0.404 -> 0.162/0.168 | 79.3% -> 27.5% | -0.053 -> -0.142 | YOLOv5 / YOLOv5 | YOLOv26_v2 / YOLOv5 |
| 21 | 0.517/0.547 -> 0.125/0.129 | 0.352/0.374 -> 0.117/0.121 | 74.7% -> 23.0% | -0.063 -> -0.143 | YOLOv5 / YOLOv5 | YOLOv26_v2 / YOLOv5 |

YOLOv5 has the strongest consistency result in the entire assessment: it leads AP/AR consistency at every kernel size on both datasets. The accuracy result is more nuanced. YOLOv26_v2 leads all test settings and competition kernels 3-9; YOLOv5 becomes the competition accuracy leader at kernels 11-21. This is one of the few cases where the consistency specialist also becomes the best labelled-accuracy choice under severe competition degradation.

Competition blur is much harsher: TP retention falls below 50% at kernel 13, compared with remaining above 74% throughout test. Lost competition TPs were originally more confident than lost test TPs (0.683 vs 0.611 overall), and persistent FP confidence falls much more (-0.083 vs -0.019). The latter does not indicate improved behaviour because the transformation is simultaneously removing far more correct detections. Cross-dataset correlations are still strong enough to transfer severity ordering: 0.931 consistency, 0.833 accuracy, and 0.828 retained-confidence change.

### Horizontal stretch (`width_ratio`, height fixed at 1.0)

| Width ratio | Consistency AP/AR | Accuracy AP/AR | TP retained | Retained-TP delta confidence | Consistency leader | Accuracy leader |
|---:|---:|---:|---:|---:|---|---|
| 1.0 | 0.993/0.998 -> 1.000/1.000 | 0.518/0.538 -> 0.679/0.697 | 99.9% -> 100.0% | -0.000 -> +0.000 | YOLOv5 / tie | YOLOv26_v2 / YOLOv26_v2 |
| 1.2 | 0.660/0.681 -> 0.585/0.608 | 0.455/0.475 -> 0.535/0.557 | 88.1% -> 89.3% | -0.020 -> -0.017 | YOLOv26_v2 / YOLOv26 | YOLOv26 / YOLOv26_v2 |
| 1.4 | 0.492/0.517 -> 0.429/0.455 | 0.349/0.371 -> 0.386/0.408 | 83.1% -> 81.9% | -0.039 -> -0.038 | YOLOv12 / YOLOv26 | YOLOv26 / YOLOv26 |
| 1.6 | 0.370/0.394 -> 0.318/0.342 | 0.271/0.291 -> 0.276/0.302 | 76.3% -> 77.6% | -0.051 -> -0.062 | YOLOv12 / YOLOv26 | YOLOv26 / YOLOv26 |
| 1.8 | 0.282/0.310 -> 0.214/0.246 | 0.210/0.234 -> 0.183/0.211 | 69.7% -> 73.3% | -0.066 -> -0.090 | YOLOv12 / YOLOv8 | YOLOv26 / YOLOv26 |
| 2.0 | 0.206/0.238 -> 0.110/0.132 | 0.157/0.182 -> 0.115/0.136 | 61.9% -> 63.1% | -0.082 -> -0.123 | YOLOv12 / YOLOv26 | YOLOv26 / YOLOv26 |
| 2.2 | 0.136/0.173 -> 0.058/0.079 | 0.104/0.132 -> 0.062/0.083 | 46.2% -> 47.8% | -0.100 -> -0.158 | YOLOv12 / YOLOv26 | YOLOv26 / YOLOv26 |
| 2.4 | 0.092/0.131 -> 0.027/0.042 | 0.072/0.101 -> 0.030/0.046 | 36.7% -> 26.4% | -0.126 -> -0.223 | YOLOv12 / YOLOv26 | YOLOv26 / YOLOv26 |
| 2.6 | 0.062/0.099 -> 0.008/0.016 | 0.050/0.077 -> 0.009/0.015 | 30.5% -> 12.0% | -0.161 -> -0.283 | YOLOv12 / YOLOv8 | YOLOv26 / YOLOv26 |
| 2.8 | 0.037/0.069 -> 0.005/0.010 | 0.031/0.054 -> 0.004/0.008 | 21.5% -> 7.6% | -0.203 -> -0.318 | YOLOv12 / YOLOv8 | YOLOv26 / YOLOv8 |

Stretch has the most transferable curve shape (correlations 0.987 consistency, 0.984 accuracy, and 0.975 retained-confidence change), but not the most transferable model ranking. YOLOv12 leads test consistency from 1.4 through 2.8. Competition instead favours YOLOv26 at 1.2-1.6 and 2.0-2.4, with YOLOv8 taking 1.8 and the already-collapsed 2.6-2.8 region.

YOLOv26 is the stable accuracy choice: it leads test from 1.2-2.8 and competition from 1.4-2.6. The YOLOv8 competition win at ratio 2.8 is not operationally meaningful because mean accuracy has fallen to 0.004/0.008. Competition stretch becomes especially destructive after ratio 2.2: TP retention drops to 26.4% at 2.4 and 7.6% at 2.8, while retained confidence falls by 0.223 and 0.318.

## Similarities and differences

### Similarities

1. Parameter severity ordering is highly transferable. Every continuously varying rule has a strong positive cross-dataset correlation in consistency, accuracy, and retained-confidence response.
2. AP and AR move together. AR is normally slightly above AP, but the same failure thresholds and most setting-level leaders appear in both metrics.
3. YOLOv5's blur consistency advantage is universal, and its low-resolution consistency is also stable.
4. YOLOv26 is consistently the strongest transformed-vs-GT choice for hue and moderate-to-extreme brightness, and for horizontal stretch.
5. Lost TPs are not marginal. Their original confidence is typically 0.61-0.80, showing that transformations remove detections that models originally considered credible.

### Differences

1. Competition brightness and blur have much steeper degradation. The test split substantially understates both TP loss and retained-confidence decline at severe parameters.
2. Competition resolution below 0.4 is more damaging, with the largest divergence at ratio 0.1.
3. Stretch curve shape transfers, but consistency leadership changes from YOLOv12 on test to YOLOv26 on most competition settings.
4. Grayscale is anomalous: competition retains more TPs and scores slightly higher, while its consistency winner changes from YOLOv8 to YOLOv12.
5. Persistent FP confidence usually falls more on competition. This should not be read as an improvement when TP retention is falling at the same time; blur, stretch, and extreme darkness suppress both correct and incorrect detections.
6. The stronger competition baselines, especially YOLOv26_v2, can make transformed-vs-GT AP/AR higher even when competition consistency is worse. Baseline-normalised degradation and raw transformed accuracy must both be considered.

## Practical recommendations

1. Use **YOLOv26 or YOLOv26_v2 as the accuracy-oriented default**, selected against the expected parameter range. YOLOv26 is more consistently robust across transformed ground truth; YOLOv26_v2 is exceptionally accurate on untransformed and mildly transformed competition images.
2. Use **YOLOv5 as the blur specialist**, especially for competition kernels 11-21, where it leads both consistency and absolute accuracy.
3. For reduced resolution, prefer **v26-family accuracy** in production, while using YOLOv5 as the consistency benchmark. Treat ratios below 0.3 as a degraded mode, particularly on competition imagery.
4. For brightness, use **YOLOv26** once gamma reaches 2.5. Gamma 4-5 needs image enhancement, exposure control, or an explicit abstention/fallback policy; model switching alone is insufficient.
5. For stretch, prefer **YOLOv26** for labelled accuracy and moderate competition distortion. Do not transfer YOLOv12's test consistency lead directly to competition deployment.
6. Treat hue 60-120, competition blur above kernel 13, competition stretch above 2.2, and gamma above 4 as out-of-envelope conditions requiring a fallback rather than a nominal model-selection decision.
7. Retest resolution with a fixed model input canvas so the ratio-0.5 preprocessing equivalence cannot masquerade as perfect robustness.

## Limitations

- The datasets differ in size, scenes, object density, and baseline accuracy; this is a domain comparison, not a paired-image statistical test.
- AP/AR values are point estimates. The JSONs contain no bootstrap confidence intervals, so small model gaps should not be treated as statistically significant.
- Confidence summaries depend on category-aware greedy GT matching and an IoU threshold of 0.5. Different matching thresholds can change counts near the boundary.
- Persistent-FP analysis only covers false positives that can be spatially matched before and after transformation; newly created and fully removed FPs are not represented by the delta mean.
- Family averages weight each AP/AR setting equally, while confidence means weight observations. They answer related but not identical aggregation questions.
