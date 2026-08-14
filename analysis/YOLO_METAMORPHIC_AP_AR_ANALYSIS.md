# AP and AR performance of YOLO models under metamorphic transformations

## Executive summary

The most important result is that **absolute accuracy and metamorphic self-consistency select different models**.

- Against ground truth (`transformed_vs_gt`), **YOLO26 and YOLO26_v2 are the only defensible default choices**. Across the 46 non-control settings, YOLO26 has the best combined AP/AR at 27 settings and YOLO26_v2 at 18. YOLOv5 wins the remaining setting (hue 100), but all models have effectively failed there.
- YOLO26 is the stronger choice for **hue changes, grayscale, severe darkness, and horizontal stretch**. YOLO26_v2 is clearly preferable for **motion blur**. The two are effectively tied under **resolution reduction**.
- In self-consistency (`original_vs_transformed`), older models often appear stronger: YOLOv5 is most consistent under blur and resolution loss, YOLOv12 under brightness and stretch, and YOLOv8 under grayscale. These are not accuracy wins. The identity-condition AP/AR of YOLOv5 is only 0.470/0.487, compared with 0.583/0.608 for YOLO26 and 0.584/0.610 for YOLO26_v2.
- By absolute family-average AP/AR, the hardest tests are **grayscale (0.100/0.139), hue rotation (0.138/0.177), and horizontal stretch (0.189/0.213)**. The worst individual settings are hue 120, hue 100, and width ratios 2.8 and 2.6.
- Color-test severity needs special interpretation. The detector classes include yellow, blue, and orange cone categories. Global hue rotation changes class-defining color, while grayscale removes it. Their very low scores therefore measure both model weakness and loss/change of label semantics. They should not be used as ordinary label-preserving fine-tuning augmentations.
- The clearest effect of the alternative YOLO26 training method is a shift toward **blur robustness**. YOLO26_v2 improves blur-family accuracy by +0.020 AP and +0.021 AR and wins all ten blur settings; its advantage increases with kernel size. YOLO26 remains better on color transformations, grayscale, severe darkness, and every non-control stretch setting.

The deployment recommendation is consequently:

1. Use **YOLO26** as the general-purpose model when hue/lighting variation, grayscale inputs, or aspect distortion are important.
2. Use **YOLO26_v2** when motion blur is a prominent operating condition.
3. Treat resolution performance as a tie and choose between the two using latency, another held-out set, or repeated-run uncertainty.
4. Do not select YOLOv5/v8/v12 solely from their higher self-consistency unless their lower absolute accuracy is acceptable for the application.

## Data and interpretation

This report uses:

- [`metrics_original_vs_transformed.json`](results/test/parameter_variations/metrics_original_vs_transformed.json), which compares detections on an original image with detections on its transformed version. It measures **prediction repeatability/self-consistency**, not correctness.
- [`metrics_transformed_vs_gt.json`](results/test/parameter_variations/metrics_transformed_vs_gt.json), which compares transformed-image detections with ground truth. It measures **absolute detection performance** under the transformation.

AP is COCO AP averaged over IoU thresholds 0.50:0.95. AR is COCO AR at `maxDets=100` (AR100). Higher is better for both.

Family averages exclude identity/control settings: hue 0 and 180, gamma 1.0, resolution ratio 1.0, and stretch ratio 1.0. Hue 180 is an identity because the implementation wraps OpenCV hue modulo 180. All blur settings and the single grayscale test are retained. The overall macro-average weights each of the six transformation families equally, rather than allowing families with more parameter values to dominate.

The files contain aggregate AP and AR only. They do not provide per-image uncertainty, confidence intervals, or repeated training runs. Differences below roughly 0.005 should be treated as practical ties, not evidence of a reliable training-method effect.

## Absolute performance against ground truth

Each cell is mean `AP / AR` over the non-control settings in that family. Bold indicates the best model in the row.

| Transformation | YOLOv5 | YOLOv8 | YOLOv12 | YOLO26 | YOLO26_v2 |
|---|---:|---:|---:|---:|---:|
| Hue rotation | 0.127 / 0.162 | 0.124 / 0.156 | 0.128 / 0.161 | **0.163 / 0.211** | 0.150 / 0.195 |
| Gamma brightness | 0.398 / 0.419 | 0.406 / 0.426 | 0.405 / 0.425 | **0.491 / 0.524** | 0.483 / 0.517 |
| Grayscale | 0.078 / 0.103 | 0.107 / 0.146 | 0.105 / 0.136 | **0.113 / 0.164** | 0.098 / 0.149 |
| Motion blur | 0.429 / 0.447 | 0.435 / 0.453 | 0.435 / 0.451 | 0.488 / 0.515 | **0.508 / 0.536** |
| Resolution reduction | 0.442 / 0.459 | 0.449 / 0.466 | 0.449 / 0.465 | 0.546 / 0.573 | **0.546 / 0.573** |
| Horizontal stretch | 0.160 / 0.182 | 0.170 / 0.193 | 0.190 / 0.209 | **0.221 / 0.252** | 0.201 / 0.229 |
| **Macro-average** | 0.272 / 0.295 | 0.282 / 0.307 | 0.285 / 0.308 | **0.337 / 0.373** | 0.331 / 0.367 |

YOLO26 has the best family-average AP and AR in four of six families. YOLO26_v2 leads blur and is numerically—but not materially—ahead on resolution. The YOLO26 macro-average advantage over YOLO26_v2 is 0.006 AP and 0.007 AR.

### AP versus AR behavior

AP and AR move together closely, so there is little evidence for a broad precision-versus-recall model trade-off. The model with the highest AP also has the highest AR in 44 of the 46 non-control settings. The two exceptions are hue 80 (YOLO26 AP, YOLOv5 AR) and resolution ratio 0.5 (YOLO26 AP, YOLO26_v2 AR); neither changes the combined model recommendation.

AR is systematically higher than AP. The largest family-level gaps occur for grayscale and hue rotation (about 0.039 averaged across models), followed by stretch (0.024). This indicates that some objects remain detectable, but correct classification, localization across IoU thresholds, precision, or confidence ranking degrades. Aggregate COCO metrics cannot isolate which of these mechanisms is responsible, so per-class confusion and localization-error analysis would be the appropriate next diagnostic.

### Identity-condition accuracy

The following values average the five equivalent identity results present in the parameter files.

| Model | Identity AP | Identity AR |
|---|---:|---:|
| YOLOv5 | 0.470 | 0.487 |
| YOLOv8 | 0.477 | 0.495 |
| YOLOv12 | 0.475 | 0.492 |
| YOLO26 | 0.583 | 0.608 |
| YOLO26_v2 | **0.584** | **0.610** |

This baseline explains why the newer models dominate transformed-vs-ground-truth comparisons even where an older model is more self-consistent. A model can reproduce its original predictions very reliably while reproducing original errors as well.

## When to use one model over another

### Accuracy-oriented selection

| Operating scenario | Preferred model | Evidence and qualification |
|---|---|---|
| General mixed transformations | **YOLO26** | Highest equally weighted macro-average: 0.337 AP / 0.373 AR. |
| Motion blur, kernels 3–21 | **YOLO26_v2** | Highest AP and AR at every tested kernel; 0.508/0.536 family mean. |
| Reduced resolution | **YOLO26 or YOLO26_v2** | Family means differ by only +0.000 AP/+0.001 AR in favor of v2. This is a tie at the precision supported by these results. |
| Mild gamma change: 0.5, 1.5, or 2.0 | **YOLO26_v2**, weak preference | v2 wins these settings, but its largest AP/AR advantage is only 0.004/0.006. |
| Darker inputs: gamma 2.5–5.0 | **YOLO26** | Wins every setting; the advantage over v2 grows with darkness to 0.019 AP/0.021 AR at gamma 5.0. |
| Horizontal stretch: width 1.2–2.8 | **YOLO26** | Highest AP and AR at all nine non-control ratios. |
| Hue rotation | **YOLO26** | Best combined AP/AR at seven of eight non-control settings. The exception, hue 100, is a failure condition for every model. |
| Grayscale | **YOLO26** | Best AP/AR at 0.113/0.164, although absolute performance remains poor. |

There is no labeled-accuracy scenario in these files where YOLOv8 or YOLOv12 wins. YOLOv5 wins hue 100 at 0.038 AP/0.042 AR, but this is not operationally useful: it is merely the highest score in a universal failure region.

### Consistency-oriented selection

Self-consistency can matter when output stability under small input changes is itself a requirement. The family means are:

| Transformation | YOLOv5 | YOLOv8 | YOLOv12 | YOLO26 | YOLO26_v2 |
|---|---:|---:|---:|---:|---:|
| Hue rotation | 0.196 / 0.258 | 0.192 / 0.248 | 0.198 / 0.258 | **0.209 / 0.271** | 0.198 / 0.255 |
| Gamma brightness | 0.738 / 0.769 | 0.740 / 0.769 | **0.746 / 0.775** | 0.707 / 0.740 | 0.692 / 0.727 |
| Grayscale | 0.121 / 0.170 | **0.167 / 0.235** | 0.166 / 0.227 | 0.134 / 0.199 | 0.127 / 0.189 |
| Motion blur | **0.810 / 0.830** | 0.794 / 0.816 | 0.795 / 0.814 | 0.670 / 0.691 | 0.712 / 0.734 |
| Resolution reduction | **0.870 / 0.883** | 0.863 / 0.876 | 0.864 / 0.877 | 0.798 / 0.815 | 0.816 / 0.833 |
| Horizontal stretch | 0.247 / 0.280 | 0.259 / 0.291 | **0.290 / 0.319** | 0.256 / 0.288 | 0.246 / 0.273 |
| **Macro-average** | 0.497 / 0.532 | 0.503 / 0.539 | **0.510 / 0.545** | 0.462 / 0.501 | 0.465 / 0.502 |

The resulting specialist choices are YOLOv5 for blur and low resolution, YOLOv12 for brightness and stretch, YOLOv8 for grayscale, and YOLO26 for hue. These choices should only override the accuracy-oriented recommendation if repeatability is more important than correctness or a separate ground-truth evaluation establishes sufficient accuracy.

## Hardest metamorphic tests

### Severity by transformation family

These values average all five models and all non-control settings in a family.

| Rank | Transformation | Mean AP | Mean AR | Interpretation |
|---:|---|---:|---:|---|
| 1 | Grayscale | **0.100** | **0.139** | Hardest family, but it removes class-defining color information. |
| 2 | Hue rotation | 0.138 | 0.177 | Severe class/color shift; middle rotations nearly eliminate performance. |
| 3 | Horizontal stretch | 0.189 | 0.213 | Hardest clearly label-preserving geometric transformation. |
| 4 | Gamma brightness | 0.437 | 0.462 | Gradual deterioration as gamma increases. |
| 5 | Motion blur | 0.459 | 0.481 | Gradual deterioration with kernel size; v2 degrades least. |
| 6 | Resolution reduction | 0.486 | 0.507 | Severe mainly at ratios 0.1–0.2; otherwise YOLO26 variants remain near baseline. |

### Selected difficult individual settings

| Transformation setting | Mean AP/AR across models | Best model and AP/AR |
|---|---:|---:|
| Hue 120 | **0.016 / 0.038** | YOLO26, 0.020 / 0.042 |
| Hue 100 | 0.027 / 0.031 | YOLOv5, 0.038 / 0.042 |
| Stretch 2.8 | 0.031 / 0.054 | YOLO26, 0.047 / 0.074 |
| Stretch 2.6 | 0.050 / 0.077 | YOLO26, 0.071 / 0.102 |
| Hue 80 | 0.065 / 0.083 | YOLO26, 0.073 / 0.092 combined; YOLOv5 has slightly higher AR |
| Hue 60 | 0.065 / 0.086 | YOLO26, 0.099 / 0.129 |
| Stretch 2.4 | 0.072 / 0.101 | YOLO26, 0.096 / 0.131 |
| Grayscale | 0.100 / 0.139 | YOLO26, 0.113 / 0.164 |
| Resolution 0.1 | 0.299 / 0.317 | YOLO26_v2, 0.324 / 0.348 |
| Gamma 5.0 | 0.325 / 0.358 | YOLO26, 0.369 / 0.413 |
| Motion blur 21 | 0.352 / 0.374 | YOLO26_v2, 0.375 / 0.403 |

The table highlights the worst settings and the severe endpoints of each label-safe family; it is not an exhaustive ranking of all 46 settings. The hue and extreme-stretch scores are so low that model switching is not an adequate mitigation. They indicate either an invalid metamorphic relation (color), an out-of-distribution failure (extreme stretch), or both.

## Fine-tuning transformation recommendations

### Priority 1: label-safe, operationally plausible augmentations

1. **Horizontal stretch:** train over width ratios 1.2–2.4, with extra sampling around 1.8–2.4 where AP deteriorates rapidly. Include 2.6–2.8 only if distortions that extreme can occur in the real camera/preprocessing pipeline; otherwise they may teach unrealistic geometry.
2. **Motion blur:** train with kernels approximately 7–21 and sample multiple orientations. The evaluation tests only orientation 0, so fine-tuning solely on that orientation would overfit the test rather than address general motion blur. Emphasize kernels 13–21, where the absolute decline becomes large and the v2 training method shows its strongest gain.
3. **Low resolution:** include ratios 0.1–0.4, especially 0.1–0.2. A robust implementation should downscale to remove detail and then resize back to the normal network input dimensions so the augmentation tests information loss rather than a different inference resize path.
4. **Darkness:** include gamma 2.0–4.0, with a smaller proportion at 4.5–5.0. Very dark images may contain too little recoverable information; over-sampling them can hurt clean-image performance without creating usable detections. Gamma 0.5 should be retained as a bright-image case, but it is not a major weakness in these results.

Use a mixture of moderate and hard examples rather than training only at the worst endpoint. A sensible starting curriculum is roughly 50% clean/near-clean, 30% moderate transformations, and 20% hard transformations, followed by validation on untouched images and on held-out parameter values. This ratio is a starting hypothesis, not something identified by the two metric files.

### Priority 2: color transformations, with semantic safeguards

The class mapping in this repository contains color-specific categories such as yellow, blue, and orange cones. Consequently:

- **Do not fine-tune with arbitrary hue 40–160 while retaining the original class label.** The transformation may turn a yellow cone visually blue while the label still says yellow, directly injecting label noise.
- **Do not treat grayscale as an ordinary label-preserving classification augmentation.** It removes the evidence needed to distinguish color-defined classes.
- If the goal is class-agnostic cone localization, grayscale and wide hue shifts can be valuable after merging color categories into a single cone/unknown class.
- If color classification must be retained, prefer modest, physically plausible white-balance/hue/saturation jitter that does not cross category boundaries. For larger color changes, update the ground-truth category using an explicit color mapping and verify the transformed cone color.

Thus, hue and grayscale are high-priority diagnostic tests, but not automatically high-priority supervised augmentations with unchanged labels.

### Fine-tuning evaluation gates

After fine-tuning, accept a model only if it improves transformed-vs-ground-truth AP/AR without materially reducing identity AP/AR. Report at least:

- clean identity AP and AR;
- family-average AP and AR with controls excluded;
- per-parameter curves, because averages hide collapse at extreme settings;
- per-class AP/confusion for color transformations;
- results over multiple training seeds or bootstrap confidence intervals, especially for deltas below 0.01.

## YOLO26 versus YOLO26_v2

The following table shows `YOLO26_v2 − YOLO26` on non-control family means. Positive values favor v2.

| Transformation | Accuracy delta AP | Accuracy delta AR | Consistency delta AP | Consistency delta AR | Preferred training result |
|---|---:|---:|---:|---:|---|
| Hue rotation | -0.012 | -0.016 | -0.012 | -0.017 | YOLO26 |
| Gamma brightness | -0.008 | -0.006 | -0.015 | -0.012 | YOLO26 overall |
| Grayscale | -0.015 | -0.016 | -0.007 | -0.010 | YOLO26 |
| Motion blur | **+0.020** | **+0.021** | **+0.043** | **+0.042** | YOLO26_v2 |
| Resolution reduction | +0.000 | +0.001 | +0.018 | +0.019 | Accuracy tie; v2 more consistent |
| Horizontal stretch | -0.020 | -0.023 | -0.010 | -0.015 | YOLO26 |

### Stable patterns across parameter severity

- **Blur is the strongest v2 improvement.** YOLO26_v2 beats YOLO26 in AP and AR at all ten kernels. The accuracy advantage grows from +0.000 AP/+0.002 AR at kernel 3 to +0.047/+0.050 at kernel 21. Its consistency advantage similarly grows from about +0.016/+0.015 to +0.069/+0.070. This is both an accuracy and invariance gain, not merely better agreement with its original output.
- **Resolution accuracy is a tie.** v2 wins the combined score at ratios 0.1–0.3, 0.7, and 0.9; YOLO26 wins at 0.4–0.6 and 0.8. All AP/AR differences are below 0.006 and the family delta is nearly zero. In contrast, v2 is more self-consistent at every non-control ratio, by +0.018 AP/+0.019 AR on average. The alternative training therefore stabilizes resolution response without producing a clear ground-truth accuracy gain.
- **Stretch consistently favors YOLO26.** YOLO26 wins AP and AR at every width ratio from 1.2 through 2.8. Its advantage generally grows with severity, reaching approximately +0.030 AP/+0.034 AR at ratio 2.4 before both models approach the failure floor.
- **Hue favors YOLO26.** It wins seven of eight non-control settings. v2 wins at hue 100 by +0.004 AP/+0.002 AR, but both models are near zero; this is not a meaningful reversal. YOLO26's largest advantage occurs around hue 40, 60, and 140.
- **Brightness has a crossover.** v2 is marginally better at gamma 0.5, 1.5, and 2.0. YOLO26 becomes better at gamma 2.5 and its advantage increases monotonically through gamma 5.0. This suggests v2 handles normal-to-moderate photometric variation as well as YOLO26 but is less robust in very dark conditions.
- **Grayscale favors YOLO26** by about 0.015 AP and AR, although neither model is accurate enough to make grayscale color classification reliable.

Overall, the observed pattern is consistent with the v2 training method reallocating robustness toward blur and prediction stability under downsampling, with a trade-off in color robustness, severe darkness, and geometric distortion. This is an empirical association, not a causal conclusion: the files do not document the two training procedures, training seeds, or uncertainty. A controlled ablation with identical data splits, initialization policy, and multiple seeds is needed to attribute the differences specifically to the training method.

## Final recommendations

- **Default:** YOLO26, because it has the best macro-average absolute AP/AR and covers more transformation families robustly.
- **Blur-heavy deployment:** YOLO26_v2. This is the clearest scenario-specific model choice in the results.
- **Resolution-heavy deployment:** either YOLO26 checkpoint; prefer v2 only if repeatability is important, because its absolute resolution advantage is negligible.
- **Aspect distortion or severe darkness:** YOLO26.
- **Color-altered or grayscale imagery:** YOLO26 has the highest reported score, but the color-specific label semantics must be resolved before interpreting or fine-tuning on these transformations.
- **Fine-tuning order:** stretch first; then motion blur, low resolution, and darkness according to expected field frequency. Use hue/grayscale only with class-aware relabeling or a class-agnostic objective.
- **Older YOLO versions:** retain them as consistency baselines or potential resource/latency alternatives, not accuracy recommendations from these results. Their apparent blur/resolution stability does not compensate for the roughly 0.10–0.11 lower clean AP relative to YOLO26.
