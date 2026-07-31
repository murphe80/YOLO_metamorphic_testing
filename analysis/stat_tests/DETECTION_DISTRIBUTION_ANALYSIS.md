# Detection Distribution Statistical Analysis

## Question and design

This analysis compares `test` with `val`, `train_samples` for every complete model, metamorphic test, and configured setting.

Within each image, original detections were matched to transformed detections using the repository's confidence-prioritized, one-to-one greedy matcher with a strict IoU > 0.5 criterion. Geometric detections used the existing coordinate-corrected files.

Each image contributed one three-count cluster:

- **Retained:** matched original/transformed detections.
- **Lost:** unmatched original detections.
- **Gained:** unmatched transformed detections.

The primary test was a Pearson 2×3 homogeneity statistic calibrated with 9,999 whole-image label permutations. Keeping each image cluster intact avoids treating detections from the same image as independent. Total variation distance and Cramér's V are reported as effect sizes.

Benjamini–Hochberg correction was applied with scope `model` at q ≤ 0.05. A non-significant result means **no difference was detected**; it does not establish that two distributions are equivalent.

## Candidate statistical tests

1. **Image-clustered permutation Pearson test (used).** Directly tests retained/lost/gained proportions, preserves within-image clustering, and has the clearest interpretation for this question.
2. **Multivariate energy-distance permutation test.** Can detect any difference in normalized per-image outcome distributions, but is less directly interpretable and needs a convention for zero-outcome images.
3. **PERMANOVA on per-image compositions.** Supports multivariate outcome profiles, but significance can reflect unequal within-dataset dispersion rather than a location difference.

## Coverage

- Models: `yolov12`, `yolov26`, `yolov5`, `yolov8`
- Base metamorphic tests: 6
- Configured settings per model: 51
- Dataset comparisons: 408
- Images per dataset/model/setting: 363

## Overall results

| Comparison | Tests | Different | No difference detected | Different proportion | Median TV |
| --- | --- | --- | --- | --- | --- |
| train_samples vs test | 204 | 0 | 204 | 0.0% | 0.0072 |
| val vs test | 204 | 1 | 203 | 0.5% | 0.0077 |

### Joint classification

The denominator is 204 model-setting combinations (4 models × 51 settings).

| Classification | Count | Proportion |
| --- | --- | --- |
| different for both | 0 | 0.0% |
| different only for val | 1 | 0.5% |
| different only for train samples | 0 | 0.0% |
| no difference detected for either | 203 | 99.5% |

## Key findings

- **`train_samples` versus `test`:** 0 of 204 comparisons (0.0%) showed an FDR-adjusted difference.
- **`val` versus `test`:** 1 of 204 comparisons (0.5%) showed an FDR-adjusted difference.
- Only 1 comparison remained significant after correction. Its effect size should be considered alongside the p/q values.
- The largest observed TV distance for `val` was 0.0351 (`yolov5`, `imgResolution_0`; q=0.8022); this was not significant.
- The largest observed TV distance for `train_samples` was 0.0300 (`yolov8`, `imgDarkBright_5`; q=0.9789); this was not significant.

### FDR-significant comparisons

| Target | Model | Variant | Test retained/lost/gained | Target retained/lost/gained | TV | Cramér's V | p | q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | yolov12 | imgMotionBlur_0 | 0.977/0.008/0.015 | 0.988/0.005/0.007 | 0.0111 | 0.0428 | 0.0003 | 0.0153 |

For `val` versus `test`, `yolov12` `imgMotionBlur_0` had +1.11 percentage points retained, -0.31 points lost, and -0.80 points gained. The small TV and Cramér's V values indicate that the statistically detectable difference is modest in magnitude.

## Results by model

| Target | Model | Different / total | Different proportion | Median TV | Maximum TV |
| --- | --- | --- | --- | --- | --- |
| train_samples | yolov12 | 0 / 51 | 0.0% | 0.0064 | 0.0237 |
| train_samples | yolov26 | 0 / 51 | 0.0% | 0.0062 | 0.0280 |
| train_samples | yolov5 | 0 / 51 | 0.0% | 0.0087 | 0.0242 |
| train_samples | yolov8 | 0 / 51 | 0.0% | 0.0070 | 0.0300 |
| val | yolov12 | 1 / 51 | 2.0% | 0.0084 | 0.0282 |
| val | yolov26 | 0 / 51 | 0.0% | 0.0079 | 0.0297 |
| val | yolov5 | 0 / 51 | 0.0% | 0.0077 | 0.0351 |
| val | yolov8 | 0 / 51 | 0.0% | 0.0065 | 0.0300 |

## Results by metamorphic test

| Target | Transformation | Different / total | Different proportion | Median TV |
| --- | --- | --- | --- | --- |
| train_samples | imgColorWheel | 0 / 40 | 0.0% | 0.0059 |
| train_samples | imgDarkBright | 0 / 40 | 0.0% | 0.0085 |
| train_samples | imgGrayscale | 0 / 4 | 0.0% | 0.0080 |
| train_samples | imgMotionBlur | 0 / 40 | 0.0% | 0.0061 |
| train_samples | imgResolution | 0 / 40 | 0.0% | 0.0059 |
| train_samples | imgStretch | 0 / 40 | 0.0% | 0.0116 |
| val | imgColorWheel | 0 / 40 | 0.0% | 0.0074 |
| val | imgDarkBright | 0 / 40 | 0.0% | 0.0092 |
| val | imgGrayscale | 0 / 4 | 0.0% | 0.0089 |
| val | imgMotionBlur | 1 / 40 | 2.5% | 0.0075 |
| val | imgResolution | 0 / 40 | 0.0% | 0.0062 |
| val | imgStretch | 0 / 40 | 0.0% | 0.0080 |

## Worked example: `imgStretch_2`

Setting 2 has `height_ratio=1.0` and `width_ratio=1.4`. Outcome proportions and corrected decisions are:

| Target | Model | Test retained/lost/gained | Target retained/lost/gained | TV | p | q | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train_samples | yolov12 | 0.728/0.204/0.068 | 0.729/0.215/0.056 | 0.0113 | 0.2257 | 0.9271 | no difference detected |
| train_samples | yolov26 | 0.780/0.153/0.067 | 0.799/0.147/0.054 | 0.0191 | 0.1036 | 0.9767 | no difference detected |
| train_samples | yolov5 | 0.736/0.207/0.058 | 0.733/0.215/0.052 | 0.0089 | 0.5291 | 0.8282 | no difference detected |
| train_samples | yolov8 | 0.715/0.213/0.072 | 0.715/0.226/0.060 | 0.0125 | 0.1683 | 0.9789 | no difference detected |
| val | yolov12 | 0.728/0.204/0.068 | 0.732/0.219/0.049 | 0.0189 | 0.0164 | 0.2788 | no difference detected |
| val | yolov26 | 0.780/0.153/0.067 | 0.788/0.154/0.058 | 0.0083 | 0.3897 | 0.9666 | no difference detected |
| val | yolov5 | 0.736/0.207/0.058 | 0.729/0.218/0.053 | 0.0113 | 0.47 | 0.8531 | no difference detected |
| val | yolov8 | 0.715/0.213/0.072 | 0.717/0.222/0.061 | 0.0112 | 0.2547 | 0.996 | no difference detected |

## Largest effect sizes (regardless of significance)

These rows are ranked by total variation distance, not by p-value. They are useful for practical follow-up but should not be called statistically different unless the final column says `different`.

| Target | Model | Variant | TV | Cramér's V | p | q | Retained Δ pp | Lost Δ pp | Gained Δ pp | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | yolov5 | imgResolution_0 | 0.0351 | 0.0373 | 0.2201 | 0.8022 | -3.51 | +2.83 | +0.68 | no difference detected |
| val | yolov8 | imgResolution_0 | 0.0300 | 0.0332 | 0.2916 | 0.996 | -2.99 | +3.00 | -0.01 | no difference detected |
| train_samples | yolov8 | imgDarkBright_5 | 0.0300 | 0.0422 | 0.118 | 0.9789 | -3.00 | +2.12 | +0.88 | no difference detected |
| val | yolov26 | imgResolution_0 | 0.0297 | 0.0312 | 0.2799 | 0.9666 | -2.61 | +2.97 | -0.36 | no difference detected |
| val | yolov12 | imgResolution_0 | 0.0282 | 0.0318 | 0.307 | 0.7456 | -2.43 | +2.82 | -0.39 | no difference detected |
| train_samples | yolov8 | imgColorWheel_6 | 0.0282 | 0.0398 | 0.0707 | 0.9789 | +1.25 | -2.82 | +1.57 | no difference detected |
| train_samples | yolov26 | imgStretch_3 | 0.0280 | 0.0360 | 0.0496 | 0.9767 | +2.80 | -1.12 | -1.67 | no difference detected |
| val | yolov12 | imgStretch_4 | 0.0275 | 0.0371 | 0.0354 | 0.4514 | -1.36 | +2.75 | -1.39 | no difference detected |
| val | yolov5 | imgDarkBright_5 | 0.0266 | 0.0437 | 0.0949 | 0.7686 | -2.64 | +2.66 | -0.02 | no difference detected |
| val | yolov5 | imgDarkBright_6 | 0.0258 | 0.0370 | 0.2144 | 0.8022 | -2.08 | +2.58 | -0.50 | no difference detected |
| val | yolov26 | imgDarkBright_5 | 0.0253 | 0.0364 | 0.1333 | 0.9666 | -2.53 | +2.26 | +0.27 | no difference detected |
| train_samples | yolov8 | imgStretch_4 | 0.0245 | 0.0299 | 0.1188 | 0.9789 | -1.51 | +2.45 | -0.94 | no difference detected |
| train_samples | yolov5 | imgColorWheel_5 | 0.0242 | 0.0321 | 0.1119 | 0.4091 | -2.42 | +1.54 | +0.89 | no difference detected |
| train_samples | yolov5 | imgColorWheel_6 | 0.0242 | 0.0310 | 0.1711 | 0.575 | +1.55 | -2.42 | +0.87 | no difference detected |
| train_samples | yolov5 | imgStretch_4 | 0.0241 | 0.0302 | 0.1123 | 0.4091 | -1.45 | +2.41 | -0.96 | no difference detected |

## Interpretation limits

- The test addresses the retained/lost/gained composition, not semantic class agreement or model accuracy against ground truth.
- Matching is class-agnostic because that is the behavior of the current original-to-transformed matcher.
- A significant result says the dataset's metamorphic detection response differs; it does not identify dataset quality as the cause.
- A non-significant result is not an equivalence result. A separate equivalence margin would be needed to claim practical sameness.
- Monte Carlo p-values have minimum resolution 0.0001.

## Reproduction

```bash
venv/bin/python analysis/utils/detection_distribution_tests.py --iou-threshold 0.5 --permutations 9999 --alpha 0.05 --correction-scope model
```

Detailed results are in `detection_distribution_tests.csv`; the machine-readable aggregate summary is in `detection_distribution_summary.json`.

## Statistical references

- Székely, G. J. and Rizzo, M. L. (2013), [Energy statistics: A class of statistics based on distances](https://doi.org/10.1016/j.jspi.2013.03.018).
- Anderson, M. J. (2001), [A new method for non-parametric multivariate analysis of variance](https://doi.org/10.1111/j.1442-9993.2001.tb00081.x).
- Benjamini, Y. and Hochberg, Y. (1995), [Controlling the false discovery rate](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).
