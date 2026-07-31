# Big Orange Cone Confidence Analysis

Generated on 2026-07-17 from `/moredata/emma/metamorphic_v2/YOLO_metamorphic_testing/results/test`.

## Scope and method

Big Orange Cone was selected using zero-indexed class ID `2`. Every detection `.txt` file below each model directory was scanned recursively. The expected row format was `class_id confidence x1 y1 x2 y2`; JSON/COCO exports and visualization files were excluded.

The main statistics are detection-weighted: every Big Orange Cone detection is one observation, including detections from the original images and all metamorphic transformation settings. Variance and standard deviation in the main tables are population measures (`ddof=0`). The detection files already reflect the confidence threshold used during inference, so this report describes saved detections rather than all pre-threshold model candidates.

## Overall results

| Model | Detections | Files with class / scanned | Mean | Median | Variance | Std. dev. | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | 13,005 | 4,613 / 18,876 | 0.800152 | 0.870017 | 0.031025 | 0.176138 | 0.250079 | 0.972049 |
| YOLOv8 | 12,822 | 4,559 / 18,876 | 0.817352 | 0.885672 | 0.029040 | 0.170413 | 0.250030 | 0.972430 |
| YOLOv12 | 12,921 | 4,695 / 18,876 | 0.813242 | 0.882140 | 0.028613 | 0.169153 | 0.250200 | 0.968283 |
| YOLOv26 | 14,302 | 4,548 / 18,876 | 0.756329 | 0.860307 | 0.048559 | 0.220362 | 0.250046 | 0.982757 |

## Distribution detail

| Model | P05 | P25 | P75 | P95 | IQR | Confidence >= 0.50 | Confidence >= 0.75 | Confidence >= 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | 0.368934 | 0.750447 | 0.919291 | 0.954981 | 0.168844 | 90.24% | 75.07% | 34.44% |
| YOLOv8 | 0.387741 | 0.784876 | 0.926758 | 0.957007 | 0.141883 | 91.22% | 78.47% | 39.91% |
| YOLOv12 | 0.387693 | 0.784751 | 0.922328 | 0.951957 | 0.137577 | 91.20% | 78.00% | 37.98% |
| YOLOv26 | 0.305686 | 0.604189 | 0.927616 | 0.964917 | 0.323428 | 81.44% | 65.19% | 38.84% |

## Results by transformation family

These rows pool all numbered settings within a transformation family. `original` is kept separate as the untransformed baseline.

| Model | Family | Detections | Mean | Median | Variance | Std. dev. |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | imgColorWheel | 2,514 | 0.713714 | 0.794340 | 0.045957 | 0.214375 |
| YOLOv5 | imgDarkBright | 2,719 | 0.828406 | 0.883897 | 0.024845 | 0.157622 |
| YOLOv5 | imgGrayscale | 66 | 0.599111 | 0.641510 | 0.053900 | 0.232164 |
| YOLOv5 | imgMotionBlur | 2,600 | 0.841313 | 0.884721 | 0.019773 | 0.140618 |
| YOLOv5 | imgResolution | 2,636 | 0.833025 | 0.883004 | 0.021863 | 0.147861 |
| YOLOv5 | imgStretch | 2,195 | 0.775782 | 0.849473 | 0.032088 | 0.179132 |
| YOLOv5 | original | 275 | 0.849469 | 0.889335 | 0.016392 | 0.128032 |
| YOLOv8 | imgColorWheel | 2,073 | 0.742089 | 0.844991 | 0.046853 | 0.216456 |
| YOLOv8 | imgDarkBright | 2,669 | 0.846615 | 0.896171 | 0.022560 | 0.150200 |
| YOLOv8 | imgGrayscale | 131 | 0.633757 | 0.665119 | 0.039064 | 0.197646 |
| YOLOv8 | imgMotionBlur | 2,673 | 0.845138 | 0.892788 | 0.020728 | 0.143972 |
| YOLOv8 | imgResolution | 2,653 | 0.848771 | 0.895261 | 0.019615 | 0.140053 |
| YOLOv8 | imgStretch | 2,346 | 0.788236 | 0.860643 | 0.030842 | 0.175618 |
| YOLOv8 | original | 277 | 0.863008 | 0.901220 | 0.015917 | 0.126163 |
| YOLOv12 | imgColorWheel | 2,273 | 0.736632 | 0.832158 | 0.045357 | 0.212971 |
| YOLOv12 | imgDarkBright | 2,647 | 0.835128 | 0.889017 | 0.023508 | 0.153323 |
| YOLOv12 | imgGrayscale | 108 | 0.640913 | 0.697699 | 0.039413 | 0.198526 |
| YOLOv12 | imgMotionBlur | 2,633 | 0.841106 | 0.890058 | 0.019956 | 0.141266 |
| YOLOv12 | imgResolution | 2,615 | 0.843128 | 0.892839 | 0.019469 | 0.139531 |
| YOLOv12 | imgStretch | 2,371 | 0.801058 | 0.873380 | 0.029245 | 0.171011 |
| YOLOv12 | original | 274 | 0.857692 | 0.897598 | 0.015482 | 0.124427 |
| YOLOv26 | imgColorWheel | 2,576 | 0.672765 | 0.738268 | 0.058180 | 0.241205 |
| YOLOv26 | imgDarkBright | 3,115 | 0.778054 | 0.882973 | 0.046686 | 0.216069 |
| YOLOv26 | imgGrayscale | 83 | 0.608615 | 0.624489 | 0.058617 | 0.242110 |
| YOLOv26 | imgMotionBlur | 2,830 | 0.791859 | 0.881317 | 0.039886 | 0.199715 |
| YOLOv26 | imgResolution | 2,871 | 0.798815 | 0.891230 | 0.039863 | 0.199657 |
| YOLOv26 | imgStretch | 2,524 | 0.725344 | 0.817819 | 0.048281 | 0.219730 |
| YOLOv26 | original | 303 | 0.807545 | 0.899508 | 0.039288 | 0.198212 |

## File-level and parsing diagnostics

| Model | Files scanned | Empty files | Files with detections | Files with class 2 | Valid rows (all classes) | Malformed rows | Out-of-range rows | Mean of per-file class-2 means |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | 18,876 | 118 | 18,758 | 4,613 | 179,659 | 0 | 0 | 0.773221 |
| YOLOv8 | 18,876 | 122 | 18,754 | 4,559 | 183,229 | 0 | 0 | 0.787610 |
| YOLOv12 | 18,876 | 126 | 18,750 | 4,695 | 181,912 | 0 | 0 | 0.787345 |
| YOLOv26 | 18,876 | 84 | 18,792 | 4,548 | 286,634 | 0 | 0 | 0.742575 |

The final column gives every class-2-positive file equal weight, unlike the detection-weighted main mean. Empty detection files are valid outputs and indicate that the model saved no detections for that image/setting.

## Summary

- Highest mean confidence: **YOLOv8** (0.817352).
- Highest median confidence: **YOLOv8** (0.885672).
- Lowest confidence variance: **YOLOv12** (0.028613).
- Most saved Big Orange Cone detections: **YOLOv26** (14,302).

Confidence is not accuracy: a model with a higher mean can be more confident without being better calibrated or more correct. Ground-truth matching is required to separate true positives from false positives and assess calibration.

## Reproduction

```bash
python3 analysis/analyse_big_orange_cone_confidence.py
```

The input path, output path, models, and class ID can be changed with `--results-dir`, `--output`, `--models`, and `--class-id`.
