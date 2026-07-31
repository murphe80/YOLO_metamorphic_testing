# Big Orange Cone True/False-Positive Confidence Analysis

Generated on 2026-07-17 from `/moredata/emma/metamorphic_v2/YOLO_metamorphic_testing/results/test` using ground truth `/moredata/emma/metamorphic_v2/YOLO_metamorphic_testing/data/base_images/test/annotations_coco.json`.

## Scope and matching method

Big Orange Cone is model class ID `2`, mapped to COCO detection category ID `4` and ground-truth category ID `4`. The ground truth contains 299 non-crowd Big Orange Cone instances across 363 images.

For each model and transformation setting, Big Orange Cone detections were sorted from highest to lowest confidence and greedily assigned to the unmatched Big Orange Cone ground-truth box with the greatest IoU above `0.50`. A matched detection is a true positive (TP); an unmatched detection is a false positive (FP). Each ground-truth box can be matched once. Unmatched ground-truth boxes are false negatives (FN) and have no detection confidence.

The script uses `detections_for_gt.json`, which is derived from each setting's detection `.txt` files and supplies ground-truth image IDs. For `imgStretch` and `imgResolution`, it uses `detections_for_gt_transformed.json`, whose boxes have been mapped back to original-image coordinates before IoU matching.

Statistics pool the original output and every metamorphic setting. Ground-truth instances therefore appear once per evaluated setting. Variance and standard deviation are population measures (`ddof=0`).

## Overall classification results

| Model | Settings | TP | FP | FN | Precision | Recall | F1 | TP mean conf. | FP mean conf. | Mean gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | 52 | 11,095 | 1,910 | 4,453 | 85.31% | 71.36% | 77.72% | 0.841279 | 0.561246 | 0.280033 |
| YOLOv8 | 52 | 11,213 | 1,609 | 4,335 | 87.45% | 72.12% | 79.05% | 0.852597 | 0.571732 | 0.280865 |
| YOLOv12 | 52 | 11,271 | 1,650 | 4,277 | 87.23% | 72.49% | 79.18% | 0.847809 | 0.577114 | 0.270696 |
| YOLOv26 | 52 | 11,684 | 2,618 | 3,864 | 81.69% | 75.15% | 78.28% | 0.824341 | 0.452792 | 0.371549 |

`Mean gap` is TP mean confidence minus FP mean confidence; a larger positive value indicates greater separation between the two groups.

## TP and FP confidence distributions

| Model | Outcome | N | Mean | Median | Variance | Std. dev. | Min | P05 | P25 | P75 | P95 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | True positive | 11,095 | 0.841279 | 0.883466 | 0.018167 | 0.134786 | 0.250128 | 0.524077 | 0.820213 | 0.925961 | 0.956691 | 0.972049 |
| YOLOv5 | False positive | 1,910 | 0.561246 | 0.549660 | 0.038812 | 0.197007 | 0.250079 | 0.271486 | 0.386992 | 0.735058 | 0.880634 | 0.948180 |
| YOLOv8 | True positive | 11,213 | 0.852597 | 0.894300 | 0.017388 | 0.131864 | 0.250714 | 0.544409 | 0.840007 | 0.932454 | 0.957647 | 0.972430 |
| YOLOv8 | False positive | 1,609 | 0.571732 | 0.565039 | 0.041260 | 0.203125 | 0.250030 | 0.276491 | 0.389846 | 0.741492 | 0.894093 | 0.946921 |
| YOLOv12 | True positive | 11,271 | 0.847809 | 0.891363 | 0.017539 | 0.132436 | 0.251250 | 0.534613 | 0.833881 | 0.926937 | 0.952951 | 0.968283 |
| YOLOv12 | False positive | 1,650 | 0.577114 | 0.570488 | 0.040335 | 0.200836 | 0.250200 | 0.277934 | 0.395800 | 0.750215 | 0.886143 | 0.949710 |
| YOLOv26 | True positive | 11,684 | 0.824341 | 0.893130 | 0.028182 | 0.167875 | 0.250052 | 0.418207 | 0.783763 | 0.936588 | 0.967061 | 0.982757 |
| YOLOv26 | False positive | 2,618 | 0.452792 | 0.408343 | 0.026723 | 0.163470 | 0.250046 | 0.262087 | 0.315769 | 0.558537 | 0.783263 | 0.937045 |

## Effect of applying higher confidence thresholds

TP and FP labels are fixed by the IoU matching above, then detections below each confidence threshold are removed. Recall remains relative to all ground-truth instances across the evaluated settings.

| Model | Threshold | TP retained | FP retained | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| YOLOv5 | 0.25 | 11,095 | 1,910 | 85.31% | 71.36% |
| YOLOv5 | 0.50 | 10,610 | 1,126 | 90.41% | 68.24% |
| YOLOv5 | 0.75 | 9,330 | 433 | 95.56% | 60.01% |
| YOLOv5 | 0.90 | 4,426 | 53 | 98.82% | 28.47% |
| YOLOv8 | 0.25 | 11,213 | 1,609 | 87.45% | 72.12% |
| YOLOv8 | 0.50 | 10,761 | 935 | 92.01% | 69.21% |
| YOLOv8 | 0.75 | 9,676 | 386 | 96.16% | 62.23% |
| YOLOv8 | 0.90 | 5,051 | 66 | 98.71% | 32.49% |
| YOLOv12 | 0.25 | 11,271 | 1,650 | 87.23% | 72.49% |
| YOLOv12 | 0.50 | 10,793 | 991 | 91.59% | 69.42% |
| YOLOv12 | 0.75 | 9,663 | 415 | 95.88% | 62.15% |
| YOLOv12 | 0.90 | 4,844 | 64 | 98.70% | 31.16% |
| YOLOv26 | 0.25 | 11,684 | 2,618 | 81.69% | 75.15% |
| YOLOv26 | 0.50 | 10,751 | 896 | 92.31% | 69.15% |
| YOLOv26 | 0.75 | 9,154 | 170 | 98.18% | 58.88% |
| YOLOv26 | 0.90 | 5,536 | 19 | 99.66% | 35.61% |

## Results by transformation family

| Model | Family | Settings | TP | FP | FN | Precision | Recall | TP mean conf. | FP mean conf. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | imgColorWheel | 10 | 1,611 | 903 | 1,379 | 64.08% | 53.88% | 0.808830 | 0.544023 |
| YOLOv5 | imgDarkBright | 10 | 2,451 | 268 | 539 | 90.14% | 81.97% | 0.858762 | 0.550784 |
| YOLOv5 | imgGrayscale | 1 | 60 | 6 | 239 | 90.91% | 20.07% | 0.629636 | 0.293856 |
| YOLOv5 | imgMotionBlur | 10 | 2,470 | 130 | 520 | 95.00% | 82.61% | 0.856507 | 0.552627 |
| YOLOv5 | imgResolution | 10 | 2,480 | 156 | 510 | 94.08% | 82.94% | 0.850925 | 0.548460 |
| YOLOv5 | imgStretch | 10 | 1,763 | 432 | 1,227 | 80.32% | 58.96% | 0.815357 | 0.614278 |
| YOLOv5 | original | 1 | 260 | 15 | 39 | 94.55% | 86.96% | 0.865457 | 0.572351 |
| YOLOv8 | imgColorWheel | 10 | 1,521 | 552 | 1,469 | 73.37% | 50.87% | 0.822028 | 0.521823 |
| YOLOv8 | imgDarkBright | 10 | 2,462 | 207 | 528 | 92.24% | 82.34% | 0.873158 | 0.530918 |
| YOLOv8 | imgGrayscale | 1 | 112 | 19 | 187 | 85.50% | 37.46% | 0.655571 | 0.505170 |
| YOLOv8 | imgMotionBlur | 10 | 2,504 | 169 | 486 | 93.68% | 83.75% | 0.865262 | 0.546973 |
| YOLOv8 | imgResolution | 10 | 2,487 | 166 | 503 | 93.74% | 83.18% | 0.867182 | 0.572943 |
| YOLOv8 | imgStretch | 10 | 1,865 | 481 | 1,125 | 79.50% | 62.37% | 0.822343 | 0.655989 |
| YOLOv8 | original | 1 | 262 | 15 | 37 | 94.58% | 87.63% | 0.876942 | 0.619627 |
| YOLOv12 | imgColorWheel | 10 | 1,584 | 689 | 1,406 | 69.69% | 52.98% | 0.821417 | 0.541713 |
| YOLOv12 | imgDarkBright | 10 | 2,416 | 231 | 574 | 91.27% | 80.80% | 0.859028 | 0.585162 |
| YOLOv12 | imgGrayscale | 1 | 91 | 17 | 208 | 84.26% | 30.43% | 0.678333 | 0.440607 |
| YOLOv12 | imgMotionBlur | 10 | 2,489 | 144 | 501 | 94.53% | 83.24% | 0.857806 | 0.552453 |
| YOLOv12 | imgResolution | 10 | 2,486 | 129 | 504 | 95.07% | 83.14% | 0.855767 | 0.599558 |
| YOLOv12 | imgStretch | 10 | 1,945 | 426 | 1,045 | 82.03% | 65.05% | 0.837204 | 0.636021 |
| YOLOv12 | original | 1 | 260 | 14 | 39 | 94.89% | 86.96% | 0.871207 | 0.606695 |
| YOLOv26 | imgColorWheel | 10 | 1,727 | 849 | 1,263 | 67.04% | 57.76% | 0.783912 | 0.446674 |
| YOLOv26 | imgDarkBright | 10 | 2,555 | 560 | 435 | 82.02% | 85.45% | 0.844035 | 0.477013 |
| YOLOv26 | imgGrayscale | 1 | 76 | 7 | 223 | 91.57% | 25.42% | 0.630480 | 0.371229 |
| YOLOv26 | imgMotionBlur | 10 | 2,515 | 315 | 475 | 88.87% | 84.11% | 0.836527 | 0.435227 |
| YOLOv26 | imgResolution | 10 | 2,586 | 285 | 404 | 90.07% | 86.49% | 0.841504 | 0.411465 |
| YOLOv26 | imgStretch | 10 | 1,955 | 569 | 1,035 | 77.46% | 65.38% | 0.799423 | 0.470822 |
| YOLOv26 | original | 1 | 270 | 33 | 29 | 89.11% | 90.30% | 0.853673 | 0.430138 |

## Matching diagnostics

| Model | Settings evaluated | Missing files | Invalid target detections | Unknown-image detections | Mean matched IoU | Median matched IoU |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | 52 | 0 | 0 | 0 | 0.868731 | 0.904112 |
| YOLOv8 | 52 | 0 | 0 | 0 | 0.867652 | 0.900817 |
| YOLOv12 | 52 | 0 | 0 | 0 | 0.872344 | 0.908384 |
| YOLOv26 | 52 | 0 | 0 | 0 | 0.868674 | 0.905232 |

Invalid Big Orange Cone ground-truth annotations excluded: `0`.

## Summary

- Largest TP-versus-FP mean confidence gap: **YOLOv26** (0.371549).
- Highest saved-detection precision: **YOLOv8** (87.45%).
- Highest recall: **YOLOv26** (75.15%).

A false positive here is class-specific: it can be a duplicate, a poorly localized box, a background detection, or a Big Orange Cone classification on an object belonging to another cone class. Inspecting those error types separately would require an additional class-agnostic matching analysis.

## Reproduction

```bash
python3 analysis/analyse_big_orange_cone_tp_fp_confidence.py
```

Use `--iou-threshold` to change the matching criterion. Input locations, models, and category mappings are also configurable through command-line options.
