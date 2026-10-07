# RQ2b Holdout Brightness Validation: yolov8

## Design

Only original holdout images with per-image COCO AP@[0.50:0.95] ≥ 0.50 were eligible. For each eligible image, the full RQ2a brute-force gamma grid was evaluated. All transformations were retained, including AP at or above the threshold; no interpolation or AP-based selection was used.

The operating ranges were fixed before validation using the unadjusted RQ2b boundary means: lower = dark/right mean and upper = bright/left mean. Eligible originals are successful controls.

Images checked: **139**; eligible originals: **73**; transformations: **14673**; transformed successes: **1671**; transformed failures: **13002**.

## Fixed mean-boundary ranges

| Method | Dark mean | Dark SD | Lower | Bright mean | Bright SD | Upper |
|---|---:|---:|---:|---:|---:|---:|
| luminance_rec601 | 0.09184 | 0.069861 | 0.09184 | 0.59159 | 0.074345 | 0.59159 |
| luminance_rec709 | 0.092894 | 0.071031 | 0.092894 | 0.59205 | 0.074276 | 0.59205 |
| hsv_value | 0.1176 | 0.079354 | 0.1176 | 0.60008 | 0.071761 | 0.60008 |
| hsi_lightness | 0.092845 | 0.068599 | 0.092845 | 0.59048 | 0.074315 | 0.59048 |
| hsi_intensity | 0.092266 | 0.06949 | 0.092266 | 0.59085 | 0.074293 | 0.59085 |
| cielab_lightness | 9.6572 | 7.5341 | 9.6572 | 59.742 | 7.1259 | 59.742 |
| rms_contrast | 0.17696 | 0.080554 | 0.17696 | 0.43385 | 0.039207 | 0.43385 |
| mean_shift_separation_spread_ratio | 139.98 | 490.31 | 139.98 | 3637.7 | 5479.8 | 3637.7 |

## Validation results

| Rank | Method | Failure recall | Bright recall | Dark recall | Original acceptance | Transformed success acceptance | All success acceptance | Balanced accuracy | MCC | TP | FP | TN | FN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | rms_contrast | 0.926 | 0.909 | 0.926 | 1.000 | 0.884 | 0.889 | 0.907 | 0.699 | 12036 | 194 | 1550 | 966 |
| 2 | luminance_rec709 | 0.893 | 0.953 | 0.892 | 1.000 | 0.882 | 0.887 | 0.890 | 0.630 | 11608 | 197 | 1547 | 1394 |
| 3 | hsv_value | 0.903 | 0.978 | 0.902 | 1.000 | 0.849 | 0.856 | 0.879 | 0.630 | 11747 | 252 | 1492 | 1255 |
| 4 | luminance_rec601 | 0.891 | 0.953 | 0.890 | 1.000 | 0.882 | 0.886 | 0.889 | 0.627 | 11587 | 198 | 1546 | 1415 |
| 5 | hsi_intensity | 0.890 | 0.953 | 0.888 | 1.000 | 0.880 | 0.885 | 0.887 | 0.623 | 11567 | 201 | 1543 | 1435 |
| 6 | hsi_lightness | 0.890 | 0.953 | 0.889 | 1.000 | 0.879 | 0.884 | 0.887 | 0.622 | 11569 | 203 | 1541 | 1433 |
| 7 | cielab_lightness | 0.896 | 0.978 | 0.894 | 1.000 | 0.853 | 0.860 | 0.878 | 0.617 | 11646 | 245 | 1499 | 1356 |
| 8 | mean_shift_separation_spread_ratio | 0.875 | 0.996 | 0.873 | 0.000 | 0.000 | 0.000 | 0.437 | -0.129 | 11375 | 1744 | 0 | 1627 |

## Classification by transformation direction

Accuracy is the fraction correctly classified: AP below the threshold and outside the operating range, or AP at/above the threshold and inside it. Each table uses only its stated direction; original images are excluded. Balanced accuracy averages successful-image acceptance and failure rejection.

### Brightened images (gamma < 1)

Images: **730**; successes: **498**; failures: **232** (AP threshold: 0.50).

| Method | Correct / total | Accuracy | Balanced accuracy | Successful images accepted | Failures rejected |
|---|---:|---:|---:|---:|---:|
| luminance_rec601 | 521 / 730 | 71.37% | 77.75% | 60.24% | 95.26% |
| luminance_rec709 | 522 / 730 | 71.51% | 77.85% | 60.44% | 95.26% |
| hsv_value | 473 / 730 | 64.79% | 73.62% | 49.40% | 97.84% |
| hsi_lightness | 516 / 730 | 70.68% | 77.25% | 59.24% | 95.26% |
| hsi_intensity | 518 / 730 | 70.96% | 77.45% | 59.64% | 95.26% |
| cielab_lightness | 480 / 730 | 65.75% | 74.32% | 50.80% | 97.84% |
| rms_contrast | 600 / 730 | 82.19% | 84.53% | 78.11% | 90.95% |
| mean_shift_separation_spread_ratio | 231 / 730 | 31.64% | 49.78% | 0.00% | 99.57% |

### Darkened images (gamma > 1)

Images: **13943**; successes: **1173**; failures: **12770** (AP threshold: 0.50).

| Method | Correct / total | Accuracy | Balanced accuracy | Successful images accepted | Failures rejected |
|---|---:|---:|---:|---:|---:|
| luminance_rec601 | 12539 / 13943 | 89.93% | 94.50% | 100.00% | 89.01% |
| luminance_rec709 | 12560 / 13943 | 90.08% | 94.58% | 100.00% | 89.17% |
| hsv_value | 12693 / 13943 | 91.03% | 95.11% | 100.00% | 90.21% |
| hsi_lightness | 12521 / 13943 | 89.80% | 94.43% | 100.00% | 88.86% |
| hsi_intensity | 12519 / 13943 | 89.79% | 94.42% | 100.00% | 88.85% |
| cielab_lightness | 12592 / 13943 | 90.31% | 94.71% | 100.00% | 89.42% |
| rms_contrast | 12913 / 13943 | 92.61% | 92.68% | 92.75% | 92.60% |
| mean_shift_separation_spread_ratio | 11144 / 13943 | 79.93% | 43.63% | 0.00% | 87.27% |


## RMS contrast

RMS rejected **12036 of 13002** transformed failures and accepted **1550 of 1744** successful images (original and transformed). Failure recall: **0.926**; transformed success acceptance: **0.884**; MCC: **0.699**.

Success acceptance is the fraction of images with AP at or above the threshold whose brightness score is inside the operating range. Balanced accuracy uses failure recall and acceptance of all successful images. TP/FP/TN/FN include originals and every transformation.

## Sample images

Representative too-dark, in-range, and out-of-range images for each valid method are in `sample_images/<method>`. Each method directory contains a JSON manifest with the fixed bounds and image metadata. The out-of-range set includes both dark and bright violations and therefore intentionally overlaps the too-dark set.

![Validation plot](brightness_validation.png)

## Confusion matrices

Failure (AP below the threshold) is the positive class. Rows show actual failure/success; columns show predicted failure (outside range)/success (inside range). Cells show counts and percentages within each actual class; colour uses the same 0–100% scale for every method. Overall includes eligible originals and all transformations; directional plots exclude originals.

![Overall confusion matrices](confusion_matrices_overall.png)

![Brightened-only confusion matrices](confusion_matrices_brightened.png)

![Darkened-only confusion matrices](confusion_matrices_darkened.png)
