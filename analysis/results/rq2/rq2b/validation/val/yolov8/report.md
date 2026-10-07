# RQ2b Brightness Validation: val / yolov8

## Design

Only original validation images with per-image COCO AP@[0.50:0.95] ≥ 0.50 were eligible. For each eligible image, the full RQ2a brute-force gamma grid was evaluated. All transformations were retained, including AP at or above the threshold; no interpolation or AP-based selection was used.

The operating ranges were fixed before validation using the unadjusted RQ2b boundary means: lower = dark/right mean and upper = bright/left mean. Eligible originals are successful controls.

Images checked: **363**; eligible originals: **235**; transformations: **47235**; transformed successes: **17368**; transformed failures: **29867**.

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
| 1 | luminance_rec709 | 0.905 | 0.649 | 0.909 | 0.996 | 0.562 | 0.568 | 0.736 | 0.514 | 27036 | 7613 | 9990 | 2831 |
| 2 | luminance_rec601 | 0.905 | 0.647 | 0.908 | 1.000 | 0.558 | 0.564 | 0.734 | 0.510 | 27017 | 7682 | 9921 | 2850 |
| 3 | cielab_lightness | 0.904 | 0.655 | 0.907 | 1.000 | 0.557 | 0.563 | 0.733 | 0.509 | 27001 | 7694 | 9909 | 2866 |
| 4 | hsi_intensity | 0.898 | 0.647 | 0.901 | 1.000 | 0.545 | 0.551 | 0.724 | 0.489 | 26813 | 7901 | 9702 | 3054 |
| 5 | hsi_lightness | 0.893 | 0.647 | 0.896 | 1.000 | 0.544 | 0.550 | 0.721 | 0.480 | 26660 | 7926 | 9677 | 3207 |
| 6 | hsv_value | 0.880 | 0.652 | 0.883 | 0.991 | 0.561 | 0.567 | 0.723 | 0.477 | 26271 | 7622 | 9981 | 3596 |
| 7 | rms_contrast | 0.718 | 0.745 | 0.718 | 0.923 | 0.683 | 0.686 | 0.702 | 0.395 | 21456 | 5519 | 12084 | 8411 |
| 8 | mean_shift_separation_spread_ratio | 0.538 | 0.673 | 0.537 | 0.043 | 0.248 | 0.245 | 0.392 | -0.215 | 16082 | 13285 | 4318 | 13785 |

## Classification by transformation direction

Accuracy is the fraction correctly classified: AP below the threshold and outside the operating range, or AP at/above the threshold and inside it. Each table uses only its stated direction; original images are excluded. Balanced accuracy averages successful-image acceptance and failure rejection.

### Brightened images (gamma < 1)

Images: **2350**; successes: **1962**; failures: **388** (AP threshold: 0.50).

| Method | Correct / total | Accuracy | Balanced accuracy | Successful images accepted | Failures rejected |
|---|---:|---:|---:|---:|---:|
| luminance_rec601 | 2064 / 2350 | 87.83% | 78.55% | 92.41% | 64.69% |
| luminance_rec709 | 2065 / 2350 | 87.87% | 78.68% | 92.41% | 64.95% |
| hsv_value | 2061 / 2350 | 87.70% | 78.68% | 92.15% | 65.21% |
| hsi_lightness | 2069 / 2350 | 88.04% | 78.68% | 92.66% | 64.69% |
| hsi_intensity | 2068 / 2350 | 88.00% | 78.65% | 92.61% | 64.69% |
| cielab_lightness | 2059 / 2350 | 87.62% | 78.73% | 92.00% | 65.46% |
| rms_contrast | 2074 / 2350 | 88.26% | 82.73% | 90.98% | 74.48% |
| mean_shift_separation_spread_ratio | 1152 / 2350 | 49.02% | 56.34% | 45.41% | 67.27% |

### Darkened images (gamma > 1)

Images: **44885**; successes: **15406**; failures: **29479** (AP threshold: 0.50).

| Method | Correct / total | Accuracy | Balanced accuracy | Successful images accepted | Failures rejected |
|---|---:|---:|---:|---:|---:|
| luminance_rec601 | 34639 / 44885 | 77.17% | 70.95% | 51.10% | 90.80% |
| luminance_rec709 | 34727 / 44885 | 77.37% | 71.21% | 51.56% | 90.86% |
| hsv_value | 33958 / 44885 | 75.66% | 69.90% | 51.54% | 88.26% |
| hsi_lightness | 34033 / 44885 | 75.82% | 69.54% | 49.49% | 89.59% |
| hsi_intensity | 34212 / 44885 | 76.22% | 69.88% | 49.66% | 90.10% |
| cielab_lightness | 34616 / 44885 | 77.12% | 70.90% | 51.08% | 90.73% |
| rms_contrast | 31249 / 44885 | 69.62% | 68.62% | 65.44% | 71.80% |
| mean_shift_separation_spread_ratio | 19238 / 44885 | 42.86% | 37.92% | 22.18% | 53.67% |


## RMS contrast

RMS rejected **21456 of 29867** transformed failures and accepted **12084 of 17603** successful images (original and transformed). Failure recall: **0.718**; transformed success acceptance: **0.683**; MCC: **0.395**.

Success acceptance is the fraction of images with AP at or above the threshold whose brightness score is inside the operating range. Balanced accuracy uses failure recall and acceptance of all successful images. TP/FP/TN/FN include originals and every transformation.

## Sample images

Representative too-dark, in-range, and out-of-range images for each valid method are in `sample_images/<method>`. Each method directory contains a JSON manifest with the fixed bounds and image metadata. The out-of-range set includes both dark and bright violations and therefore intentionally overlaps the too-dark set.

![Validation plot](brightness_validation.png)

## Confusion matrices

Failure (AP below the threshold) is the positive class. Rows show actual failure/success; columns show predicted failure (outside range)/success (inside range). Cells show counts and percentages within each actual class; colour uses the same 0–100% scale for every method. Overall includes eligible originals and all transformations; directional plots exclude originals.

![Overall confusion matrices](confusion_matrices_overall.png)

![Brightened-only confusion matrices](confusion_matrices_brightened.png)

![Darkened-only confusion matrices](confusion_matrices_darkened.png)
