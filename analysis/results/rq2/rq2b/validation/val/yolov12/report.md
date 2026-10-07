# RQ2b Brightness Validation: val / yolov12

## Design

Only original validation images with per-image COCO AP@[0.50:0.95] ≥ 0.50 were eligible. For each eligible image, the full RQ2a brute-force gamma grid was evaluated. All transformations were retained, including AP at or above the threshold; no interpolation or AP-based selection was used.

The operating ranges were fixed before validation using the unadjusted RQ2b boundary means: lower = dark/right mean and upper = bright/left mean. Eligible originals are successful controls.

Images checked: **363**; eligible originals: **237**; transformations: **47637**; transformed successes: **16579**; transformed failures: **31058**.

## Fixed mean-boundary ranges

| Method | Dark mean | Dark SD | Lower | Bright mean | Bright SD | Upper |
|---|---:|---:|---:|---:|---:|---:|
| luminance_rec601 | 0.085995 | 0.057151 | 0.085995 | 0.59288 | 0.069892 | 0.59288 |
| luminance_rec709 | 0.087065 | 0.05838 | 0.087065 | 0.59334 | 0.06988 | 0.59334 |
| hsv_value | 0.11146 | 0.066445 | 0.11146 | 0.6011 | 0.067566 | 0.6011 |
| hsi_lightness | 0.08697 | 0.05591 | 0.08697 | 0.59176 | 0.069801 | 0.59176 |
| hsi_intensity | 0.086373 | 0.056688 | 0.086373 | 0.59213 | 0.069825 | 0.59213 |
| cielab_lightness | 9.0197 | 6.1813 | 9.0197 | 59.857 | 6.701 | 59.857 |
| rms_contrast | 0.17517 | 0.078425 | 0.17517 | 0.43565 | 0.034884 | 0.43565 |
| mean_shift_separation_spread_ratio | 140.21 | 489.04 | 140.21 | 4050.6 | 8029.7 | 4050.6 |

## Validation results

| Rank | Method | Failure recall | Bright recall | Dark recall | Original acceptance | Transformed success acceptance | All success acceptance | Balanced accuracy | MCC | TP | FP | TN | FN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | luminance_rec709 | 0.888 | 0.630 | 0.892 | 1.000 | 0.614 | 0.619 | 0.754 | 0.534 | 27583 | 6399 | 10417 | 3475 |
| 2 | luminance_rec601 | 0.887 | 0.630 | 0.890 | 1.000 | 0.610 | 0.615 | 0.751 | 0.528 | 27535 | 6474 | 10342 | 3523 |
| 3 | cielab_lightness | 0.886 | 0.630 | 0.890 | 1.000 | 0.608 | 0.614 | 0.750 | 0.526 | 27522 | 6493 | 10323 | 3536 |
| 4 | hsi_intensity | 0.879 | 0.628 | 0.882 | 1.000 | 0.597 | 0.603 | 0.741 | 0.506 | 27294 | 6680 | 10136 | 3764 |
| 5 | hsi_lightness | 0.872 | 0.626 | 0.875 | 1.000 | 0.594 | 0.599 | 0.736 | 0.494 | 27071 | 6736 | 10080 | 3987 |
| 6 | hsv_value | 0.858 | 0.633 | 0.861 | 0.996 | 0.608 | 0.613 | 0.736 | 0.487 | 26641 | 6501 | 10315 | 4417 |
| 7 | rms_contrast | 0.696 | 0.716 | 0.696 | 0.924 | 0.709 | 0.712 | 0.704 | 0.392 | 21624 | 4846 | 11970 | 9434 |
| 8 | mean_shift_separation_spread_ratio | 0.544 | 0.652 | 0.542 | 0.042 | 0.243 | 0.240 | 0.392 | -0.212 | 16884 | 12774 | 4042 | 14174 |

## Classification by transformation direction

Accuracy is the fraction correctly classified: AP below the threshold and outside the operating range, or AP at/above the threshold and inside it. Each table uses only its stated direction; original images are excluded. Balanced accuracy averages successful-image acceptance and failure rejection.

### Brightened images (gamma < 1)

Images: **2370**; successes: **1948**; failures: **422** (AP threshold: 0.50).

| Method | Correct / total | Accuracy | Balanced accuracy | Successful images accepted | Failures rejected |
|---|---:|---:|---:|---:|---:|
| luminance_rec601 | 2078 / 2370 | 87.68% | 78.03% | 93.02% | 63.03% |
| luminance_rec709 | 2078 / 2370 | 87.68% | 78.03% | 93.02% | 63.03% |
| hsv_value | 2074 / 2370 | 87.51% | 78.02% | 92.76% | 63.27% |
| hsi_lightness | 2078 / 2370 | 87.68% | 77.84% | 93.12% | 62.56% |
| hsi_intensity | 2079 / 2370 | 87.72% | 77.96% | 93.12% | 62.80% |
| cielab_lightness | 2069 / 2370 | 87.30% | 77.79% | 92.56% | 63.03% |
| rms_contrast | 2093 / 2370 | 88.31% | 81.75% | 91.94% | 71.56% |
| mean_shift_separation_spread_ratio | 1163 / 2370 | 49.07% | 55.38% | 45.59% | 65.17% |

### Darkened images (gamma > 1)

Images: **45267**; successes: **14631**; failures: **30636** (AP threshold: 0.50).

| Method | Correct / total | Accuracy | Balanced accuracy | Successful images accepted | Failures rejected |
|---|---:|---:|---:|---:|---:|
| luminance_rec601 | 35562 / 45267 | 78.56% | 72.85% | 56.68% | 89.01% |
| luminance_rec709 | 35685 / 45267 | 78.83% | 73.18% | 57.19% | 89.17% |
| hsv_value | 34646 / 45267 | 76.54% | 71.31% | 56.54% | 86.09% |
| hsi_lightness | 34836 / 45267 | 76.96% | 71.19% | 54.88% | 87.50% |
| hsi_intensity | 35114 / 45267 | 77.57% | 71.74% | 55.26% | 88.23% |
| cielab_lightness | 35539 / 45267 | 78.51% | 72.79% | 56.61% | 88.97% |
| rms_contrast | 31282 / 45267 | 69.11% | 68.84% | 68.07% | 69.60% |
| mean_shift_separation_spread_ratio | 19753 / 45267 | 43.64% | 37.85% | 21.49% | 54.21% |


## RMS contrast

RMS rejected **21624 of 31058** transformed failures and accepted **11970 of 16816** successful images (original and transformed). Failure recall: **0.696**; transformed success acceptance: **0.709**; MCC: **0.392**.

Success acceptance is the fraction of images with AP at or above the threshold whose brightness score is inside the operating range. Balanced accuracy uses failure recall and acceptance of all successful images. TP/FP/TN/FN include originals and every transformation.

## Sample images

Representative too-dark, in-range, and out-of-range images for each valid method are in `sample_images/<method>`. Each method directory contains a JSON manifest with the fixed bounds and image metadata. The out-of-range set includes both dark and bright violations and therefore intentionally overlaps the too-dark set.

![Validation plot](brightness_validation.png)

## Confusion matrices

Failure (AP below the threshold) is the positive class. Rows show actual failure/success; columns show predicted failure (outside range)/success (inside range). Cells show counts and percentages within each actual class; colour uses the same 0–100% scale for every method. Overall includes eligible originals and all transformations; directional plots exclude originals.

![Overall confusion matrices](confusion_matrices_overall.png)

![Brightened-only confusion matrices](confusion_matrices_brightened.png)

![Darkened-only confusion matrices](confusion_matrices_darkened.png)
