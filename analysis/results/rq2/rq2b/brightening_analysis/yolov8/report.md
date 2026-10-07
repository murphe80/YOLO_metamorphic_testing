# yolov8: performance on gamma-brightened images

730 transformations from 73 originals, with 498 successes (AP ≥ 0.5) and 232 failures. Fixed calibration mean boundaries; no fitting or threshold selection.

| Method | Accuracy | Balanced accuracy | Successful images accepted | Failures rejected | MCC |
|---|---:|---:|---:|---:|---:|
| rms_contrast | 82.19% | 84.53% | 78.11% | 90.95% | 0.648 |
| luminance_rec709 | 71.51% | 77.85% | 60.44% | 95.26% | 0.524 |
| luminance_rec601 | 71.37% | 77.75% | 60.24% | 95.26% | 0.523 |
| hsi_intensity | 70.96% | 77.45% | 59.64% | 95.26% | 0.518 |
| hsi_lightness | 70.68% | 77.25% | 59.24% | 95.26% | 0.514 |
| cielab_lightness | 65.75% | 74.32% | 50.80% | 97.84% | 0.474 |
| hsv_value | 64.79% | 73.62% | 49.40% | 97.84% | 0.463 |
| mean_shift_separation_spread_ratio | 31.64% | 49.78% | 0.00% | 99.57% | -0.054 |

## RMS by transformation direction

| Direction | N | Accuracy | Balanced accuracy | Success acceptance | Failure rejection |
|---|---:|---:|---:|---:|---:|
| original | 73 | 100.00% | N/A | 100.00% | N/A |
| bright | 730 | 82.19% | 84.53% | 78.11% | 90.95% |
| dark | 13943 | 92.61% | 92.68% | 92.75% | 92.60% |

## Brightening by gamma

| Gamma | Successes | Failures | RMS range | Accepted successes | Rejected successes | Rejected failures | Accepted failures |
|---|---:|---:|---|---:|---:|---:|---:|
| 0.01 | 0 | 73 | 0.0070–0.0274 | 0 | 0 | 73 | 0 |
| 0.11 | 7 | 66 | 0.0650–0.0798 | 0 | 7 | 66 | 0 |
| 0.21 | 31 | 42 | 0.1076–0.1263 | 0 | 31 | 42 | 0 |
| 0.31 | 49 | 24 | 0.1395–0.1604 | 0 | 49 | 24 | 0 |
| 0.41 | 59 | 14 | 0.1628–0.1871 | 37 | 22 | 6 | 8 |
| 0.51 | 65 | 8 | 0.1798–0.2101 | 65 | 0 | 0 | 8 |
| 0.61 | 71 | 2 | 0.1920–0.2281 | 71 | 0 | 0 | 2 |
| 0.71 | 72 | 1 | 0.2005–0.2424 | 72 | 0 | 0 | 1 |
| 0.81 | 71 | 2 | 0.2061–0.2534 | 71 | 0 | 0 | 2 |
| 0.91 | 73 | 0 | 0.2095–0.2621 | 73 | 0 | 0 | 0 |

## Interpretation and uncertainty

RMS fixed operating interval: [0.176956, 0.433849]. Bright-image score locations: {'below': {'n': 320, 'successes': 109}, 'inside': {'n': 410, 'successes': 389}, 'above': {'n': 0, 'successes': 0}}.

Positive means AP ≥ 0.5. Success acceptance is not object-detection recall. Balanced accuracy is the mean of success acceptance and failure rejection. Source images were screened for original AP ≥ 0.5, so results apply to that subset and this gamma grid. Bright and dark distributions differ, making direction comparisons descriptive rather than controlled causal comparisons.

Intervals below use 1,000 paired resamples of contiguous blocks of ten timestamp-ordered originals (all gamma variants stay together). Only eight blocks are available; intervals are exploratory, not adjusted for multiple comparisons, and do not prove cross-dataset generalisation.

| Comparator | RMS accuracy improvement, 95% interval | RMS balanced-accuracy improvement, 95% interval |
|---|---|---|
| luminance_rec601 | [+5.48, +15.62] pp | [+2.58, +11.11] pp |
| luminance_rec709 | [+5.45, +15.25] pp | [+2.58, +10.83] pp |
| hsv_value | [+11.12, +21.92] pp | [+5.87, +15.67] pp |
| hsi_lightness | [+6.00, +16.16] pp | [+3.05, +11.65] pp |
| hsi_intensity | [+5.75, +16.03] pp | [+2.72, +11.38] pp |
| cielab_lightness | [+10.67, +20.55] pp | [+5.37, +14.67] pp |
| mean_shift_separation_spread_ratio | [+44.74, +55.21] pp | [+30.64, +39.19] pp |

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

