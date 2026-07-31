# Big Orange Cone Confidence Across imgDarkBright Settings

Generated on 2026-07-17 from `/moredata/emma/metamorphic_v2/YOLO_metamorphic_testing/results/test`.

## Scope and interpretation

This analysis selects Big Orange Cone predictions using zero-indexed class ID `2` from exactly `imgDarkBright_0` through `imgDarkBright_9` for YOLOv5, YOLOv8, YOLOv12, and YOLOv26. Each valid detection row has the format `class_id confidence x1 y1 x2 y2`; its saved confidence is one observation. JSON exports are not read.

The setting-to-gamma mapping is 0.5, 1.0, ..., 5.0. The transform computes output intensity proportional to `input_intensity ** gamma`: gamma 0.5 brightens, gamma 1.0 is intensity-neutral, and increasing gamma above 1.0 progressively darkens. These confidence statistics do **not** use ground-truth matching, so true and false positive class-2 detections are pooled. They describe saved, post-threshold detections and should not be interpreted as accuracy or calibration measurements.

Variance and standard deviation are population measures (`ddof=0`). The main tables are detection-weighted, meaning an image with multiple class-2 detections contributes multiple observations.

## Pooled results across all ten settings

| Model | Detections | Mean | Median | Variance | Std. dev. | Min | Max | >= 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | 2,719 | 0.828406 | 0.883897 | 0.024845 | 0.157622 | 0.250079 | 0.972049 | 41.45% |
| YOLOv8 | 2,669 | 0.846615 | 0.896171 | 0.022560 | 0.150200 | 0.253413 | 0.970249 | 47.43% |
| YOLOv12 | 2,647 | 0.835128 | 0.889017 | 0.023508 | 0.153323 | 0.251250 | 0.965712 | 42.88% |
| YOLOv26 | 3,115 | 0.778054 | 0.882973 | 0.046686 | 0.216069 | 0.250202 | 0.982757 | 44.56% |

Pooling is useful as a compact summary, but settings with more saved detections receive more weight. The per-setting results below are the primary comparison for brightness sensitivity.

## Results by model and setting

| Model | Setting | Gamma | Detections | Files with class / scanned | Mean | Median | Variance | Std. dev. | Mean of file means |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | imgDarkBright_0 | 0.5 | 273 | 92 / 363 | 0.841695 | 0.886953 | 0.020078 | 0.141697 | 0.815692 |
| YOLOv5 | imgDarkBright_1 | 1.0 | 276 | 89 / 363 | 0.847387 | 0.888898 | 0.017383 | 0.131846 | 0.833910 |
| YOLOv5 | imgDarkBright_2 | 1.5 | 276 | 87 / 363 | 0.846017 | 0.891495 | 0.019158 | 0.138411 | 0.842590 |
| YOLOv5 | imgDarkBright_3 | 2.0 | 275 | 87 / 363 | 0.839964 | 0.892595 | 0.023128 | 0.152080 | 0.831287 |
| YOLOv5 | imgDarkBright_4 | 2.5 | 271 | 88 / 363 | 0.839651 | 0.892252 | 0.022392 | 0.149638 | 0.827495 |
| YOLOv5 | imgDarkBright_5 | 3.0 | 265 | 90 / 363 | 0.842184 | 0.890197 | 0.020239 | 0.142264 | 0.822017 |
| YOLOv5 | imgDarkBright_6 | 3.5 | 264 | 90 / 363 | 0.833848 | 0.886051 | 0.022827 | 0.151088 | 0.814985 |
| YOLOv5 | imgDarkBright_7 | 4.0 | 276 | 97 / 363 | 0.805457 | 0.875992 | 0.031879 | 0.178546 | 0.776157 |
| YOLOv5 | imgDarkBright_8 | 4.5 | 271 | 93 / 363 | 0.802392 | 0.867498 | 0.031832 | 0.178415 | 0.776510 |
| YOLOv5 | imgDarkBright_9 | 5.0 | 272 | 92 / 363 | 0.785550 | 0.857379 | 0.035119 | 0.187401 | 0.767098 |
| YOLOv8 | imgDarkBright_0 | 0.5 | 273 | 91 / 363 | 0.861641 | 0.899261 | 0.015955 | 0.126312 | 0.845542 |
| YOLOv8 | imgDarkBright_1 | 1.0 | 277 | 91 / 363 | 0.862641 | 0.900647 | 0.015895 | 0.126076 | 0.846623 |
| YOLOv8 | imgDarkBright_2 | 1.5 | 277 | 91 / 363 | 0.860201 | 0.899879 | 0.017860 | 0.133640 | 0.842681 |
| YOLOv8 | imgDarkBright_3 | 2.0 | 271 | 89 / 363 | 0.865344 | 0.900467 | 0.015786 | 0.125642 | 0.849030 |
| YOLOv8 | imgDarkBright_4 | 2.5 | 271 | 91 / 363 | 0.857734 | 0.899689 | 0.018692 | 0.136719 | 0.835413 |
| YOLOv8 | imgDarkBright_5 | 3.0 | 271 | 92 / 363 | 0.845366 | 0.895978 | 0.023144 | 0.152130 | 0.818625 |
| YOLOv8 | imgDarkBright_6 | 3.5 | 262 | 89 / 363 | 0.841044 | 0.894694 | 0.024895 | 0.157783 | 0.818181 |
| YOLOv8 | imgDarkBright_7 | 4.0 | 257 | 89 / 363 | 0.833667 | 0.891956 | 0.028401 | 0.168527 | 0.807315 |
| YOLOv8 | imgDarkBright_8 | 4.5 | 258 | 88 / 363 | 0.817360 | 0.886606 | 0.032013 | 0.178922 | 0.795934 |
| YOLOv8 | imgDarkBright_9 | 5.0 | 252 | 85 / 363 | 0.815979 | 0.884262 | 0.031702 | 0.178052 | 0.796422 |
| YOLOv12 | imgDarkBright_0 | 0.5 | 279 | 93 / 363 | 0.843678 | 0.892722 | 0.019773 | 0.140618 | 0.826654 |
| YOLOv12 | imgDarkBright_1 | 1.0 | 274 | 92 / 363 | 0.858060 | 0.897166 | 0.014995 | 0.122456 | 0.841166 |
| YOLOv12 | imgDarkBright_2 | 1.5 | 272 | 91 / 363 | 0.858285 | 0.898358 | 0.015694 | 0.125276 | 0.842789 |
| YOLOv12 | imgDarkBright_3 | 2.0 | 268 | 90 / 363 | 0.858335 | 0.898425 | 0.016036 | 0.126634 | 0.842772 |
| YOLOv12 | imgDarkBright_4 | 2.5 | 266 | 91 / 363 | 0.850129 | 0.895631 | 0.019506 | 0.139665 | 0.824606 |
| YOLOv12 | imgDarkBright_5 | 3.0 | 266 | 92 / 363 | 0.837182 | 0.891407 | 0.023770 | 0.154175 | 0.813334 |
| YOLOv12 | imgDarkBright_6 | 3.5 | 262 | 92 / 363 | 0.826979 | 0.887474 | 0.024881 | 0.157737 | 0.804000 |
| YOLOv12 | imgDarkBright_7 | 4.0 | 263 | 92 / 363 | 0.808098 | 0.880635 | 0.032505 | 0.180292 | 0.790550 |
| YOLOv12 | imgDarkBright_8 | 4.5 | 251 | 91 / 363 | 0.811983 | 0.874213 | 0.027469 | 0.165738 | 0.800719 |
| YOLOv12 | imgDarkBright_9 | 5.0 | 246 | 93 / 363 | 0.791757 | 0.862051 | 0.037549 | 0.193775 | 0.763628 |
| YOLOv26 | imgDarkBright_0 | 0.5 | 300 | 91 / 363 | 0.806536 | 0.896051 | 0.038773 | 0.196909 | 0.805462 |
| YOLOv26 | imgDarkBright_1 | 1.0 | 301 | 91 / 363 | 0.811269 | 0.901641 | 0.037622 | 0.193965 | 0.806815 |
| YOLOv26 | imgDarkBright_2 | 1.5 | 299 | 91 / 363 | 0.809037 | 0.898204 | 0.038789 | 0.196950 | 0.804298 |
| YOLOv26 | imgDarkBright_3 | 2.0 | 297 | 94 / 363 | 0.804275 | 0.899176 | 0.040876 | 0.202179 | 0.783201 |
| YOLOv26 | imgDarkBright_4 | 2.5 | 301 | 94 / 363 | 0.789716 | 0.893000 | 0.046278 | 0.215123 | 0.768892 |
| YOLOv26 | imgDarkBright_5 | 3.0 | 304 | 96 / 363 | 0.783475 | 0.890762 | 0.047565 | 0.218093 | 0.759843 |
| YOLOv26 | imgDarkBright_6 | 3.5 | 316 | 96 / 363 | 0.763005 | 0.877447 | 0.051407 | 0.226732 | 0.745257 |
| YOLOv26 | imgDarkBright_7 | 4.0 | 329 | 98 / 363 | 0.747289 | 0.862438 | 0.053575 | 0.231462 | 0.730567 |
| YOLOv26 | imgDarkBright_8 | 4.5 | 335 | 99 / 363 | 0.744469 | 0.858922 | 0.050127 | 0.223891 | 0.737391 |
| YOLOv26 | imgDarkBright_9 | 5.0 | 333 | 98 / 363 | 0.734135 | 0.841336 | 0.051529 | 0.227001 | 0.730863 |

`Mean of file means` gives each class-2-positive file equal weight; files without a saved class-2 prediction are excluded from that mean.

## Distribution detail by setting

| Model | Setting (gamma) | Min | P05 | P25 | P75 | P95 | Max | IQR | >= 0.50 | >= 0.75 | >= 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | 0 (0.5) | 0.264839 | 0.513904 | 0.828785 | 0.925845 | 0.955820 | 0.967516 | 0.097060 | 95.24% | 83.88% | 41.76% |
| YOLOv5 | 1 (1.0) | 0.254707 | 0.522964 | 0.835744 | 0.930523 | 0.958676 | 0.970385 | 0.094778 | 96.74% | 83.70% | 44.20% |
| YOLOv5 | 2 (1.5) | 0.251868 | 0.546048 | 0.832691 | 0.931559 | 0.959241 | 0.971445 | 0.098868 | 95.65% | 84.06% | 45.65% |
| YOLOv5 | 3 (2.0) | 0.262864 | 0.467546 | 0.834145 | 0.931647 | 0.958551 | 0.971397 | 0.097503 | 93.82% | 82.91% | 45.45% |
| YOLOv5 | 4 (2.5) | 0.273932 | 0.485131 | 0.828458 | 0.931881 | 0.958755 | 0.970734 | 0.103423 | 94.46% | 83.76% | 45.02% |
| YOLOv5 | 5 (3.0) | 0.250079 | 0.555096 | 0.821829 | 0.930187 | 0.959071 | 0.969655 | 0.108358 | 95.85% | 83.40% | 44.53% |
| YOLOv5 | 6 (3.5) | 0.254537 | 0.468827 | 0.819429 | 0.930159 | 0.958870 | 0.970772 | 0.110730 | 93.94% | 82.95% | 43.56% |
| YOLOv5 | 7 (4.0) | 0.261338 | 0.376942 | 0.770122 | 0.923521 | 0.958247 | 0.971654 | 0.153400 | 89.86% | 78.62% | 38.41% |
| YOLOv5 | 8 (4.5) | 0.255310 | 0.384220 | 0.768666 | 0.920839 | 0.957440 | 0.971878 | 0.152173 | 88.93% | 77.12% | 33.95% |
| YOLOv5 | 9 (5.0) | 0.252065 | 0.320064 | 0.721173 | 0.916172 | 0.956390 | 0.972049 | 0.194999 | 87.87% | 71.32% | 31.99% |
| YOLOv8 | 0 (0.5) | 0.257039 | 0.624336 | 0.856425 | 0.934008 | 0.957843 | 0.967385 | 0.077583 | 96.70% | 86.81% | 48.72% |
| YOLOv8 | 1 (1.0) | 0.261898 | 0.616869 | 0.853377 | 0.935501 | 0.959057 | 0.966255 | 0.082124 | 96.39% | 87.36% | 50.90% |
| YOLOv8 | 2 (1.5) | 0.261771 | 0.551515 | 0.854294 | 0.937807 | 0.959641 | 0.966083 | 0.083513 | 95.67% | 88.45% | 49.10% |
| YOLOv8 | 3 (2.0) | 0.253413 | 0.581570 | 0.855013 | 0.939532 | 0.959277 | 0.968025 | 0.084519 | 96.68% | 89.30% | 51.29% |
| YOLOv8 | 4 (2.5) | 0.259004 | 0.540323 | 0.851764 | 0.939307 | 0.959384 | 0.969510 | 0.087543 | 95.57% | 87.82% | 49.45% |
| YOLOv8 | 5 (3.0) | 0.287709 | 0.440812 | 0.844332 | 0.937475 | 0.959022 | 0.970249 | 0.093144 | 93.36% | 85.24% | 47.97% |
| YOLOv8 | 6 (3.5) | 0.261781 | 0.405775 | 0.838572 | 0.935541 | 0.958192 | 0.970151 | 0.096969 | 93.13% | 83.21% | 47.33% |
| YOLOv8 | 7 (4.0) | 0.276660 | 0.350484 | 0.835676 | 0.934388 | 0.957768 | 0.968430 | 0.098712 | 92.61% | 82.88% | 44.75% |
| YOLOv8 | 8 (4.5) | 0.258509 | 0.375330 | 0.806764 | 0.933063 | 0.957482 | 0.965291 | 0.126299 | 91.09% | 80.23% | 42.64% |
| YOLOv8 | 9 (5.0) | 0.260838 | 0.361825 | 0.801368 | 0.930184 | 0.957270 | 0.964398 | 0.128816 | 90.08% | 79.37% | 41.27% |
| YOLOv12 | 0 (0.5) | 0.256121 | 0.481068 | 0.832456 | 0.927415 | 0.952950 | 0.960333 | 0.094959 | 94.62% | 86.02% | 43.73% |
| YOLOv12 | 1 (1.0) | 0.266322 | 0.557582 | 0.846521 | 0.930748 | 0.953355 | 0.961739 | 0.084227 | 96.72% | 86.86% | 47.08% |
| YOLOv12 | 2 (1.5) | 0.289275 | 0.561296 | 0.853239 | 0.931173 | 0.953629 | 0.962226 | 0.077934 | 97.06% | 87.13% | 48.16% |
| YOLOv12 | 3 (2.0) | 0.262693 | 0.555542 | 0.850933 | 0.931840 | 0.954808 | 0.962474 | 0.080908 | 97.01% | 88.06% | 48.51% |
| YOLOv12 | 4 (2.5) | 0.253839 | 0.536116 | 0.844700 | 0.931355 | 0.954381 | 0.962450 | 0.086655 | 95.49% | 85.71% | 46.62% |
| YOLOv12 | 5 (3.0) | 0.255711 | 0.418453 | 0.827796 | 0.930660 | 0.953154 | 0.963215 | 0.102863 | 93.23% | 82.33% | 42.86% |
| YOLOv12 | 6 (3.5) | 0.285274 | 0.410175 | 0.798870 | 0.928941 | 0.952827 | 0.965141 | 0.130071 | 92.37% | 79.77% | 42.37% |
| YOLOv12 | 7 (4.0) | 0.251760 | 0.347012 | 0.785384 | 0.927516 | 0.952276 | 0.965712 | 0.142132 | 89.73% | 77.95% | 37.64% |
| YOLOv12 | 8 (4.5) | 0.259441 | 0.436746 | 0.773463 | 0.926744 | 0.952558 | 0.965632 | 0.153281 | 92.43% | 76.49% | 37.05% |
| YOLOv12 | 9 (5.0) | 0.251250 | 0.321130 | 0.763820 | 0.924374 | 0.951516 | 0.964071 | 0.160554 | 87.80% | 76.42% | 33.33% |
| YOLOv26 | 0 (0.5) | 0.251992 | 0.334013 | 0.767475 | 0.939571 | 0.969013 | 0.981379 | 0.172096 | 87.33% | 76.00% | 48.33% |
| YOLOv26 | 1 (1.0) | 0.251474 | 0.348271 | 0.754602 | 0.943244 | 0.968196 | 0.982170 | 0.188642 | 88.04% | 75.75% | 50.17% |
| YOLOv26 | 2 (1.5) | 0.251244 | 0.335334 | 0.761954 | 0.941767 | 0.967376 | 0.982562 | 0.179813 | 86.96% | 75.92% | 49.83% |
| YOLOv26 | 3 (2.0) | 0.256680 | 0.330691 | 0.736953 | 0.942484 | 0.968531 | 0.982757 | 0.205531 | 86.20% | 74.41% | 49.83% |
| YOLOv26 | 4 (2.5) | 0.261155 | 0.314940 | 0.714474 | 0.937083 | 0.969050 | 0.982718 | 0.222609 | 82.72% | 71.76% | 48.17% |
| YOLOv26 | 5 (3.0) | 0.253142 | 0.315204 | 0.688212 | 0.939743 | 0.969155 | 0.982633 | 0.251531 | 83.55% | 69.74% | 47.37% |
| YOLOv26 | 6 (3.5) | 0.254177 | 0.302970 | 0.605205 | 0.935701 | 0.968921 | 0.982453 | 0.330496 | 81.01% | 66.46% | 42.72% |
| YOLOv26 | 7 (4.0) | 0.250843 | 0.296580 | 0.571837 | 0.932346 | 0.968839 | 0.981608 | 0.360509 | 78.42% | 63.53% | 39.51% |
| YOLOv26 | 8 (4.5) | 0.253989 | 0.319028 | 0.555438 | 0.929073 | 0.967709 | 0.980874 | 0.373636 | 80.00% | 61.19% | 37.91% |
| YOLOv26 | 9 (5.0) | 0.250202 | 0.308973 | 0.537955 | 0.923355 | 0.967968 | 0.980849 | 0.385400 | 78.98% | 59.76% | 34.23% |

## Brightness trends and model comparisons

| Model | Highest-mean setting (gamma) | Highest mean | Lowest-mean setting (gamma) | Lowest mean | Mean change 0->9 | Detection-count change 0->9 | Mean slope / gamma | Pearson r |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | 1 (1.0) | 0.847387 | 9 (5.0) | 0.785550 | -0.056145 | -1 | -0.012592 | -0.865452 |
| YOLOv8 | 3 (2.0) | 0.865344 | 9 (5.0) | 0.815979 | -0.045662 | -21 | -0.011465 | -0.930144 |
| YOLOv12 | 3 (2.0) | 0.858335 | 9 (5.0) | 0.791757 | -0.051922 | -33 | -0.013913 | -0.883687 |
| YOLOv26 | 1 (1.0) | 0.811269 | 9 (5.0) | 0.734135 | -0.072400 | +33 | -0.018885 | -0.962432 |

The slope and Pearson correlation summarize the ten per-setting means. They describe association with gamma, not necessarily a linear or causal response; the highest/lowest rows retain non-monotonic behavior that a slope can hide.

### Highest mean confidence at each setting

| Setting | Gamma | Highest-mean model | Mean | All model means |
| --- | --- | --- | --- | --- |
| imgDarkBright_0 | 0.5 | YOLOv8 | 0.861641 | YOLOv5 0.841695 / YOLOv8 0.861641 / YOLOv12 0.843678 / YOLOv26 0.806536 |
| imgDarkBright_1 | 1.0 | YOLOv8 | 0.862641 | YOLOv5 0.847387 / YOLOv8 0.862641 / YOLOv12 0.858060 / YOLOv26 0.811269 |
| imgDarkBright_2 | 1.5 | YOLOv8 | 0.860201 | YOLOv5 0.846017 / YOLOv8 0.860201 / YOLOv12 0.858285 / YOLOv26 0.809037 |
| imgDarkBright_3 | 2.0 | YOLOv8 | 0.865344 | YOLOv5 0.839964 / YOLOv8 0.865344 / YOLOv12 0.858335 / YOLOv26 0.804275 |
| imgDarkBright_4 | 2.5 | YOLOv8 | 0.857734 | YOLOv5 0.839651 / YOLOv8 0.857734 / YOLOv12 0.850129 / YOLOv26 0.789716 |
| imgDarkBright_5 | 3.0 | YOLOv8 | 0.845366 | YOLOv5 0.842184 / YOLOv8 0.845366 / YOLOv12 0.837182 / YOLOv26 0.783475 |
| imgDarkBright_6 | 3.5 | YOLOv8 | 0.841044 | YOLOv5 0.833848 / YOLOv8 0.841044 / YOLOv12 0.826979 / YOLOv26 0.763005 |
| imgDarkBright_7 | 4.0 | YOLOv8 | 0.833667 | YOLOv5 0.805457 / YOLOv8 0.833667 / YOLOv12 0.808098 / YOLOv26 0.747289 |
| imgDarkBright_8 | 4.5 | YOLOv8 | 0.817360 | YOLOv5 0.802392 / YOLOv8 0.817360 / YOLOv12 0.811983 / YOLOv26 0.744469 |
| imgDarkBright_9 | 5.0 | YOLOv8 | 0.815979 | YOLOv5 0.785550 / YOLOv8 0.815979 / YOLOv12 0.791757 / YOLOv26 0.734135 |

## Key findings

- **YOLOv8** has the highest pooled mean confidence (0.846615).
- All four models have lower mean confidence at gamma 5.0 than at gamma 0.5. The largest endpoint decrease is **YOLOv26** (-0.072400).
- **YOLOv5** has the highest setting mean in 0 of 10 settings; **YOLOv8** has the highest setting mean in 10 of 10 settings; **YOLOv12** has the highest setting mean in 0 of 10 settings; **YOLOv26** has the highest setting mean in 0 of 10 settings.

## Parsing diagnostics

| Model | Setting (gamma) | Files | Empty files | Files with any detection | Files with class 2 | Valid rows (all classes) | Malformed rows | Out-of-range confidence rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | 0 (0.5) | 363 | 0 | 363 | 92 | 3,829 | 0 | 0 |
| YOLOv5 | 1 (1.0) | 363 | 0 | 363 | 89 | 3,896 | 0 | 0 |
| YOLOv5 | 2 (1.5) | 363 | 0 | 363 | 87 | 3,889 | 0 | 0 |
| YOLOv5 | 3 (2.0) | 363 | 0 | 363 | 87 | 3,850 | 0 | 0 |
| YOLOv5 | 4 (2.5) | 363 | 0 | 363 | 88 | 3,791 | 0 | 0 |
| YOLOv5 | 5 (3.0) | 363 | 0 | 363 | 90 | 3,699 | 0 | 0 |
| YOLOv5 | 6 (3.5) | 363 | 2 | 361 | 90 | 3,541 | 0 | 0 |
| YOLOv5 | 7 (4.0) | 363 | 8 | 355 | 97 | 3,358 | 0 | 0 |
| YOLOv5 | 8 (4.5) | 363 | 14 | 349 | 93 | 3,185 | 0 | 0 |
| YOLOv5 | 9 (5.0) | 363 | 19 | 344 | 92 | 3,018 | 0 | 0 |
| YOLOv8 | 0 (0.5) | 363 | 0 | 363 | 91 | 3,877 | 0 | 0 |
| YOLOv8 | 1 (1.0) | 363 | 0 | 363 | 91 | 3,969 | 0 | 0 |
| YOLOv8 | 2 (1.5) | 363 | 0 | 363 | 91 | 3,962 | 0 | 0 |
| YOLOv8 | 3 (2.0) | 363 | 0 | 363 | 89 | 3,920 | 0 | 0 |
| YOLOv8 | 4 (2.5) | 363 | 0 | 363 | 91 | 3,842 | 0 | 0 |
| YOLOv8 | 5 (3.0) | 363 | 0 | 363 | 92 | 3,704 | 0 | 0 |
| YOLOv8 | 6 (3.5) | 363 | 2 | 361 | 89 | 3,500 | 0 | 0 |
| YOLOv8 | 7 (4.0) | 363 | 14 | 349 | 89 | 3,266 | 0 | 0 |
| YOLOv8 | 8 (4.5) | 363 | 25 | 338 | 88 | 3,076 | 0 | 0 |
| YOLOv8 | 9 (5.0) | 363 | 28 | 335 | 85 | 2,885 | 0 | 0 |
| YOLOv12 | 0 (0.5) | 363 | 0 | 363 | 93 | 3,776 | 0 | 0 |
| YOLOv12 | 1 (1.0) | 363 | 0 | 363 | 92 | 3,878 | 0 | 0 |
| YOLOv12 | 2 (1.5) | 363 | 0 | 363 | 91 | 3,918 | 0 | 0 |
| YOLOv12 | 3 (2.0) | 363 | 0 | 363 | 90 | 3,894 | 0 | 0 |
| YOLOv12 | 4 (2.5) | 363 | 0 | 363 | 91 | 3,849 | 0 | 0 |
| YOLOv12 | 5 (3.0) | 363 | 2 | 361 | 92 | 3,713 | 0 | 0 |
| YOLOv12 | 6 (3.5) | 363 | 4 | 359 | 92 | 3,494 | 0 | 0 |
| YOLOv12 | 7 (4.0) | 363 | 16 | 347 | 92 | 3,292 | 0 | 0 |
| YOLOv12 | 8 (4.5) | 363 | 26 | 337 | 91 | 3,120 | 0 | 0 |
| YOLOv12 | 9 (5.0) | 363 | 29 | 334 | 93 | 2,905 | 0 | 0 |
| YOLOv26 | 0 (0.5) | 363 | 0 | 363 | 91 | 6,068 | 0 | 0 |
| YOLOv26 | 1 (1.0) | 363 | 0 | 363 | 91 | 6,152 | 0 | 0 |
| YOLOv26 | 2 (1.5) | 363 | 0 | 363 | 91 | 6,137 | 0 | 0 |
| YOLOv26 | 3 (2.0) | 363 | 0 | 363 | 94 | 6,078 | 0 | 0 |
| YOLOv26 | 4 (2.5) | 363 | 0 | 363 | 94 | 6,009 | 0 | 0 |
| YOLOv26 | 5 (3.0) | 363 | 0 | 363 | 96 | 5,852 | 0 | 0 |
| YOLOv26 | 6 (3.5) | 363 | 4 | 359 | 96 | 5,607 | 0 | 0 |
| YOLOv26 | 7 (4.0) | 363 | 9 | 354 | 98 | 5,321 | 0 | 0 |
| YOLOv26 | 8 (4.5) | 363 | 21 | 342 | 99 | 5,033 | 0 | 0 |
| YOLOv26 | 9 (5.0) | 363 | 25 | 338 | 98 | 4,762 | 0 | 0 |

An empty file is a valid inference output containing no saved detections. A malformed row has the wrong field count, a non-numeric value, or a non-finite value. Rows with confidence outside [0, 1] are counted separately and excluded.

## Reproduction

```bash
python3 analysis/analyse_big_orange_cone_darkbright_confidence.py
```

Use `--results-dir`, `--output`, `--models`, or `--class-id` to override the defaults. All ten required model setting directories must exist.
