# YOLO robustness on FSOCO test vs real competition data

## Executive summary

The real competition data is substantially harder than the held-out FSOCO test data under brightness changes, motion blur, reduced resolution, and horizontal stretch. Averaged over models and non-control settings, competition AP/AR is lower by 0.423/0.434 for brightness, 0.387/0.399 for motion blur, 0.264/0.267 for resolution, and 0.111/0.126 for stretch. Hue-shift performance is also lower, but only by 0.030/0.050. Grayscale is the exception: competition results are slightly higher.

The most defensible model choices from the metamorphic consistency results are:

- **Motion blur (all tested kernel sizes 3–21): YOLOv5.** It wins AP and AR at every setting on both datasets.
- **Very low resolution (`size_ratio` 0.1–0.3): YOLOv5.** This is the most stable cross-dataset low-resolution choice.
- **FSOCO-like horizontal distortion (`width_ratio` 1.4–2.8): YOLOv12.** It dominates the test set, but this does not transfer to competition data.
- **Competition-like moderate horizontal distortion (`width_ratio` 1.6–2.2): YOLOv26.** It has the best combined AP/AR in this range.
- **Competition brightness:** YOLOv8 is best at gamma 1.5–2.0, YOLOv26 at 2.5–3.0, and YOLOv5 at 3.5–5.0. At the darker settings, however, every model is already performing poorly.
- **Large hue shifts:** YOLOv26 is the clearest specialist at 140 degrees on both datasets, and often has the best recall from 60–140 degrees. Most models collapse for several shifts in that range, so this is mitigation rather than robust operation.
- **Grayscale:** there is no transferable winner. YOLOv8 leads on test, while competition AP favors YOLOv12 and competition AR favors YOLOv26_v2.

These recommendations describe **prediction consistency**, not absolute detection accuracy. In `original_vs_transformed`, the original model predictions act as the reference. A model can therefore score highly by reproducing its own errors. The available test-set ground-truth results reverse several consistency rankings: YOLOv26/YOLOv26_v2 are substantially more accurate than YOLOv5 on original test images. Consequently, YOLOv5 is a strong choice for repeatability under blur and resolution changes only after its absolute accuracy has been shown to be sufficient for the target environment.

## Scope and method

Primary inputs:

- `analysis/results/test/parameter_variations/metrics_original_vs_transformed.json`
- `analysis/results/competition/parameter_variations/metrics_original_vs_transformed.json`

The files cover YOLOv5, YOLOv8, YOLOv12, YOLOv26, and YOLOv26_v2. AP is COCO AP averaged over IoU 0.50:0.95; AR is COCO AR at `maxDets=100` (`AR100`). Both datasets contain 363 source images.

Unless explicitly described as a raw overall mean, family averages exclude identity/control settings:

- Hue 0 and 180 degrees are excluded because the implementation wraps hue modulo 180, making both identity transformations.
- Gamma 1.0, resolution ratio 1.0, and stretch ratio 1.0 are excluded.
- Competition resolution ratio 0.5 is additionally excluded as a preprocessing artifact; see [Data-quality finding](#data-quality-finding-competition-resolution-ratio-05).
- All motion-blur settings and the single grayscale setting are retained.

When a single winner is needed despite an AP/AR split, the report uses the unweighted mean `(AP + AR) / 2`. AP and AR are still shown separately so that a precision- or recall-sensitive application can make a different choice.

## Overall comparison

The following are raw means across all 51 settings, including controls. They are useful as a broad summary but should not be interpreted as a balanced operational workload.

| Dataset | Model | Mean AP | Mean AR |
|---|---:|---:|---:|
| Test | YOLOv5 | 0.617 | 0.645 |
| Test | YOLOv8 | 0.615 | 0.643 |
| Test | YOLOv12 | **0.623** | **0.650** |
| Test | YOLOv26 | 0.574 | 0.605 |
| Test | YOLOv26_v2 | 0.580 | 0.609 |
| Competition | YOLOv5 | **0.419** | **0.435** |
| Competition | YOLOv8 | 0.383 | 0.401 |
| Competition | YOLOv12 | 0.409 | 0.425 |
| Competition | YOLOv26 | 0.381 | 0.402 |
| Competition | YOLOv26_v2 | 0.352 | 0.368 |

AR is normally slightly higher than AP, and the rankings are usually similar. The mean AR-minus-AP gap is 0.029 on test and 0.018 on competition. Grayscale and hue shift produce the largest AP/AR gaps, indicating that some detections remain recoverable while their precision, localization, class consistency, or confidence ordering deteriorates.

### Dataset severity by transformation

These values average all five models over non-control settings.

| Transformation | Test AP/AR | Competition AP/AR | Competition − test AP/AR |
|---|---:|---:|---:|
| Hue rotation | 0.199 / 0.258 | 0.169 / 0.208 | -0.030 / -0.050 |
| Gamma brightness | 0.725 / 0.756 | 0.302 / 0.322 | **-0.423 / -0.434** |
| Grayscale | 0.143 / 0.204 | 0.152 / 0.229 | +0.009 / +0.025 |
| Motion blur | 0.756 / 0.777 | 0.369 / 0.378 | **-0.387 / -0.399** |
| Resolution | 0.842 / 0.857 | 0.578 / 0.590 | -0.264 / -0.267 |
| Horizontal stretch | 0.260 / 0.290 | 0.149 / 0.165 | -0.111 / -0.126 |

The shapes of the degradation curves nevertheless transfer well. Across models, test-to-competition Pearson correlations are approximately 0.98–1.00 for hue, 0.94–0.97 for brightness, 0.91–0.96 for blur, 0.86–0.95 for resolution, and 0.97–0.99 for stretch. Thus FSOCO generally predicts **which settings are harder**, but often overestimates the AP/AR that will remain on real competition frames.

## Non-control family averages

Each cell is `AP / AR`. Bold marks the highest AP or AR in that row; when different models lead the two metrics, each respective value is bolded.

### Held-out FSOCO test data

| Transformation | YOLOv5 | YOLOv8 | YOLOv12 | YOLOv26 | YOLOv26_v2 |
|---|---:|---:|---:|---:|---:|
| Hue rotation | 0.196 / 0.258 | 0.192 / 0.248 | 0.198 / 0.258 | **0.209 / 0.271** | 0.198 / 0.255 |
| Gamma brightness | 0.738 / 0.769 | 0.740 / 0.769 | **0.746 / 0.775** | 0.707 / 0.740 | 0.692 / 0.727 |
| Grayscale | 0.121 / 0.170 | **0.167 / 0.235** | 0.166 / 0.227 | 0.134 / 0.199 | 0.127 / 0.189 |
| Motion blur | **0.810 / 0.830** | 0.794 / 0.816 | 0.794 / 0.814 | 0.670 / 0.691 | 0.712 / 0.734 |
| Resolution | **0.870 / 0.883** | 0.863 / 0.876 | 0.864 / 0.877 | 0.798 / 0.815 | 0.816 / 0.833 |
| Horizontal stretch | 0.247 / 0.280 | 0.259 / 0.291 | **0.290 / 0.319** | 0.256 / 0.288 | 0.246 / 0.273 |

### Real competition data

| Transformation | YOLOv5 | YOLOv8 | YOLOv12 | YOLOv26 | YOLOv26_v2 |
|---|---:|---:|---:|---:|---:|
| Hue rotation | **0.189** / 0.219 | 0.151 / 0.187 | 0.182 / 0.215 | 0.173 / **0.232** | 0.150 / 0.186 |
| Gamma brightness | **0.325 / 0.348** | 0.305 / 0.324 | 0.310 / 0.330 | 0.310 / 0.331 | 0.259 / 0.275 |
| Grayscale | 0.130 / 0.184 | 0.164 / 0.237 | **0.196** / 0.251 | 0.118 / 0.194 | 0.154 / **0.279** |
| Motion blur | **0.458 / 0.469** | 0.369 / 0.377 | 0.410 / 0.419 | 0.300 / 0.308 | 0.310 / 0.318 |
| Resolution | **0.628 / 0.637** | 0.553 / 0.568 | 0.620 / 0.628 | 0.587 / 0.604 | 0.504 / 0.511 |
| Horizontal stretch | 0.137 / 0.152 | 0.147 / 0.167 | 0.151 / 0.168 | **0.170 / 0.186** | 0.140 / 0.151 |

## Parameter-level findings and model selection

### Motion blur: a clear YOLOv5 environment

YOLOv5 has the highest AP and AR at **every kernel size from 3 through 21 on both datasets**. This is the strongest and most transferable result in the study.

- At kernel 3, YOLOv5 scores 0.982/0.988 on test and 0.930/0.937 on competition.
- At kernel 21, it falls to 0.591/0.623 on test and 0.153/0.160 on competition, but still leads all other models.
- On competition at kernel 21, the runner-up YOLOv12 is only 0.115/0.116.

**Recommendation:** use YOLOv5 when prediction repeatability under directional motion blur is the main requirement. The advantage becomes more pronounced on competition data, although all models become unreliable under the strongest blur.

### Resolution: YOLOv5 at genuinely low resolution; competition crossovers at higher ratios

For severe downscaling, YOLOv5 is consistently best:

| `size_ratio` | Test YOLOv5 AP/AR | Competition YOLOv5 AP/AR | Cross-dataset winner |
|---:|---:|---:|---|
| 0.1 | 0.471 / 0.497 | 0.134 / 0.139 | YOLOv5 |
| 0.2 | 0.751 / 0.773 | 0.447 / 0.461 | YOLOv5 |
| 0.3 | 0.855 / 0.873 | 0.652 / 0.664 | YOLOv5 |

The datasets diverge at less severe settings:

- Test: YOLOv5 leads the combined score at 0.1–0.8 and 1.0; YOLOv8 is ahead at 0.9 by less than 0.001.
- Competition: YOLOv12 leads at 0.4 (0.698/0.709) and 0.6 (0.781/0.791); YOLOv26 leads at 0.7 (0.791/0.807), 0.8 (0.800/0.816), and 0.9 (0.879/0.895).

**Recommendation:** use YOLOv5 for low-pixel-count imagery at ratios 0.1–0.3. Treat the YOLOv26 competition advantage at 0.7–0.9 as specific to this camera/preprocessing pipeline and validate it with ground truth before deployment.

### Brightness: the real-data winner changes with gamma

The transform computes `output = input^gamma`; gamma below 1 brightens midtones, while gamma above 1 darkens them.

| Environment | Test combined winner | Competition combined winner | Competition winner AP/AR |
|---|---|---|---:|
| Brightened, gamma 0.5 | YOLOv5 | YOLOv5 | 0.782 / 0.810 |
| Mildly darker, gamma 1.5 | YOLOv12 | YOLOv8 | 0.773 / 0.794 |
| Darker, gamma 2.0 | YOLOv12 | YOLOv8 | 0.545 / 0.571 |
| Dark, gamma 2.5 | YOLOv12 | YOLOv26 | 0.367 / 0.394 |
| Dark, gamma 3.0 | YOLOv12 | YOLOv26 | 0.231 / 0.262 |
| Very dark, gamma 3.5–5.0 | YOLOv12 | YOLOv5 | 0.140/0.163 down to 0.040/0.046 |

**Recommendation:** on competition-like images, use YOLOv8 for mild darkening, YOLOv26 for moderate darkening, and YOLOv5 for extreme darkness if a model must still run. Gamma 3.5–5.0 is better treated as outside the safe operating envelope: the winning AP falls from 0.140 to 0.040, so model switching does not restore adequate robustness.

### Horizontal stretch: YOLOv12 on FSOCO, YOLOv26 on competition

This is the clearest domain-dependent reversal.

- Test: YOLOv26_v2 narrowly leads at width ratio 1.2; YOLOv12 leads at every ratio from 1.4 to 2.8.
- Competition: YOLOv26 leads at 1.2 and 1.6–2.4; YOLOv8 leads at 1.4 and 2.6–2.8.
- The meaningful competition recommendation is YOLOv26 at ratios 1.6–2.2. Its AP/AR declines from 0.266/0.287 at 1.6 to 0.070/0.085 at 2.2.
- At ratios 2.6–2.8, YOLOv8 technically wins competition, but its scores are only 0.014/0.025 and 0.010/0.019. This is a failure region, not a useful model-selection advantage.

**Recommendation:** YOLOv12 is the repeatability choice for FSOCO-like stretched images. For the real competition camera/domain, YOLOv26 is preferable for moderate aspect distortion. Avoid treating the extreme-stretch YOLOv8 win as operationally meaningful.

### Hue rotation: similar failure shape, unstable model ranking

Both datasets show a wrapped/U-shaped response: near-identity scores occur at 0 and 180 degrees, performance falls sharply in the middle, and partially recovers around 140–160 degrees. The curve shape transfers almost perfectly, but individual winners often do not.

Notable settings:

- 20 degrees: YOLOv12 leads test at 0.583/0.632; YOLOv5 leads competition at 0.569/0.603.
- 40 degrees: YOLOv26 leads test at 0.198/0.274; YOLOv12 leads competition at 0.118/0.151.
- 60 degrees: YOLOv26 leads both datasets, but competition is effectively collapsed at 0.011/0.014.
- 120 degrees: competition AP is highest for YOLOv8 (0.063), while AR is highest for YOLOv26 (0.138); this is a precision/recall trade-off.
- 140 degrees: YOLOv26 clearly leads both datasets—0.216/0.335 on test and 0.283/0.383 on competition.
- 160 degrees: test is nearly tied; competition AP favors YOLOv5 (0.539) while AR favors YOLOv26 (0.609).

**Recommendation:** YOLOv26 is the best candidate where large hue shifts around 140 degrees are expected or recall is more important. YOLOv5 is preferable for competition-like mild shifts around 20 degrees and for AP at 160 degrees. No tested model is robust through the full hue range.

### Grayscale: no transferable winner

- Test: YOLOv8 leads both metrics at 0.167/0.235, narrowly ahead of YOLOv12.
- Competition: YOLOv12 has the highest AP (0.196), while YOLOv26_v2 has the highest AR (0.279) but only 0.154 AP.

**Recommendation:** choose YOLOv12 when precision/localization consistency matters more, or YOLOv26_v2 when recall matters more, on competition-like grayscale imagery. The test winner does not transfer, so this conclusion needs labeled competition validation.

## Similarities between datasets

1. **The degradation ordering transfers.** Motion blur, darkening, downscaling, stretching, and mid-range hue shifts all become worse as perturbation severity increases (apart from expected hue wraparound and the resolution artifact).
2. **AP and AR usually move together.** Most model winners are the same for both metrics, and AR remains slightly above AP.
3. **YOLOv5's blur result is completely stable.** It wins every blur setting on both datasets.
4. **YOLOv5 is also stable at the lowest resolution ratios.** It wins ratios 0.1–0.3 in both domains.
5. **YOLOv26 has repeatable niches under large hue shifts.** It wins at 60 and 140 degrees in both datasets, although 60-degree competition performance is too low for practical use.

## Differences between datasets

1. **Competition frames amplify common environmental degradation.** Brightness and blur suffer the largest additional drops, suggesting that naturally difficult lighting, texture, scale, or motion characteristics compound the synthetic transform.
2. **Stretch changes the preferred model.** YOLOv12 is strongest on FSOCO, while YOLOv26 is strongest through most moderate competition stretch settings.
3. **Brightness changes the preferred model by severity on competition data.** The smooth YOLOv12 advantage on test becomes YOLOv8 → YOLOv26 → YOLOv5 as competition images get darker.
4. **Grayscale behaves differently.** It is the only family whose competition average is not worse, and it creates a large AP/AR split for YOLOv26_v2.
5. **Competition resolution is non-monotonic.** The exact score of 1.0 at ratio 0.5 and the YOLOv26 recovery at 0.7–0.9 do not resemble a normal resolution-degradation curve and are sensitive to the inference resize path.

## Data-quality finding: competition resolution ratio 0.5

At `size_ratio=0.5`, every competition model reports AP=1.000 and AR=1.000. This point should not be used in robustness averages or model selection.

The competition originals are 1280×720, while the 0.5 transformed images are 640×360. The inference scripts call Ultralytics without overriding `imgsz`, and inspection of corresponding YOLOv5 output files shows identical classes/confidences with bounding-box coordinates exactly halved. This is consistent with the original and transformed images reaching the same effective 640-pixel inference representation. It measures a preprocessing equivalence, not immunity to lost image detail.

Recommended correction:

- Evaluate resolution robustness at a fixed explicit model input size while degrading detail by downscaling and then upscaling back to the original dimensions; or
- Keep variable source dimensions but ensure the tested ratios do not coincide with the inference pre-resize, and document the complete resize/letterbox chain.

## Ground-truth cross-check and deployment caution

The test-set `original_vs_gt` baseline is:

| Model | Original-vs-GT AP | Original-vs-GT AR |
|---|---:|---:|
| YOLOv5 | 0.470 | 0.488 |
| YOLOv8 | 0.477 | 0.494 |
| YOLOv12 | 0.475 | 0.492 |
| YOLOv26 | 0.583 | 0.609 |
| YOLOv26_v2 | **0.584** | **0.611** |

This explains why metamorphic consistency alone must not select the production model. YOLOv5 is often the most self-consistent model, but YOLOv26/YOLOv26_v2 detect the labeled test data much more accurately. The test-set `transformed_vs_gt` family averages likewise favor YOLOv26 for hue, brightness, grayscale, and stretch; YOLOv26_v2 for blur; and YOLOv26/YOLOv26_v2 for resolution.

A practical deployment policy is therefore:

1. **Default accuracy-oriented candidate:** YOLOv26 or YOLOv26_v2, based on the labeled FSOCO results.
2. **Blur specialist:** consider YOLOv5 only if competition ground-truth evaluation confirms that its absolute accuracy remains adequate; it is decisively the most repeatable under blur.
3. **Low-resolution specialist:** make the same conditional choice for YOLOv5 at ratios 0.1–0.3.
4. **Competition stretch specialist:** YOLOv26 at ratios 1.6–2.2 is supported by both its strong test ground-truth accuracy and its competition consistency ranking.
5. **Dynamic brightness routing:** YOLOv8 (gamma 1.5–2), YOLOv26 (2.5–3), then YOLOv5 (3.5–5) is supported only as a consistency policy. A safer system would declare the very-dark region degraded and trigger exposure correction, temporal fusion, a fallback sensor, or a reduced-speed mode.

The highest-value next experiment is to annotate a representative subset of competition frames and compute `original_vs_gt` and `transformed_vs_gt`. That will distinguish a model that is consistently correct from one that is merely consistent with its own baseline predictions.
