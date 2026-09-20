# Inspecting research paper results

## Open the archived evidence

```bash
python3 dashboard/serve.py
```

Open **http://127.0.0.1:8080**, select **Paper results**, then select a research question. These views use `analysis/original_results/`; **My experiments** uses `analysis/results/`. Viewing saved results does not require inference or the original images.

`original_results/` at the repository root holds archived inference outputs. It is distinct from the archived analysis under `analysis/original_results/`, which is what the dashboard reads. Old absolute source paths inside archived JSON are provenance, not paths followed by the dashboard.

## RQ1: transformation robustness

The saved analysis is under `analysis/original_results/rq1/test/`.

| Relative path beneath that directory | Evidence |
| --- | --- |
| `<model>/original_vs_gt/stats.json` | Original-image accuracy against labelled ground truth |
| `<model>/<variant>/original_vs_transformed/stats.json` | Consistency using original model detections as the reference |
| `<model>/<variant>/transformed_vs_gt/stats.json` | Transformed-image accuracy against ground truth |
| `parameter_variations/` | AP/AR plots and `metrics_<comparison>.json` exports |
| `detection_overlap_metrics/` | Retained, lost and new detection metrics and plots |
| `confidence_change/` | Confidence-change figures |
| `cone_colour_analysis/` | Cone-colour comparisons |

In the dashboard, choose **Detection consistency** or **Accuracy against ground truth**, then use **Explore a transformation**. AP is COCO AP@[.50:.95]; AR uses `AR100` or the exported `AR` value. Missing, non-finite and COCO `-1` metrics are unavailable, not zero. Consistency measures agreement with the model's own predictions and can remain high even when those predictions are wrong.

Raw `stats.json` metrics take precedence over metric exports. Exports supply saved parameter labels; raw-only records use the current transformation config. Baselines describe this dataset, not an external COCO benchmark. Detection-overlap rates alone do not identify intersections of failed-image sets or establish train/test generalisation.

## RQ2: brightness tuning

Paths below are relative to `analysis/original_results/rq2/`:

| Path | Evidence |
| --- | --- |
| `rq2a/brute_force/<model>/results.json` | Brute-force threshold-search results |
| `rq2a/golden_section/<model>/results.json` | Golden-section threshold-search results |
| `rq2a/comparison/<model>/` | Saved matched-image comparison statistics and plots |
| `rq2b/brightness_scoring/<method>/<model-or-model-group>/` | Brightness-score results and plots |
| `rq2b/validation/<model>/` | Separate holdout-validation results, where available |
| `rq2c/performance_distributions/` | Original/transformed accuracy distributions |

Use the three RQ2 dashboard tabs to follow search efficiency, brightness scoring and performance distributions. Mean per-image speedup is not the ratio of mean runtimes. Check saved dataset, threshold, processed/skipped/failed counts and search method before comparing runs. The workflow wrapper does not generate the separate holdout-validation outputs automatically.

## RQ3: sample-size equivalence

Start with the saved [report](../analysis/original_results/rq3/sample_equivalence/report.md), [results JSON](../analysis/original_results/rq3/sample_equivalence/results.json) and plots under `analysis/original_results/rq3/sample_equivalence/plots/`.

The archived experiment covers five models, six rules and 51 settings on 363 test images. Its fixed sample contains 182 images (50%, rounded). It uses 2,000 independent bootstrap repetitions per arm, 90% percentile confidence intervals and a ±0.05 equivalence margin.

The two analyses answer different questions:

- **Detection consistency:** original detections versus transformed detections, without baseline subtraction.
- **Accuracy degradation:** original ground-truth AP/AR minus transformed ground-truth AP/AR.

An interval wholly inside the equivalence margin supports equivalence; one wholly beyond a boundary supports a meaningful difference; other intervals are inconclusive. Inconclusive does not mean different. Read the saved validation section for discrepancies between recomputed full-dataset metrics and stored point estimates. The dashboard uses the saved classifications and metadata.

The results apply to that sample and experiment. They do not establish that 10% suffices, nor that one metamorphic test subsumes another. Recompute a different sample with the [RQ3 workflow](running_end_to_end.md), which also requires raw detections and annotations; aggregate paper metrics alone are insufficient.
