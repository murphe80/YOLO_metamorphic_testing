# Running individual scripts

Run these commands from the repository root with the experiment environment activated. See [end-to-end prerequisites and current limitations](running_end_to_end.md) for data preparation and known inference-import issues. Replace example image paths with real files. Use Python programs' `--help` to inspect their full interfaces; some import experiment dependencies before displaying help.

## Generate transformations

Apply one rule to one image:

```bash
python3 utils/transform_base_metamorphic_rules.py \
  --rule imgDarkBright \
  --config '{"brightness_multiplier":0.2,"gamma":1,"method":"gamma"}' \
  --base_image data/base_images/test/images/example.jpg \
  --output_path data/transformed_images/fsoco_test/imgDarkBright
```

Available base rules are `imgDarkBright`, `imgResolution`, `imgMotionBlur`, `imgStretch`, `imgGrayscale` and `imgColorWheel`.

```bash
# Six base transformations without inference
bash scripts/run_all_base_transformations.sh --dataset test

# Segmentation from labels, followed by cone-colour transformations
bash scripts/generate_cone_segmentation_masks.sh --dataset test
bash scripts/run_all_cone_color_transformations.sh --dataset test
```

The standalone base transformation shell script uses one hard-coded configuration per rule. It does **not** generate the full variant grid in `config/brute_force_transformations.json`. Use `run_rq1_base_experiment.sh --config <path>` for the configured experiment grid; do not use the standalone outputs with `--skip-transform` unless they match the expected variants. Cone-colour transformations require a dataset class schema and segmentation inputs.

## Run inference on one image

After resolving the adapter import issue described in the prerequisites:

```bash
python3 models/yolov8/infer_v8.py \
  --weights models/yolov8/yolov8_param_kfold.pt \
  --image data/base_images/test/images/example.jpg \
  --device cpu --output-subdir ../test/yolov8/original
```

This writes `results/test/yolov8/original/example.txt`. Plain `--output-subdir original` instead writes under `results/yolov8/original`, without the dataset level. Use the dataset layout when feeding the analysis runners. Substitute the registered script and weights for other models; see [adding models](adding_models.md) for the output contract.

## Convert and evaluate detections

With COCO annotations already available:

```bash
python3 analysis/utils/format_detections.py \
  --detection_dir results/test/yolov8/original \
  --output_file results/test/yolov8/original/detections_coco.json \
  --gt_coco data/base_images/test/annotations_coco.json

python3 analysis/coco_eval.py \
  --gt data/base_images/test/annotations_coco.json \
  --dt results/test/yolov8/original/detections_coco.json \
  --image-dir data/base_images/test/images \
  --stats-out analysis/results/rq1/test/yolov8/original_vs_gt/stats.json

python3 analysis/utils/evaluate_coco_per_class.py \
  --gt data/base_images/test/annotations_coco.json \
  --dt results/test/yolov8/original/detections_coco.json \
  --stats-out analysis/results/rq1/test/yolov8/per_class_stats.json
```

`--gt_coco` aligns detection image IDs to the annotations. `--image-dir` defines the evaluation subset, including images with empty detection files. For original-vs-transformed consistency and geometric/class adjustments, use the analysis runners instead of substituting an arbitrary detection list as ground truth:

```bash
bash analysis/scripts/run_base_metamorphic_analysis.sh --dataset test
bash analysis/scripts/run_cone_colour_analysis.sh --dataset test
```

[convert_bounding_boxes_to_coco.py](../analysis/utils/convert_bounding_boxes_to_coco.py) handles annotation conversion. Inspect its CLI for the input format you have.

## Generate RQ1 plots and overlap metrics

New false-positive confidence distributions can be generated from saved detections with
`venv/bin/python analysis/visualisation/plot_confidence_change.py --dataset test --new-fp-only`.
This writes one box-plot figure and a JSON file of confidence values per rule under
`analysis/results/rq1/test/confidence_change/new_FP/`, including `cone_colour.pdf`.
Each model has a panel, each setting has a box, and confidence axes span 0–1.
New FPs are transformed false positives with no spatial match to an original FP;
FP membership uses category-aware GT matching, including adjusted cone-colour labels.
A TP that becomes an unmatched FP is included. These are absolute transformed
confidence scores, not confidence differences or the separate three-way overlap
metric's `trans_only` region. JSON records the definition, counts, scores, and hue cohorts.

```bash
python3 analysis/visualisation/plot_parameter_variations.py \
  --dataset test --comparison original_vs_transformed

python3 analysis/visualisation/plot_parameter_variations.py \
  --dataset test --comparison transformed_vs_gt

python3 analysis/visualisation/plot_cone_colour_analysis.py --datasets test

python3 analysis/visualisation/plot_confidence_change.py \
  --dataset test --results_dir results/test \
  --gt_path data/base_images/test/annotations_coco.json \
  --output_dir analysis/results/rq1/test/confidence_change

python3 analysis/utils/calculate_detection_overlap_metrics.py \
  --dataset test --results_dir results/test \
  --gt_path data/base_images/test/annotations_coco.json \
  --output_path analysis/results/rq1/test/detection_overlap_metrics/detection_overlap_metrics.json

python3 analysis/visualisation/plot_detection_overlap_metrics.py \
  --dataset test \
  --statistics analysis/results/rq1/test/detection_overlap_metrics/detection_overlap_metrics.json \
  --output-dir analysis/results/rq1/test/detection_overlap_metrics/plots
```

Overlap bars retain their existing total rates and metric colours, with stacked
size sections distinguished by shades of each metric's colour: small (light),
medium (original colour), and large (dark). Thin white separators mark the
sections. Each bar is labelled with its metric; the shade legend identifies
size only. Charts are vector PDFs.

Cone-colour detection-overlap metrics and the cone-colour hallucination chart
use source-only cohorts, matching imgColorWheel. For yellow-to-blue, only
yellow-source references enter the cohort and their expected labels become
blue. Predictions owned by other source classes are excluded; wrong-class
predictions at source objects and unowned predictions remain error candidates.
The metrics JSON records the cohort, source/target mapping, and evaluation policy.
Rerun the analysis before plotting older JSON files that lack size information.

Each result (including each hue source/target cohort) now includes
`size_breakdown.regions` for the seven overlap groups and
`size_breakdown.metrics` for the five plotted metrics. Each entry contains:

- `counts`: raw object counts by size.
- `weighted_counts`: counts divided by each image's total across all seven
  groups, then summed over images, preserving the existing normalisation.
- `proportions`: each size's share of that weighted region or metric numerator.
- `rate_contributions` (metrics only): each size's weighted numerator divided
  by the existing metric denominator. These are the stacked bar heights and
  sum to the unchanged total metric. Empty groups have zero proportions.

Size uses area in **transformed-image pixels**, with COCO cutoffs at 32² and
96² pixels: small < 1024, medium 1024–<9216, large ≥ 9216. Exact cutoffs are
assigned to the larger bin so the stacks are disjoint (COCO AP's inclusive
ranges overlap at the cutoffs). Regions with a ground-truth member use its
area projected into the transformed image. Otherwise, an existing transformed
prediction uses its native box area; a lost original prediction uses its box
area projected into the transformed image. Thus a newly introduced FP is
classified by its size after stretching/resizing, not its back-projected size.

These consume existing detections and metrics. `analysis/scripts/generate_detection_overlap_metrics.sh` is a convenience runner for the test dataset. Use `end_to_end_scripts/run_rq1_analysis.sh --dataset test` to coordinate all seven analysis stages.

## Run RQ2 steps independently

RQ2a requires registered weights, dataset images and YOLO labels:

```bash
python3 scripts/rq2/rq2a_brute_force_search.py \
  --dataset test --model yolov8 --num-images 10 --target-ap 0.5

python3 scripts/rq2/rq2a_golden_section_search.py \
  --dataset test --model yolov8 --num-images 10 --target-ap 0.5

python3 scripts/rq2/rq2a_compare_results.py \
  --brute-force analysis/results/rq2/rq2a/brute_force/yolov8/results.json \
  --golden-section analysis/results/rq2/rq2a/golden_section/yolov8/results.json
```

The RQ2a comparison writes two plots to `analysis/results/rq2/rq2a/comparison/<model>/`:
`runtime_efficiency.pdf` retains the brute-force versus golden-section runtime scatter,
and `threshold_accuracy.pdf` shows two violins of `|γ_GS − γ_BF|`: brightened
threshold (left crossing) and darkened threshold (right crossing), matched by
image and side. Crossing errors and the median are overlaid. Missing crossing
pairs are excluded; zero means exact agreement. `comparison.json` retains the
per-image data and summary statistics. The former six-panel `comparison.pdf`
is no longer generated.

RQ2b and RQ2c consume the earlier search outputs:

```bash
python3 scripts/rq2/rq2b_brightness_calculation.py \
  --model yolov8 --search-method brute_force --workers 2

python3 scripts/rq2/rq2c_perf_comparison.py \
  --models yolov8 --search-method brute_force
```

For explicit RQ2b inputs, use `--results-json <path> [<path> ...]`; it overrides search-method discovery. RQ2b `--model` and RQ2c `--models` accept space-separated model names, whereas the end-to-end wrapper uses a comma-separated list. Multiple-model RQ2b scoring produces a combined model-group directory.

RQ2c writes separate `<model>_<method>_performance_distribution_original.pdf`,
`_lower.pdf`, and `_upper.pdf` files, each with one violin plot on the same AP scale.

To add an RQ2c comparison using saved AP results from both search methods:

```bash
venv/bin/python scripts/rq2/rq2c_perf_comparison.py --models yolov8 \
  --overlay-from-summaries \
    analysis/results/rq2/rq2c/performance_distributions/yolov8/results_brute_force.json \
    analysis/results/rq2/rq2c/performance_distributions/yolov8/results_golden_section.json \
  --output-dir analysis/results/rq2/rq2c/performance_distributions/yolov8
```

This writes separate `<model>_overlapping_performance_distribution_original.pdf`,
`_lower.pdf`, and `_upper.pdf` files. Lower and upper threshold plots overlay brute force
and golden search with transparent colours and a method legend. Original
includes all unique images from either summary, counting shared images once
(their AP values use `--search-method`, golden section by default). Each method keeps
all its saved observations, so sample counts can differ. Existing plots and AP
summaries are retained; AP is not recomputed. Models without records from both
actual search methods are skipped, including summaries that used a fallback.

These additional programs are not invoked by `run_rq2.sh`:

| Program | Use |
| --- | --- |
| `scripts/rq2/rq2a_visualize_gamma_search.py` | Visualize saved gamma-search traces; inspect `--help` for model/method and image selection |
| `scripts/rq2/rq2b_validate_brightness_method.py` | Reassess saved validation observations using existing RQ2b boundary scores and three-way predictions |

The test-dataset brightness scores already exist in `analysis/results/rq2/rq2b/brightness_scoring/brute_force/<model>/results.json`. Use these as the boundary calibration input and reassess the saved validation AP/brightness observations:

```bash
venv/bin/python scripts/rq2/rq2b_validate_brightness_method.py \
  --dataset val --model yolov8 --sigma-multiplier 0.204
```

This reads two existing files:

- Boundary scores: `analysis/results/rq2/rq2b/brightness_scoring/brute_force/yolov8/results.json`.
- Evaluation observations (AP >= 0.1): `analysis/results/rq2/rq2b/validation/val/yolov8/filtered_ap_0.1/validation_results.json`.

Override them independently with `--rq2b-results PATH` and `--validation-results PATH`. The boundary file contains `scores_right` (dark) and `scores_left` (bright) for each model/image; it is not a full evaluation dataset. The evaluation file supplies the original/transformed AP and brightness measurements to classify. By default, it is the saved `filtered_ap_0.1` subset (AP >= 0.1); pass `--validation-results` explicitly to assess a different saved population. Existing binary schema-3 validation files and schema-4/5/6 results are supported. `--dataset` identifies the evaluation dataset and defaults to `val`; use `--dataset competition_dataset` for the saved competition observations.

The validation script performs **no model inference, brightness scoring, or boundary search**. The redundant standalone scoring utility has been removed. No extra scoring cache is needed. All available saved boundary scores are used without new AP-based filtering. Separate dark and bright distributions use sample SD (`ddof=1`) and mean ± `--sigma-multiplier` × SD (default 0.204 for every brightness method and model).

Scores inside either uncertainty band, including endpoints, predict `uncertain`. Scores strictly between separated bands predict `in_range`; scores beyond the outer limits predict `out_of_range`. Limits are not clipped. Overlapping bands leave no in-range interval. Fewer than two scores on either side, or reversed boundary means, produce uncertain predictions with a reason. The original evaluation population, AP threshold, and any existing AP-filter metadata are preserved; old predictions are replaced.

Uncertain predictions receive no correct/incorrect label. The separate `uncertain_residuals.pdf` compares signed residuals `AP - 0.5` across all methods using box plots with a shared axis and sample counts. Lines inside the boxes show medians; no mean markers are drawn. JSON and report tables include residual summaries (mean, mean absolute distance, RMSE, and quantiles).

Confusion matrices contain only definite predictions, with actual and predicted in-range/out-of-range as the two rows and columns. Actual in-range means AP >= the saved threshold (normally 0.5), and actual out-of-range means AP < that threshold. Cell percentages use each actual row among definite predictions. Reports retain prediction counts, coverage, accuracy among definite predictions, and false in-range/out-of-range rates; the latter rates continue to use all actual AP failures/successes as their denominators. Each matrix panel states the uncertain count excluded and coverage. Reports identify the separate calibration/evaluation sources.

Updated schema-6 JSON, reports, boundary histograms, AP scatter plots, and confusion matrices are saved under `analysis/results/rq2/rq2b/validation/<dataset>/<model>/three_class/`. This preserves the input validation results. `--output-dir` overrides the base directory, with `<dataset>/<model>/three_class/` appended.

`boundary_distributions.pdf` compares the seven bounded methods with shared x limits (0–1), y limits (0–100% of boundary images), and identical 0.04-wide bins. CIELAB L* is divided by 100 for display; RMS retains its native scale. The mean-shift separation-to-spread ratio is omitted from all RQ2b plots. These display choices do not change the stored scores or prediction bands. Confusion-matrix panels show only overall accuracy below the method name, calculated across the predictions included in each matrix.

Sample export is disabled by default, so reassessment needs only the two JSON files. Set `--samples-per-category 5` to recreate representative in-range, uncertain, and out-of-range images from the evaluation dataset's original image directory. `--holdout-root` can relocate that directory for sample export. Recreating samples does not run inference or brightness scoring. The legacy binary AP-filtering utility remains restricted to schema <= 3.

Run the reassessment checks with:

```bash
venv/bin/python -m unittest scripts.rq2.test_rq2b_calibration -v
```

## Run RQ3 analysis directly

```bash
python3 analysis/utils/calculate_sample_equivalence.py \
  --models yolov8 --sample-fraction 0.5 --bootstrap-repetitions 2000 --workers 2
```

This is the program behind `run_rq3.sh`, with identical prerequisites and options. [render_sample_equivalence.py](../analysis/visualisation/render_sample_equivalence.py) provides report/plot rendering; inspect `--help` before rerendering existing JSON. RQ3 defaults and interpretation are described in the [end-to-end](running_end_to_end.md) and [paper-result](research_paper_results.md) guides.

After generating outputs, refresh **My experiments** in the [dashboard](running_dashboard.md).
