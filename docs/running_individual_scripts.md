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

RQ2b and RQ2c consume the earlier search outputs:

```bash
python3 scripts/rq2/rq2b_brightness_calculation.py \
  --model yolov8 --search-method brute_force --workers 2

python3 scripts/rq2/rq2c_perf_comparison.py \
  --models yolov8 --search-method brute_force
```

For explicit RQ2b inputs, use `--results-json <path> [<path> ...]`; it overrides search-method discovery. RQ2b `--model` and RQ2c `--models` accept space-separated model names, whereas the end-to-end wrapper uses a comma-separated list. Multiple-model RQ2b scoring produces a combined model-group directory.

Two additional programs are not invoked by `run_rq2.sh`:

| Program | Use |
| --- | --- |
| `scripts/rq2/rq2a_visualize_gamma_search.py` | Visualize saved gamma-search traces; inspect `--help` for model/method and image selection |
| `scripts/rq2/rq2b_validate_brightness_method.py` | Validate learned brightness thresholds on separate labelled holdout data |

For holdout validation:

```bash
python3 scripts/rq2/rq2b_validate_brightness_method.py \
  --model yolov8 --num-images 10
```

This requires the RQ2b results and holdout inputs, including an `annotations_coco.json` file and its referenced images under `images/`. Defaults are `data/base_images/comp_labelled` and `results/competition_labelled`; override with `--holdout-root`, `--detections-root` and `--rq2b-results` as needed. It runs additional inference; it is not just a report renderer.

## Run RQ3 analysis directly

```bash
python3 analysis/utils/calculate_sample_equivalence.py \
  --models yolov8 --sample-fraction 0.5 --bootstrap-repetitions 2000 --workers 2
```

This is the program behind `run_rq3.sh`, with identical prerequisites and options. [render_sample_equivalence.py](../analysis/visualisation/render_sample_equivalence.py) provides report/plot rendering; inspect `--help` before rerendering existing JSON. RQ3 defaults and interpretation are described in the [end-to-end](running_end_to_end.md) and [paper-result](research_paper_results.md) guides.

After generating outputs, refresh **My experiments** in the [dashboard](running_dashboard.md).
