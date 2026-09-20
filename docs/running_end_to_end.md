# Running end-to-end workflows

## Environment and inputs

Run the examples from the repository root using Bash. Install the experiment dependencies in an environment compatible with the versions in [requirements.txt](../requirements.txt):

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 utils/model_registry.py --list
python3 utils/model_registry.py --validate yolov8
```

The wrappers resolve the repository root themselves, and experiment wrappers activate `venv/bin/activate` when present. The dependency set includes CUDA packages; use an environment suitable for those pins. The lightweight dashboard does not need these packages. Registry validation checks file existence, not whether weights load correctly.

Prepare datasets beneath `data/base_images/<dataset>/`:

```text
images/                 Original images
labels/                 YOLO ground-truth TXT labels, where applicable
bounding_boxes/         Supervisely JSON annotations used by fsoco_sample
annotations_coco.json   COCO annotations for ground-truth evaluation and RQ3
segmented_images/       Segmentation products for cone-colour transformations
```

Ground-truth YOLO labels use normalized `class cx cy width height`. Model prediction TXT files use a different format: `class confidence x1 y1 x2 y2` in pixels. Images and labels must have matching stems. Register a custom dataset's ordered class names in [config/cone_classes.json](../config/cone_classes.json) before cone-colour transformations; only `test` and `fsoco_sample` currently have entries.

The base RQ1 experiment runner reads `data/base_images/<dataset>/images/`. Its `--images-dir` overrides this for one dataset, including a flat unlabelled dataset. The standalone `scripts/run_all_base_transformations.sh` instead reads `segmented_images/` for `fsoco_sample`. Base consistency experiments can use unlabelled images, but the full RQ1 workflow includes stages requiring annotations. RQ2 needs images, YOLO labels and compatible weights. RQ3 requires existing COCO detections, RQ1 metrics and COCO annotations.

## Current execution limitations

Several model adapters still import `utils.visualise`, although the drawing module is now `utils/visualise_detections.py`. This can raise `ModuleNotFoundError` before inference. The corresponding import is `from utils.visualise_detections import visualize_detections as visualise`; use the current module when adding an adapter. These documentation updates do not repair existing adapters.

`run_rq1.sh` defaults to `fsoco_sample`, but the final parameter-plot CLI currently accepts only `test`, `val`, `train_samples`, `competition` and `comp_labelled`. The examples use `test`, which also has a cone-class schema. Other dataset names may need changes in downstream tools even when an experiment wrapper accepts them.

Reruns write to standard paths. The RQ1 base runner explicitly clears selected `yolov26_v2` detection and analysis outputs before regenerating them. Preserve prior local runs separately if needed. RQ2 output directories are keyed by model/search method, not dataset, so changing datasets does not create an independent run directory.

## RQ1: complete experiment and analysis

```bash
bash end_to_end_scripts/run_rq1.sh --dataset test --model v8
```

This runs base transformations and inference, cone-colour transformations and inference, then the seven-stage analysis workflow. `--model` accepts a registry **version** (`v5`, `v8`, `v12`, `v26`, `v26_v2`) or `all`. This limits experiment inference; the final analysis wrapper scans available results rather than accepting a model filter.

| Option | Meaning |
| --- | --- |
| `--dataset test` or `--datasets test,val` | Dataset or comma-separated datasets; downstream prerequisites still apply |
| `--model v8` | Selected registry version; default `all` |
| `--viz-samples 0` | Disable saved detection illustrations; default 5 per result directory |
| `--skip-transform` | Reuse existing transformed images; inference and analysis still run |

Run only part of RQ1 when appropriate:

```bash
# Base rules, inference and base evaluation
bash end_to_end_scripts/run_rq1_base_experiment.sh --dataset test --model v8

# Base inference with existing transformed images, without analysis
bash end_to_end_scripts/run_rq1_base_experiment.sh --dataset test --model v8 --skip-transform --skip-analysis

# Selected cone-colour rules; generates missing segmentation products
bash end_to_end_scripts/run_rq1_cone_colour_experiment.sh \
  --dataset test --model v8 --transformations yellow_to_blue,blue_to_yellow

# All RQ1 analysis from existing detections
bash end_to_end_scripts/run_rq1_analysis.sh --dataset test
```

The base runner also accepts `--config <path>`, `--images-dir <path>`, `--analysis-only` and `--skip-analysis`. Its default configuration is `config/brute_force_transformations.json`, including variant IDs. These extra flags are not all forwarded by `run_rq1.sh`; call the base runner directly when using them.

The cone-colour runner defaults to yellow→blue, blue→yellow, yellow→orange and blue→orange. It accepts `--skip-analysis` and `--skip-transform`.

The analysis-only wrapper has seven steps: base evaluation, cone-colour evaluation, cone-colour plots, confidence-change plots, detection-overlap calculation, detection-overlap plots and parameter-response plots. `--skip-steps N` skips the first N steps (0–7); it is not a step selector.

Outputs:

| Product | Location |
| --- | --- |
| Transformed images | `data/transformed_images/<output-dataset>/<variant>/` |
| Original/transformed detections | `results/<dataset>/<model>/{original,<variant>}/` |
| Per-variant metrics | `analysis/results/rq1/<dataset>/<model>/<variant>/<comparison>/stats.json` |
| Original baseline | `analysis/results/rq1/<dataset>/<model>/original_vs_gt/stats.json` |
| Aggregate plots and exports | `analysis/results/rq1/<dataset>/` |

Transformed-image output names map `test` to `fsoco_test` and `val` to `fsoco_val`; inference and analysis retain `test`/`val`. Custom datasets retain their names.

## RQ2: brightness search, scoring and distributions

```bash
bash end_to_end_scripts/run_rq2.sh \
  --dataset test --models yolov8 --num-images 10 --device cuda:0
```

The wrapper runs **both** brute-force and golden-section search, compares their results, computes brightness scores and generates performance distributions. Its `--models` uses registry **names**, comma-separated, or `all` (the default).

| Option | Default / behavior |
| --- | --- |
| `--dataset` | `fsoco_sample` |
| `--target-ap` | `0.5` |
| `--num-images` | No limit; set a small limit for smoke runs |
| `--device` | `cuda:0`; use `cpu` where appropriate |
| `--workers` | `1`, for brightness scoring |
| `--search-method` | `brute_force`; chooses the source for RQ2b/c, not which RQ2a searches run |
| `--start-at` | `rq2a`; use `rq2b` or `rq2c` to consume existing earlier outputs |

For example, resume scoring and distributions from saved golden-section results:

```bash
bash end_to_end_scripts/run_rq2.sh --models yolov8 --start-at rq2b --search-method golden_section
```

Results are under `analysis/results/rq2/rq2a/`, `rq2b/` and `rq2c/`. Holdout validation and search-trace visualization are separate [individual scripts](running_individual_scripts.md), not stages invoked by this wrapper.

## RQ3: sample-size equivalence

RQ3 performs analysis of existing results; it does not generate transformations or run inference. Prepare `results/test/<model>/` COCO detections, `analysis/results/rq1/test/<model>/` metrics and `data/base_images/test/annotations_coco.json` first.

```bash
# Default configuration: all five listed models, test data, 50%, 2,000 repetitions
bash end_to_end_scripts/run_rq3.sh

# Smaller diagnostic analysis using one model's existing results
bash end_to_end_scripts/run_rq3.sh \
  --models yolov8 --sample-fraction 0.1 --bootstrap-repetitions 20 --workers 2
```

The wrapper forwards arguments to `analysis/utils/calculate_sample_equivalence.py`. Defaults include 90% confidence intervals, ±0.05 equivalence margin and seed `20260810`. A small bootstrap run is for checking execution, not for drawing the paper's conclusions.

Use `--results-root`, `--analysis-results-root`, `--data-root` and `--output-dir` to select other input/output trees. For archived inputs, point `--results-root` at `original_results` and `--analysis-results-root` at `analysis/original_results/rq1`, after checking the needed detections and annotations are present. The default output remains local.

Other controls include `--source-dataset`, `--rules`, `--confidence-level`, `--equivalence-margin`, `--seed` and `--skip-plots`. RQ3's default model list is explicit in the Python script, not dynamically expanded from the registry; pass `--models` for new models.

Outputs are `analysis/results/rq3/sample_equivalence/results.json`, `report.md` and `plots/`. Open the dashboard, select **My experiments**, and refresh to inspect them.
