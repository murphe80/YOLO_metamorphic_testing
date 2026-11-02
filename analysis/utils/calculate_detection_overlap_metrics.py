#!/usr/bin/env python3
"""Calculate detection-overlap statistics for metamorphic transformations.

The utility compares original detections, transformed detections, and ground
truth annotations, then writes all seven overlap regions and their derived
retention metrics to one JSON file. It does not generate Venn diagrams.
Each region and metric also contains a small/medium/large size breakdown in
transformed-image pixels, preserving the existing image-normalized totals.
For cone-colour rules, source-only original predictions and ground truth are
mapped to the expected target class using the same cohort policy as imgColorWheel.

Defaults are resolved from the repository root, so the command can be run from
any working directory.

Examples:
    python3 analysis/utils/calculate_detection_overlap_metrics.py
    python3 analysis/utils/calculate_detection_overlap_metrics.py --dataset val
    python3 analysis/utils/calculate_detection_overlap_metrics.py --iou-threshold 0.7
    python3 analysis/utils/calculate_detection_overlap_metrics.py \
        --model yolov8 --transformation imgGrayscale
"""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils.detection_matching import (  # noqa: E402
    aggregate_three_way_overlap,
    compute_retention_metrics,
)
from analysis.utils.analyse_cone_colours import (  # noqa: E402
    TRANSFORMATION_MAP,
)
from analysis.utils.format_detections import format_detections  # noqa: E402
from analysis.utils.color_wheel import (  # noqa: E402
    metadata as hue_metadata, cone_colour_mapping, overlap_cohort, OVERLAP_POLICY,
)
from analysis.utils.format_gt import convert_to_coco_format  # noqa: E402
from analysis.utils.transform_geometric_detections import (  # noqa: E402
    normalize_transform_params,
    transform_detections_file,
)


MODEL_ORDER = ["yolov5", "yolov8", "yolov12", "yolov26_v2"]
BASE_TRANSFORMATIONS = [
    "imgStretch",
    "imgResolution",
    "imgMotionBlur",
    "imgDarkBright",
    "imgGrayscale",
    "imgColorWheel",
]
GEOMETRIC_TRANSFORMS = {"imgStretch", "imgResolution"}
DEFAULT_DATASET = "test"
DEFAULT_BRUTE_FORCE_CONFIG = PROJECT_ROOT / "config" / "brute_force_transformations.json"


def dataset_results_dir(dataset: str) -> Path:
    """Return the default detection-results directory for a dataset."""
    return PROJECT_ROOT / "results" / dataset


def dataset_gt_path(dataset: str) -> Path:
    """Return the default ground-truth path for a dataset."""
    return PROJECT_ROOT / "data" / "base_images" / dataset / "annotations_coco.json"


def dataset_output_path(dataset: str) -> Path:
    """Return the default statistics output path for a dataset."""
    return (
        PROJECT_ROOT
        / "analysis"
        / "results"
        / "rq1"
        / dataset
        / "detection_overlap_metrics"
        / "detection_overlap_metrics.json"
    )


def infer_gt_path(results_dir: Path, dataset: str) -> Path:
    """Infer the dataset ground-truth path from a results directory."""
    inferred_dataset = results_dir.name if results_dir.name != "results" else dataset
    inferred_path = dataset_gt_path(inferred_dataset)
    if inferred_path.parent.exists():
        return inferred_path
    return dataset_gt_path(dataset)


def ensure_gt_coco(gt_path: Path) -> Path:
    """Create a missing COCO GT file from sibling bounding-box JSON or YOLO labels."""
    if gt_path.exists():
        return gt_path

    dataset_dir = gt_path.parent
    bounding_boxes_dir = dataset_dir / "bounding_boxes"
    labels_dir = dataset_dir / "labels"
    images_dir = dataset_dir / "images"

    if bounding_boxes_dir.is_dir():
        print(f"Ground truth COCO not found. Generating {gt_path} from {bounding_boxes_dir}...")
        convert_to_coco_format(
            input_dir=str(bounding_boxes_dir),
            output_file=str(gt_path),
            image_dir=str(images_dir) if images_dir.is_dir() else None,
        )
        return gt_path

    if labels_dir.is_dir():
        print(f"Ground truth COCO not found. Generating {gt_path} from YOLO labels in {labels_dir}...")
        format_detections(
            detection_dir=str(labels_dir),
            output_file=str(gt_path),
            as_ground_truth=True,
            is_yolo_format=True,
            image_dir=str(images_dir) if images_dir.is_dir() else None,
        )
        return gt_path

    raise FileNotFoundError(
        f"Ground truth not found: {gt_path}. Expected either "
        f"{bounding_boxes_dir} or {labels_dir} to generate it."
    )


def resolve_base_transformation(results_dir: Path, model: str, transformation: str) -> str:
    """Resolve brute-force variant directories back to their base transformation."""
    transform_dir = results_dir / model / transformation

    for config_name in ("configs.json", "config.json"):
        config_path = transform_dir / config_name
        if not config_path.exists():
            continue
        try:
            with open(config_path) as file:
                config = json.load(file)
        except json.JSONDecodeError:
            continue

        if isinstance(config, dict):
            if isinstance(config.get("transformation"), str):
                return config["transformation"]
            for key in config:
                if key in BASE_TRANSFORMATIONS:
                    return key

    for base_name in BASE_TRANSFORMATIONS:
        if transformation == base_name or transformation.startswith(f"{base_name}_"):
            return base_name

    return transformation


def resolve_transform_params(
    results_dir: Path,
    model: str,
    transformation: str,
    base_transformation: str,
    brute_force_config: Path,
) -> Optional[dict]:
    """Resolve variant-specific geometric transformation parameters."""
    transform_dir = results_dir / model / transformation

    for config_name in ("configs.json", "config.json"):
        config_path = transform_dir / config_name
        if not config_path.exists():
            continue
        try:
            with open(config_path) as file:
                config = json.load(file)
        except json.JSONDecodeError:
            continue

        if not isinstance(config, dict):
            continue

        nested_config = config.get(base_transformation)
        if isinstance(nested_config, dict):
            return nested_config

        geometric_keys = {
            "scale_x",
            "scale_y",
            "width_ratio",
            "height_ratio",
            "size_ratio",
        }
        if any(key in config for key in geometric_keys):
            return config

    if not brute_force_config.exists():
        return None

    try:
        with open(brute_force_config) as file:
            config = json.load(file)
    except json.JSONDecodeError:
        return None

    variants = config.get(base_transformation, [])
    if isinstance(variants, dict):
        variants = [variants]

    for variant in variants:
        if isinstance(variant, dict) and variant.get("variant_id") == transformation:
            return variant

    return None


def discover_models(results_dir: Path) -> list:
    """Discover model directories that contain original detections."""
    if not results_dir.exists():
        return []

    models = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        original_path = model_dir / "original" / "detections_original_coco.json"
        if original_path.exists():
            models.append(model_dir.name)

    return sorted(
        models,
        key=lambda model: (
            MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER),
            model,
        ),
    )


def discover_transformations(results_dir: Path, models: list) -> list:
    """Discover legacy-rule and brute-force-variant result directories."""
    transformations = set()

    for model in models:
        model_dir = results_dir / model
        if not model_dir.exists():
            continue

        for transform_dir in model_dir.iterdir():
            if not transform_dir.is_dir() or transform_dir.name.startswith("."):
                continue
            if transform_dir.name == "original":
                continue
            if (transform_dir / "detections_coco.json").exists():
                transformations.add(transform_dir.name)

    return sorted(transformations)


def calculate_statistics_for_model_transformation(
    model: str,
    transformation: str,
    results_dir: Path,
    gt_path: Path,
    iou_threshold: float = 0.5,
    brute_force_config: Path = DEFAULT_BRUTE_FORCE_CONFIG,
    require_category_match: bool = True,
) -> dict:
    """Calculate overlap statistics for one model/transformation pair."""
    original_path = results_dir / model / "original" / "detections_original_coco.json"
    transformed_path = results_dir / model / transformation / "detections_coco.json"

    if not original_path.exists():
        raise FileNotFoundError(f"Original detections not found: {original_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    base_transformation = resolve_base_transformation(results_dir, model, transformation)

    if not transformed_path.exists():
        if base_transformation in GEOMETRIC_TRANSFORMS:
            print(f"  Generating COCO detections for {transformation}...")
            format_detections(
                detection_dir=str(results_dir / model / transformation),
                output_file=str(transformed_path),
            )
        else:
            raise FileNotFoundError(f"Transformed detections not found: {transformed_path}")

    native_transformed_path = transformed_path
    reference_area_scale = 1.0
    if base_transformation in GEOMETRIC_TRANSFORMS:
        transform_params = resolve_transform_params(
            results_dir, model, transformation, base_transformation, brute_force_config,
        )
        coordinate_scale = normalize_transform_params(base_transformation, transform_params)
        if coordinate_scale is None:
            raise ValueError(f"Missing geometric parameters for {transformation}; cannot determine transformed object sizes")
        # Matching uses original coordinates. Sizes use the actual transformed
        # detections, with GT/lost-original areas projected forward into that space.
        reference_area_scale = 1.0 / (coordinate_scale["scale_x"] * coordinate_scale["scale_y"])
        pre_transformed_path = (
            results_dir / model / transformation / "detections_coco_transformed.json"
        )

        if pre_transformed_path.exists():
            print(f"  Using pre-computed transformed coordinates from: {pre_transformed_path.name}")
            transformed_path = pre_transformed_path
        else:
            print(f"  Computing coordinate transformation for {transformation}...")
            if transform_params:
                print(f"  Using transform parameters for {transformation}: {transform_params}")

            try:
                transform_detections_file(
                    detection_file=str(transformed_path),
                    original_image_dir=str(results_dir / model / "original"),
                    transformed_image_dir=str(results_dir / model / transformation),
                    transformation_name=base_transformation,
                    output_file=str(pre_transformed_path),
                    transform_params=transform_params,
                )
                transformed_path = pre_transformed_path
                print("  ✓ Coordinate transformation complete")
            except Exception as error:
                raise RuntimeError(
                    f"Failed to transform coordinates for {transformation}: {error}"
                ) from error

    if base_transformation == "imgColorWheel":
        result = hue_metadata(transformation)
        result["evaluation_policy"] = OVERLAP_POLICY
        result["class_shifts"] = {}
        original = json.loads(original_path.read_text())
        transformed = json.loads(transformed_path.read_text())
        ground_truth = json.loads(gt_path.read_text())
        with TemporaryDirectory(prefix="color_wheel_overlap_") as directory:
            paths = [Path(directory) / name for name in ("original.json", "transformed.json", "gt.json")]
            for mapping in result["included_mappings"]:
                cohort = overlap_cohort(original, transformed, ground_truth, mapping)
                for path, data in zip(paths, cohort):
                    path.write_text(json.dumps(data))
                statistics = aggregate_three_way_overlap(
                    *paths, iou_threshold, require_category_match, include_sizes=True)
                statistics.update(compute_retention_metrics(statistics))
                result["class_shifts"][mapping["key"]] = {**mapping, **statistics}
        return result

    if base_transformation in TRANSFORMATION_MAP:
        mapping = cone_colour_mapping(base_transformation)
        cohort = overlap_cohort(
            json.loads(original_path.read_text()),
            json.loads(transformed_path.read_text()),
            json.loads(gt_path.read_text()),
            mapping,
        )
        with TemporaryDirectory(prefix="cone_colour_overlap_") as temporary_dir:
            paths = [Path(temporary_dir) / name
                     for name in ("original.json", "transformed.json", "gt.json")]
            for path, data in zip(paths, cohort):
                path.write_text(json.dumps(data))
            statistics = aggregate_three_way_overlap(
                *paths, iou_threshold, require_category_match, include_sizes=True,
            )
        statistics.update(mapping, cohort="source_only", evaluation_policy=OVERLAP_POLICY)
    else:
        statistics = aggregate_three_way_overlap(
            original_path,
            transformed_path,
            gt_path,
            iou_threshold,
            require_category_match,
            include_sizes=True,
            transformed_native_path=native_transformed_path,
            reference_area_scale=reference_area_scale,
        )
    statistics.update(compute_retention_metrics(statistics))
    return statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate detection-overlap statistics without generating diagrams."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset name used by default input/output paths (default: test)",
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        dest="results_dir",
        type=Path,
        default=None,
        help="Detection results directory (default: <repo>/results/<dataset>)",
    )
    parser.add_argument(
        "--gt-path",
        "--gt_path",
        dest="gt_path",
        type=Path,
        default=None,
        help="Ground-truth annotations (default: inferred from dataset/results directory)",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        dest="output_path",
        type=Path,
        default=None,
        help=(
            "Statistics JSON path (default: "
            "<repo>/analysis/results/rq1/<dataset>/detection_overlap_metrics/"
            "detection_overlap_metrics.json)"
        ),
    )
    parser.add_argument(
        "--brute-force-config",
        "--brute_force_config",
        dest="brute_force_config",
        type=Path,
        default=DEFAULT_BRUTE_FORCE_CONFIG,
        help="Brute-force transformation configuration path",
    )
    parser.add_argument(
        "--iou-threshold",
        "--iou_threshold",
        dest="iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching (default: 0.5)",
    )
    parser.add_argument("--model", help="Calculate statistics for one model only")
    parser.add_argument(
        "--transformation", help="Calculate statistics for one transformation only"
    )
    parser.add_argument(
        "--ignore-category",
        "--ignore_category",
        dest="ignore_category",
        action="store_true",
        help=(
            "Match detections by spatial IoU only. By default, matching also "
            "requires equal category_id values."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = (args.results_dir or dataset_results_dir(args.dataset)).resolve()
    gt_path = (args.gt_path or infer_gt_path(results_dir, args.dataset)).resolve()
    output_path = (args.output_path or dataset_output_path(args.dataset)).resolve()
    brute_force_config = args.brute_force_config.resolve()

    try:
        gt_path = ensure_gt_coco(gt_path)
    except (FileNotFoundError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    models = [args.model] if args.model else discover_models(results_dir)
    if not models:
        print(f"Error: no model detections found in {results_dir}", file=sys.stderr)
        return 1

    transformations = (
        [args.transformation]
        if args.transformation
        else discover_transformations(results_dir, models)
    )
    if args.transformation == "imgColorWheel":
        transformations = [name for name in discover_transformations(results_dir, models)
                           if name.startswith("imgColorWheel_")]
    if not transformations:
        print(f"Error: no transformed detections found in {results_dir}", file=sys.stderr)
        return 1

    print("=" * 80)
    print("DETECTION OVERLAP METRICS")
    print("=" * 80)
    print(f"Models: {', '.join(models)}")
    print(f"Transformations: {len(transformations)} total")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Category-aware matching: {not args.ignore_category}")
    print(f"Output: {output_path}")
    print("=" * 80)

    all_statistics = {}
    success_count = 0
    skip_count = 0

    for transformation in transformations:
        for model in models:
            try:
                statistics = calculate_statistics_for_model_transformation(
                    model,
                    transformation,
                    results_dir,
                    gt_path,
                    args.iou_threshold,
                    brute_force_config,
                    not args.ignore_category,
                )
                all_statistics[f"{model}_{transformation}"] = statistics
                success_count += 1
            except FileNotFoundError as error:
                print(f"⚠ Skipping {model}/{transformation}: {error}")
                skip_count += 1
            except Exception as error:
                print(f"✗ Error processing {model}/{transformation}: {error}")
                skip_count += 1

    if not all_statistics:
        print("Error: no statistics could be calculated; output was not written", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if (args.model or args.transformation) and output_path.exists():
        previous = json.loads(output_path.read_text())
        previous.update(all_statistics)
        all_statistics = previous
    with open(output_path, "w") as file:
        json.dump(all_statistics, file, indent=2)
        file.write("\n")

    print("=" * 80)
    print(f"✓ Calculated {success_count} model/transformation pair(s)")
    if skip_count:
        print(f"⚠ Skipped {skip_count} model/transformation pair(s)")
    print(f"✓ Saved statistics to: {output_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
