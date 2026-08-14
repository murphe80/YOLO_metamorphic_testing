#!/usr/bin/env python3
"""
Generate Venn diagram visualizations showing detection overlap between
original images, transformed images, and ground truth annotations.

Usage:
    # Generate one model-comparison grid per transformation
    python3 analysis/visualisation/plot_detection_venn.py

    # Custom IoU threshold
    python3 analysis/visualisation/plot_detection_venn.py --iou_threshold 0.7

    # Specific model and transformation (a one-panel grid)
    python3 analysis/visualisation/plot_detection_venn.py --model yolov8 --transformation imgGrayscale

    # Custom paths
    python3 analysis/visualisation/plot_detection_venn.py --results_dir results --gt_path data/base_images/fsoco_sample/annotations_coco.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.detection_matching import aggregate_three_way_overlap, compute_retention_metrics
from utils.format_detections import format_detections
from utils.format_gt import convert_to_coco_format
from utils.transform_geometric_detections import transform_detections_file


# Model color scheme (from dashboard.html)
MODEL_COLORS = {
    'yolov5': '#1976d2',  # Blue
    'yolov8': '#7b1fa2',  # Purple
    'yolov12': '#388e3c',  # Green
    'yolov26': '#ff9800',  # Orange
    'yolov26_v2': '#d32f2f',  # Red
}
MODEL_ORDER = ['yolov5', 'yolov8', 'yolov12', 'yolov26', 'yolov26_v2']

BASE_TRANSFORMATIONS = [
    'imgStretch', 'imgResolution', 'imgMotionBlur',
    'imgDarkBright', 'imgGrayscale', 'imgColorWheel'
]
GEOMETRIC_TRANSFORMS = {'imgStretch', 'imgResolution'}
DEFAULT_BRUTE_FORCE_CONFIG = Path('config/brute_force_transformations.json')


def infer_gt_path(results_dir: Path) -> Path:
    """Infer the dataset GT path from a results directory."""
    if results_dir.name and results_dir.name != 'results':
        candidate = Path('data/base_images') / results_dir.name / 'annotations_coco.json'
        if candidate.parent.exists():
            return candidate
    return Path('data/base_images/test/annotations_coco.json')


def ensure_gt_coco(gt_path: Path) -> Path:
    """Create a missing COCO GT file from sibling bounding-box JSON or YOLO labels."""
    if gt_path.exists():
        return gt_path

    dataset_dir = gt_path.parent
    bounding_boxes_dir = dataset_dir / 'bounding_boxes'
    labels_dir = dataset_dir / 'labels'
    images_dir = dataset_dir / 'images'

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
        f"Ground truth not found: {gt_path}. Expected either {bounding_boxes_dir} or {labels_dir} to generate it."
    )


def resolve_base_transformation(results_dir: Path, model: str, transformation: str) -> str:
    """Resolve brute-force variant directories back to their base transformation."""
    transform_dir = results_dir / model / transformation

    for config_name in ('configs.json', 'config.json'):
        config_path = transform_dir / config_name
        if not config_path.exists():
            continue
        try:
            with open(config_path) as f:
                config = json.load(f)
        except json.JSONDecodeError:
            continue

        if isinstance(config, dict):
            if isinstance(config.get('transformation'), str):
                return config['transformation']
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
    brute_force_config: Path
) -> Optional[dict]:
    """Resolve variant-specific geometric transform params, mirroring the analysis bash script."""
    transform_dir = results_dir / model / transformation

    for config_name in ('configs.json', 'config.json'):
        config_path = transform_dir / config_name
        if not config_path.exists():
            continue
        try:
            with open(config_path) as f:
                config = json.load(f)
        except json.JSONDecodeError:
            continue

        if not isinstance(config, dict):
            continue

        nested_config = config.get(base_transformation)
        if isinstance(nested_config, dict):
            return nested_config

        if any(key in config for key in ('scale_x', 'scale_y', 'width_ratio', 'height_ratio', 'size_ratio')):
            return config

    if not brute_force_config.exists():
        return None

    try:
        with open(brute_force_config) as f:
            config = json.load(f)
    except json.JSONDecodeError:
        return None

    variants = config.get(base_transformation, [])
    if isinstance(variants, dict):
        variants = [variants]

    for variant in variants:
        if isinstance(variant, dict) and variant.get('variant_id') == transformation:
            return variant

    return None


def discover_models(results_dir: Path) -> list:
    """Discover model directories that contain original detections."""
    if not results_dir.exists():
        return []

    models = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
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
    """Discover legacy rule and brute-force variant result directories."""
    transformations = set()

    for model in models:
        model_dir = results_dir / model
        if not model_dir.exists():
            continue

        for transform_dir in model_dir.iterdir():
            if not transform_dir.is_dir() or transform_dir.name.startswith('.'):
                continue
            if transform_dir.name == 'original':
                continue
            if (transform_dir / "detections_coco.json").exists():
                transformations.add(transform_dir.name)

    return sorted(transformations)


def compute_venn_for_model_transformation(
    model: str,
    transformation: str,
    results_dir: Path,
    gt_path: Path,
    iou_threshold: float = 0.5,
    brute_force_config: Path = DEFAULT_BRUTE_FORCE_CONFIG
) -> dict:
    """
    Compute three-way overlap statistics for one model-transformation pair.

    Args:
        model: Model name (yolov5, yolov8, yolov12, yolov26)
        transformation: Transformation name (e.g., imgGrayscale)
        results_dir: Base results directory
        gt_path: Path to ground truth COCO JSON
        iou_threshold: IoU threshold for matching (default: 0.5)

    Returns:
        Dictionary with image-normalized region weights and computed metrics
    """
    # Construct paths
    original_path = results_dir / model / "original" / "detections_original_coco.json"
    transformed_path = results_dir / model / transformation / "detections_coco.json"

    # Check if files exist
    if not original_path.exists():
        raise FileNotFoundError(f"Original detections not found: {original_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    # Handle geometric transformations (need coordinate transformation)
    base_transformation = resolve_base_transformation(results_dir, model, transformation)

    if not transformed_path.exists():
        if base_transformation in GEOMETRIC_TRANSFORMS:
            print(f"  Generating COCO detections for {transformation}...")
            format_detections(
                detection_dir=str(results_dir / model / transformation),
                output_file=str(transformed_path)
            )
        else:
            raise FileNotFoundError(f"Transformed detections not found: {transformed_path}")

    if base_transformation in GEOMETRIC_TRANSFORMS:
        # Try to use pre-computed transformed coordinates first
        pre_transformed_path = results_dir / model / transformation / "detections_coco_transformed.json"

        if pre_transformed_path.exists():
            print(f"  Using pre-computed transformed coordinates from: {pre_transformed_path.name}")
            transformed_path = pre_transformed_path
        else:
            # Compute coordinate transformation on the fly
            print(f"  Computing coordinate transformation for {transformation}...")
            original_image_dir = results_dir / model / "original"
            transformed_image_dir = results_dir / model / transformation
            transform_params = resolve_transform_params(
                results_dir,
                model,
                transformation,
                base_transformation,
                brute_force_config
            )
            if transform_params:
                print(f"  Using transform parameters for {transformation}: {transform_params}")

            try:
                transform_detections_file(
                    detection_file=str(transformed_path),
                    original_image_dir=str(original_image_dir),
                    transformed_image_dir=str(transformed_image_dir),
                    transformation_name=base_transformation,
                    output_file=str(pre_transformed_path),
                    transform_params=transform_params
                )
                transformed_path = pre_transformed_path
                print(f"  ✓ Coordinate transformation complete")
            except Exception as e:
                raise RuntimeError(f"Failed to transform coordinates for {transformation}: {e}")

    # Get three-way overlap statistics
    stats = aggregate_three_way_overlap(
        original_path, transformed_path, gt_path, iou_threshold
    )

    # Compute derived metrics
    metrics = compute_retention_metrics(stats)
    stats.update(metrics)

    return stats


def draw_venn_panel(ax, model: str, stats: dict) -> None:
    """Draw one model's Venn diagram and metrics on an existing grid axis."""
    # venn3 expects: (Abc, aBc, ABc, abC, AbC, aBC, ABC)
    # where A=Original, B=Transformed, C=GT
    # lowercase=absent, uppercase=present
    venn = venn3(
        subsets=(
            stats['orig_only'],      # Abc - Original only
            stats['trans_only'],     # aBc - Transformed only
            stats['orig_trans'],     # ABc - Original + Transformed
            stats['gt_only'],        # abC - GT only
            stats['orig_gt'],        # AbC - Original + GT
            stats['trans_gt'],       # aBC - Transformed + GT
            stats['orig_trans_gt']   # ABC - All three
        ),
        set_labels=('Original', 'Transformed', 'Ground Truth'),
        ax=ax
    )

    # Style the diagram
    if venn:
        # Color the Original circle with model-specific color
        for subset_id in ['100', '110', '101', '111']:
            patch = venn.get_patch_by_id(subset_id)
            if patch:
                patch.set_color(MODEL_COLORS.get(model, '#999999'))
                patch.set_alpha(0.5)

        # Add circle outlines for clarity
        venn3_circles(
            subsets=(
                stats['orig_only'], stats['trans_only'], stats['orig_trans'],
                stats['gt_only'], stats['orig_gt'], stats['trans_gt'],
                stats['orig_trans_gt']
            ),
            ax=ax,
            linewidth=1.5,
            linestyle='solid'
        )

    ax.set_title(model.upper(), fontsize=13, fontweight='bold', pad=12)

    # Report all derived metrics below the Venn diagram
    metric_lines = [
        f"TP Retention: {stats['tp_retention_rate']:.1%}",
        f"New TP Rate: {stats['new_tp_rate']:.1%}",
        f"FP Consistency: {stats['fp_consistency_rate']:.1%}",
        f"Detection Loss: {stats['detection_loss_rate']:.1%}",
        f"New Hallucination: {stats['new_hallucination_rate']:.1%}",
    ]
    ax.text(
        0.5,
        -0.10,
        "\n".join((
            " | ".join(metric_lines[:2]),
            " | ".join(metric_lines[2:4]),
            metric_lines[4],
        )),
        ha='center',
        va='top',
        fontsize=9,
        transform=ax.transAxes,
    )


def plot_venn_grid_for_transformation(
    models: list,
    transformation: str,
    results_dir: Path,
    gt_path: Path,
    output_dir: Path,
    iou_threshold: float = 0.5,
    brute_force_config: Path = DEFAULT_BRUTE_FORCE_CONFIG,
    columns: int = 3,
) -> tuple:
    """Generate one PNG containing a grid of model Venn diagrams."""
    columns = max(1, min(columns, len(models)))
    rows = math.ceil(len(models) / columns)
    fig, _ = plt.subplots(
        rows,
        columns,
        figsize=(6 * columns, 5.5 * rows),
        squeeze=False,
    )
    axes = fig.axes
    model_stats = {}
    success_count = 0
    skip_count = 0

    for ax, model in zip(axes, models):
        try:
            stats = compute_venn_for_model_transformation(
                model,
                transformation,
                results_dir,
                gt_path,
                iou_threshold,
                brute_force_config,
            )
            draw_venn_panel(ax, model, stats)
            model_stats[model] = stats
            success_count += 1
        except FileNotFoundError as exc:
            print(f"⚠ Skipping {model}/{transformation}: {exc}")
            ax.set_title(model.upper(), fontsize=13, fontweight='bold')
            ax.text(0.5, 0.5, "Data unavailable", ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            skip_count += 1
        except Exception as exc:
            print(f"✗ Error processing {model}/{transformation}: {exc}")
            ax.set_title(model.upper(), fontsize=13, fontweight='bold')
            ax.text(0.5, 0.5, "Could not generate diagram", ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            skip_count += 1

    for ax in axes[len(models):]:
        ax.set_axis_off()

    if success_count == 0:
        plt.close(fig)
        return model_stats, success_count, skip_count

    fig.suptitle(
        f"{transformation}\nDetection Overlap (IoU ≥ {iou_threshold})",
        fontsize=16,
        fontweight='bold',
    )
    fig.subplots_adjust(top=0.88, bottom=0.09, hspace=0.48, wspace=0.15)

    output_path = output_dir / f"{transformation}_venn.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Generated grid: {output_path.name} ({success_count} models)")

    return model_stats, success_count, skip_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate Venn diagrams for detection overlap analysis"
    )
    parser.add_argument(
        '--results_dir',
        type=Path,
        default=Path('results'),
        help='Base results directory (default: results)'
    )
    parser.add_argument(
        '--gt_path',
        type=Path,
        default=None,
        help='Path to ground truth annotations. If omitted, inferred from --results_dir.'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('analysis/results/venn_diagrams'),
        help='Output directory for Venn diagrams'
    )
    parser.add_argument(
        '--brute_force_config',
        type=Path,
        default=DEFAULT_BRUTE_FORCE_CONFIG,
        help='Path to brute-force transformation config (default: config/brute_force_transformations.json)'
    )
    parser.add_argument(
        '--iou_threshold',
        type=float,
        default=0.5,
        help='IoU threshold for matching (default: 0.5)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['yolov5', 'yolov8', 'yolov12', 'yolov26', 'yolov26_v2'],
        help='Generate for specific model only'
    )
    parser.add_argument(
        '--transformation',
        type=str,
        help='Generate for specific transformation only'
    )
    parser.add_argument(
        '--columns',
        type=int,
        default=3,
        help='Maximum number of model diagrams per grid row (default: 3)'
    )

    args = parser.parse_args()

    if args.gt_path is None:
        args.gt_path = infer_gt_path(args.results_dir)

    args.gt_path = ensure_gt_coco(args.gt_path)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define models and transformations
    models = [args.model] if args.model else discover_models(args.results_dir)
    if not models:
        models = MODEL_ORDER

    transformations = [args.transformation] if args.transformation else discover_transformations(
        args.results_dir,
        models
    )
    if not transformations:
        transformations = [
            'imgGrayscale', 'imgColorWheel', 'imgDarkBright',
            'imgMotionBlur', 'imgResolution', 'imgStretch'
        ]

    print("=" * 80)
    print("VENN DIAGRAM GENERATION")
    print("=" * 80)
    print(f"Models: {', '.join(models)}")
    print(f"Transformations: {len(transformations)} total")
    print(f"IoU Threshold: {args.iou_threshold}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)

    # Generate one model-comparison grid per transformation
    all_stats = {}
    success_count = 0
    skip_count = 0
    generated_grid_count = 0

    for transformation in transformations:
        transformation_stats, successes, skips = plot_venn_grid_for_transformation(
            models,
            transformation,
            args.results_dir,
            args.gt_path,
            args.output_dir,
            args.iou_threshold,
            args.brute_force_config,
            args.columns,
        )
        for model, stats in transformation_stats.items():
            all_stats[f"{model}_{transformation}"] = stats
        success_count += successes
        skip_count += skips
        if successes:
            generated_grid_count += 1

    # Save aggregate statistics
    stats_path = args.output_dir / "venn_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print("=" * 80)
    print(f"✓ Successfully generated {generated_grid_count} Venn diagram grids ({success_count} model panels)")
    if skip_count > 0:
        print(f"⚠ Skipped {skip_count} model-transformation pairs")
    print(f"✓ Saved statistics to: {stats_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
