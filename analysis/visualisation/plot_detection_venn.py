#!/usr/bin/env python3
"""
Generate Venn diagram visualizations showing detection overlap between
original images, transformed images, and ground truth annotations.

Usage:
    # Generate all Venn diagrams
    python3 analysis/visualisation/plot_detection_venn.py

    # Custom IoU threshold
    python3 analysis/visualisation/plot_detection_venn.py --iou_threshold 0.7

    # Specific model and transformation
    python3 analysis/visualisation/plot_detection_venn.py --model yolov8 --transformation imgGrayscale

    # Custom paths
    python3 analysis/visualisation/plot_detection_venn.py --results_dir results --gt_path data/base_images/fsoco_sample/annotations_coco.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.detection_matching import aggregate_three_way_overlap, compute_retention_metrics


# Model color scheme (from dashboard.html)
MODEL_COLORS = {
    'yolov5': '#1976d2',  # Blue
    'yolov8': '#7b1fa2',  # Purple
    'yolov12': '#388e3c'  # Green
}


def plot_venn_for_model_transformation(
    model: str,
    transformation: str,
    results_dir: Path,
    gt_path: Path,
    output_dir: Path,
    iou_threshold: float = 0.5
) -> dict:
    """
    Generate a three-circle Venn diagram for one model-transformation pair.

    Args:
        model: Model name (yolov5, yolov8, yolov12)
        transformation: Transformation name (e.g., imgGrayscale)
        results_dir: Base results directory
        gt_path: Path to ground truth COCO JSON
        output_dir: Output directory for PNG files
        iou_threshold: IoU threshold for matching (default: 0.5)

    Returns:
        Dictionary with region counts and computed metrics
    """
    # Construct paths
    original_path = results_dir / model / "original" / "detections_original_coco.json"
    transformed_path = results_dir / model / transformation / "detections_coco.json"

    # Check if files exist
    if not original_path.exists():
        raise FileNotFoundError(f"Original detections not found: {original_path}")
    if not transformed_path.exists():
        raise FileNotFoundError(f"Transformed detections not found: {transformed_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    # Get three-way overlap statistics
    stats = aggregate_three_way_overlap(
        original_path, transformed_path, gt_path, iou_threshold
    )

    # Compute derived metrics
    metrics = compute_retention_metrics(stats)
    stats.update(metrics)

    # Create three-circle Venn diagram
    # venn3 expects: (Abc, aBc, ABc, abC, AbC, aBC, ABC)
    # where A=Original, B=Transformed, C=GT
    # lowercase=absent, uppercase=present
    fig, ax = plt.subplots(figsize=(10, 8))

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

    # Add title with key metrics
    title = (
        f"{model.upper()} - {transformation}\n"
        f"Detection Overlap (IoU ≥ {iou_threshold})\n"
        f"TP Retention: {stats['tp_retention_rate']:.1%} | "
        f"Detection Loss: {stats['detection_loss_rate']:.1%}"
    )
    plt.title(title, fontsize=12, fontweight='bold', pad=20)

    # Save figure
    output_path = output_dir / f"{model}_{transformation}_venn.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"✓ Generated: {output_path.name}")

    return stats


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
        default=Path('data/base_images/fsoco_sample/annotations_coco.json'),
        help='Path to ground truth annotations'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('analysis/results/venn_diagrams'),
        help='Output directory for Venn diagrams'
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
        choices=['yolov5', 'yolov8', 'yolov12'],
        help='Generate for specific model only'
    )
    parser.add_argument(
        '--transformation',
        type=str,
        help='Generate for specific transformation only'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define models and transformations
    models = [args.model] if args.model else ['yolov5', 'yolov8', 'yolov12']
    transformations = [args.transformation] if args.transformation else [
        'imgGrayscale', 'imgColorWheel', 'imgDarkBright',
        'imgMotionBlur', 'imgResolution', 'imgStretch',
        'labelRemove_black', 'labelRemove_white', 'labelRemove_avg'
    ]

    print("=" * 80)
    print("VENN DIAGRAM GENERATION")
    print("=" * 80)
    print(f"Models: {', '.join(models)}")
    print(f"Transformations: {len(transformations)} total")
    print(f"IoU Threshold: {args.iou_threshold}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)

    # Generate Venn diagrams
    all_stats = {}
    success_count = 0
    skip_count = 0

    for model in models:
        for transformation in transformations:
            try:
                stats = plot_venn_for_model_transformation(
                    model,
                    transformation,
                    args.results_dir,
                    args.gt_path,
                    args.output_dir,
                    args.iou_threshold
                )
                all_stats[f"{model}_{transformation}"] = stats
                success_count += 1

            except FileNotFoundError as e:
                print(f"⚠ Skipping {model}/{transformation}: {e.args[0].split(':')[0]}")
                skip_count += 1

            except Exception as e:
                print(f"✗ Error processing {model}/{transformation}: {e}")
                skip_count += 1

    # Save aggregate statistics
    stats_path = args.output_dir / "venn_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print("=" * 80)
    print(f"✓ Successfully generated {success_count} Venn diagrams")
    if skip_count > 0:
        print(f"⚠ Skipped {skip_count} model-transformation pairs")
    print(f"✓ Saved statistics to: {stats_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
