#!/usr/bin/env python3
"""
Generate a comprehensive grid visualization showing all model-transformation
Venn diagrams in a single figure for easy comparison.

Usage:
    # Generate grid with default settings
    python3 analysis/visualisation/plot_venn_grid.py

    # Custom layout
    python3 analysis/visualisation/plot_venn_grid.py --layout row  # Models as rows
    python3 analysis/visualisation/plot_venn_grid.py --layout col  # Models as columns
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.detection_matching import aggregate_three_way_overlap


MODEL_COLORS = {
    'yolov5': '#1976d2',
    'yolov8': '#7b1fa2',
    'yolov12': '#388e3c'
}


def plot_grid(
    results_dir: Path,
    gt_path: Path,
    output_path: Path,
    iou_threshold: float = 0.5,
    layout: str = 'row'
):
    """
    Generate a grid of Venn diagrams for all model-transformation combinations.

    Args:
        results_dir: Base results directory
        gt_path: Path to ground truth annotations
        output_path: Output PNG file path
        iou_threshold: IoU threshold for matching
        layout: 'row' (models as rows) or 'col' (models as columns)
    """
    models = ['yolov5', 'yolov8', 'yolov12']
    transformations = [
        'imgGrayscale', 'imgColorWheel', 'imgDarkBright',
        'imgMotionBlur', 'imgResolution', 'imgStretch',
        'labelRemove_black', 'labelRemove_white', 'labelRemove_avg'
    ]

    # Determine grid layout
    if layout == 'row':
        nrows, ncols = len(models), len(transformations)
        figsize = (28, 12)
    else:  # layout == 'col'
        nrows, ncols = len(transformations), len(models)
        figsize = (12, 32)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    print("Generating grid visualization...")
    print(f"Layout: {nrows} rows × {ncols} cols")

    for i, model in enumerate(models):
        for j, transformation in enumerate(transformations):
            # Determine subplot position
            if layout == 'row':
                ax = axes[i, j]
                row_label = model.upper()
                col_label = transformation
            else:
                ax = axes[j, i]
                row_label = transformation
                col_label = model.upper()

            # Load data
            original_path = results_dir / model / "original" / "detections_original_coco.json"
            transformed_path = results_dir / model / transformation / "detections_coco.json"

            try:
                stats = aggregate_three_way_overlap(
                    original_path, transformed_path, gt_path, iou_threshold
                )

                # Create mini Venn diagram
                venn = venn3(
                    subsets=(
                        stats['orig_only'], stats['trans_only'], stats['orig_trans'],
                        stats['gt_only'], stats['orig_gt'], stats['trans_gt'],
                        stats['orig_trans_gt']
                    ),
                    set_labels=None,  # No labels in grid (saves space)
                    ax=ax
                )

                # Color based on model
                if venn:
                    for subset_id in ['100', '110', '101', '111']:
                        patch = venn.get_patch_by_id(subset_id)
                        if patch:
                            patch.set_color(MODEL_COLORS[model])
                            patch.set_alpha(0.4)

                    # Make text smaller
                    for text in venn.set_labels + venn.subset_labels:
                        if text:
                            text.set_fontsize(6)

                # Add labels on edges
                if layout == 'row':
                    if j == 0:
                        ax.set_ylabel(row_label, fontsize=10, fontweight='bold')
                    if i == 0:
                        ax.set_title(col_label, fontsize=8, pad=5)
                else:
                    if i == 0:
                        ax.set_ylabel(row_label, fontsize=10, fontweight='bold')
                    if j == 0:
                        ax.set_title(col_label, fontsize=8, pad=5)

            except FileNotFoundError:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                        fontsize=10, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            ax.axis('off')

    # Add main title and legend
    fig.suptitle(
        f"Detection Overlap: Original vs Transformed vs Ground Truth (IoU ≥ {iou_threshold})",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLORS['yolov5'], alpha=0.4, label='YOLOv5'),
        plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLORS['yolov8'], alpha=0.4, label='YOLOv8'),
        plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLORS['yolov12'], alpha=0.4, label='YOLOv12')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"✓ Grid visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate grid visualization of all Venn diagrams"
    )
    parser.add_argument(
        '--results_dir',
        type=Path,
        default=Path('results'),
        help='Base results directory'
    )
    parser.add_argument(
        '--gt_path',
        type=Path,
        default=Path('data/base_images/fsoco_sample/annotations_coco.json'),
        help='Path to ground truth annotations'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('analysis/results/venn_diagrams/all_models_grid.png'),
        help='Output PNG file path'
    )
    parser.add_argument(
        '--iou_threshold',
        type=float,
        default=0.5,
        help='IoU threshold for matching'
    )
    parser.add_argument(
        '--layout',
        type=str,
        choices=['row', 'col'],
        default='row',
        help='Grid layout: row (models as rows) or col (models as columns)'
    )

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GRID VISUALIZATION GENERATION")
    print("=" * 80)

    plot_grid(
        args.results_dir,
        args.gt_path,
        args.output,
        args.iou_threshold,
        args.layout
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
