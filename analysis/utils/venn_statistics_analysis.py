#!/usr/bin/env python3
"""
Analyze and export Venn diagram statistics to CSV format for further analysis.

Computes retention rates, detection loss metrics, and cross-model comparisons.

Usage:
    # Generate CSV from venn_statistics.json
    python3 analysis/utils/venn_statistics_analysis.py

    # Custom input/output paths
    python3 analysis/utils/venn_statistics_analysis.py --input analysis/results/venn_diagrams/venn_statistics.json --output analysis/results/venn_statistics.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_stats_key(key: str) -> tuple:
    """
    Parse a statistics key like 'yolov5_imgGrayscale' into (model, transformation).

    Args:
        key: Statistics key in format 'model_transformation'

    Returns:
        Tuple of (model, transformation)
    """
    parts = key.split('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return key, ''


def compute_all_metrics(regions: Dict[str, int]) -> Dict[str, float]:
    """
    Compute comprehensive metrics from Venn region counts.

    Args:
        regions: Dictionary with counts for all 7 Venn regions

    Returns:
        Dictionary with computed metrics
    """
    metrics = {}

    # Total counts
    total_orig = (regions['orig_trans_gt'] + regions['orig_trans'] +
                  regions['orig_gt'] + regions['orig_only'])
    total_trans = (regions['orig_trans_gt'] + regions['orig_trans'] +
                   regions['trans_gt'] + regions['trans_only'])
    total_gt = (regions['orig_trans_gt'] + regions['orig_gt'] +
                regions['trans_gt'] + regions['gt_only'])

    metrics['total_original'] = total_orig
    metrics['total_transformed'] = total_trans
    metrics['total_ground_truth'] = total_gt

    # True positive retention rate
    # How many of the true positives in original are still detected in transformed?
    orig_tp_count = regions['orig_trans_gt'] + regions['orig_gt']
    all_gt_matched = regions['orig_trans_gt'] + regions['orig_gt'] + regions['trans_gt']

    if all_gt_matched > 0:
        metrics['tp_retention_rate'] = (regions['orig_trans_gt'] + regions['orig_gt']) / all_gt_matched
    else:
        metrics['tp_retention_rate'] = 0.0

    # Detection consistency rate
    # What proportion of detections appear in both original and transformed?
    total_detections = total_orig + total_trans
    if total_detections > 0:
        metrics['detection_consistency'] = (2 * (regions['orig_trans_gt'] + regions['orig_trans'])) / total_detections
    else:
        metrics['detection_consistency'] = 0.0

    # Detection loss rate
    # What proportion of original detections disappeared?
    if total_orig > 0:
        metrics['detection_loss_rate'] = regions['orig_only'] / total_orig
    else:
        metrics['detection_loss_rate'] = 0.0

    # Spurious detection rate
    # What proportion of transformed detections are new (not in original)?
    if total_trans > 0:
        metrics['spurious_detection_rate'] = regions['trans_only'] / total_trans
    else:
        metrics['spurious_detection_rate'] = 0.0

    # Precision-like metrics
    # Original precision: what % of original detections match GT?
    if total_orig > 0:
        metrics['original_precision'] = (regions['orig_trans_gt'] + regions['orig_gt']) / total_orig
    else:
        metrics['original_precision'] = 0.0

    # Transformed precision: what % of transformed detections match GT?
    if total_trans > 0:
        metrics['transformed_precision'] = (regions['orig_trans_gt'] + regions['trans_gt']) / total_trans
    else:
        metrics['transformed_precision'] = 0.0

    # Recall-like metrics
    # Original recall: what % of GT detections are detected in original?
    if total_gt > 0:
        metrics['original_recall'] = (regions['orig_trans_gt'] + regions['orig_gt']) / total_gt
    else:
        metrics['original_recall'] = 0.0

    # Transformed recall: what % of GT detections are detected in transformed?
    if total_gt > 0:
        metrics['transformed_recall'] = (regions['orig_trans_gt'] + regions['trans_gt']) / total_gt
    else:
        metrics['transformed_recall'] = 0.0

    # Metamorphic consistency: maintained true positives vs total true positives
    if orig_tp_count > 0:
        metrics['metamorphic_tp_consistency'] = regions['orig_trans_gt'] / orig_tp_count
    else:
        metrics['metamorphic_tp_consistency'] = 0.0

    return metrics


def export_to_csv(stats_json_path: Path, csv_output_path: Path):
    """
    Export Venn statistics to CSV format.

    Args:
        stats_json_path: Path to venn_statistics.json
        csv_output_path: Path for output CSV file
    """
    # Load JSON statistics
    with open(stats_json_path) as f:
        all_stats = json.load(f)

    # Prepare CSV rows
    rows = []

    for key, regions in all_stats.items():
        model, transformation = parse_stats_key(key)

        # Compute all metrics
        metrics = compute_all_metrics(regions)

        # Build row
        row = {
            'model': model,
            'transformation': transformation,
            # Raw region counts
            'orig_only': regions['orig_only'],
            'trans_only': regions['trans_only'],
            'gt_only': regions['gt_only'],
            'orig_trans': regions['orig_trans'],
            'orig_gt': regions['orig_gt'],
            'trans_gt': regions['trans_gt'],
            'orig_trans_gt': regions['orig_trans_gt'],
            # Totals
            'total_original': metrics['total_original'],
            'total_transformed': metrics['total_transformed'],
            'total_ground_truth': metrics['total_ground_truth'],
            # Computed metrics
            'tp_retention_rate': metrics['tp_retention_rate'],
            'detection_consistency': metrics['detection_consistency'],
            'detection_loss_rate': metrics['detection_loss_rate'],
            'spurious_detection_rate': metrics['spurious_detection_rate'],
            'original_precision': metrics['original_precision'],
            'transformed_precision': metrics['transformed_precision'],
            'original_recall': metrics['original_recall'],
            'transformed_recall': metrics['transformed_recall'],
            'metamorphic_tp_consistency': metrics['metamorphic_tp_consistency']
        }

        rows.append(row)

    # Sort by model, then transformation
    rows.sort(key=lambda x: (x['model'], x['transformation']))

    # Write CSV
    fieldnames = [
        'model', 'transformation',
        'orig_only', 'trans_only', 'gt_only',
        'orig_trans', 'orig_gt', 'trans_gt', 'orig_trans_gt',
        'total_original', 'total_transformed', 'total_ground_truth',
        'tp_retention_rate', 'detection_consistency',
        'detection_loss_rate', 'spurious_detection_rate',
        'original_precision', 'transformed_precision',
        'original_recall', 'transformed_recall',
        'metamorphic_tp_consistency'
    ]

    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Exported {len(rows)} rows to {csv_output_path}")


def print_summary(stats_json_path: Path):
    """
    Print a summary of key findings from the Venn statistics.

    Args:
        stats_json_path: Path to venn_statistics.json
    """
    with open(stats_json_path) as f:
        all_stats = json.load(f)

    print("\n" + "=" * 80)
    print("VENN DIAGRAM STATISTICS SUMMARY")
    print("=" * 80)

    # Group by model
    by_model = {}
    for key, regions in all_stats.items():
        model, transformation = parse_stats_key(key)
        if model not in by_model:
            by_model[model] = []

        metrics = compute_all_metrics(regions)
        by_model[model].append({
            'transformation': transformation,
            'metrics': metrics,
            'regions': regions
        })

    # Print model summaries
    for model in sorted(by_model.keys()):
        print(f"\n{model.upper()}")
        print("-" * 80)

        transformations = by_model[model]

        # Find best and worst transformations
        best_retention = max(transformations, key=lambda x: x['metrics']['tp_retention_rate'])
        worst_retention = min(transformations, key=lambda x: x['metrics']['tp_retention_rate'])

        print(f"  Best TP Retention: {best_retention['transformation']:<20} "
              f"({best_retention['metrics']['tp_retention_rate']:.1%})")
        print(f"  Worst TP Retention: {worst_retention['transformation']:<20} "
              f"({worst_retention['metrics']['tp_retention_rate']:.1%})")

        # Average metrics
        avg_tp_retention = sum(t['metrics']['tp_retention_rate'] for t in transformations) / len(transformations)
        avg_detection_loss = sum(t['metrics']['detection_loss_rate'] for t in transformations) / len(transformations)

        print(f"  Average TP Retention: {avg_tp_retention:.1%}")
        print(f"  Average Detection Loss: {avg_detection_loss:.1%}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and export Venn diagram statistics"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('analysis/results/venn_diagrams/venn_statistics.json'),
        help='Input JSON statistics file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('analysis/results/venn_diagrams/venn_statistics.csv'),
        help='Output CSV file'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary statistics to console'
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run plot_detection_venn.py first to generate statistics.")
        return 1

    # Export to CSV
    export_to_csv(args.input, args.output)

    # Print summary if requested
    if args.summary:
        print_summary(args.input)

    return 0


if __name__ == "__main__":
    exit(main())
