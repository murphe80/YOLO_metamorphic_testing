#!/usr/bin/env python3
"""
RQ2.a: Compare Brute Force vs Golden-Section Results

Loads results from both search methods and generates comparison report.

Usage:
    python scripts/rq2a_compare_results.py \
        --brute-force analysis/results/rq2a/rq2a_brute_force_yolov5/results.json \
        --golden-section analysis/results/rq2a/rq2a_golden_section_yolov5/results.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_results(filepath):
    """Load results JSON file."""
    with open(filepath) as f:
        return json.load(f)


def compare_results(bf_results, gs_results):
    """
    Compare brute force vs golden-section results.

    Returns comparison statistics and per-image differences.
    """
    # Match images by name
    bf_by_name = {r['image_name']: r for r in bf_results['results']}
    gs_by_name = {r['image_name']: r for r in gs_results['results']}

    common_images = set(bf_by_name.keys()) & set(gs_by_name.keys())

    if not common_images:
        print("⚠ Warning: No common images found between results!")
        return None

    comparisons = []

    for img_name in sorted(common_images):
        bf = bf_by_name[img_name]
        gs = gs_by_name[img_name]

        # Calculate differences
        gamma_left_diff = abs(bf['gamma_left'] - gs['gamma_left']) if bf['gamma_left'] and gs['gamma_left'] else None
        gamma_right_diff = abs(bf['gamma_right'] - gs['gamma_right']) if bf['gamma_right'] and gs['gamma_right'] else None
        peak_gamma_diff = abs(bf['peak_gamma'] - gs['peak_gamma']) if bf['peak_gamma'] and gs['peak_gamma'] else None

        iteration_reduction = 1 - (gs['iterations'] / bf['iterations']) if bf['iterations'] > 0 else 0
        speedup = bf['time_seconds'] / gs['time_seconds'] if gs['time_seconds'] > 0 else 0

        comparisons.append({
            'image_name': img_name,
            'bf_iterations': bf['iterations'],
            'gs_iterations': gs['iterations'],
            'iteration_reduction': iteration_reduction,
            'bf_time': bf['time_seconds'],
            'gs_time': gs['time_seconds'],
            'speedup': speedup,
            'gamma_left_bf': bf['gamma_left'],
            'gamma_left_gs': gs['gamma_left'],
            'gamma_left_diff': gamma_left_diff,
            'gamma_right_bf': bf['gamma_right'],
            'gamma_right_gs': gs['gamma_right'],
            'gamma_right_diff': gamma_right_diff,
            'peak_gamma_bf': bf['peak_gamma'],
            'peak_gamma_gs': gs['peak_gamma'],
            'peak_gamma_diff': peak_gamma_diff,
        })

    # Calculate summary statistics
    summary = {
        'num_images': len(comparisons),
        'avg_bf_iterations': np.mean([c['bf_iterations'] for c in comparisons]),
        'avg_gs_iterations': np.mean([c['gs_iterations'] for c in comparisons]),
        'avg_iteration_reduction': np.mean([c['iteration_reduction'] for c in comparisons]),
        'avg_bf_time': np.mean([c['bf_time'] for c in comparisons]),
        'avg_gs_time': np.mean([c['gs_time'] for c in comparisons]),
        'avg_speedup': np.mean([c['speedup'] for c in comparisons]),
        'avg_gamma_left_diff': np.mean([c['gamma_left_diff'] for c in comparisons if c['gamma_left_diff'] is not None]),
        'avg_gamma_right_diff': np.mean([c['gamma_right_diff'] for c in comparisons if c['gamma_right_diff'] is not None]),
        'avg_peak_gamma_diff': np.mean([c['peak_gamma_diff'] for c in comparisons if c['peak_gamma_diff'] is not None]),
        'max_gamma_left_diff': max([c['gamma_left_diff'] for c in comparisons if c['gamma_left_diff'] is not None], default=0),
        'max_gamma_right_diff': max([c['gamma_right_diff'] for c in comparisons if c['gamma_right_diff'] is not None], default=0),
    }

    return comparisons, summary


def print_comparison_report(bf_results, gs_results, comparisons, summary):
    """Print formatted comparison report."""

    print("\n" + "="*80)
    print("RQ2.a: Brute Force vs Golden-Section Comparison")
    print("="*80)

    print(f"\nDataset Configuration:")
    print(f"  Model: {bf_results['model']}")
    print(f"  Dataset: {bf_results['dataset']}")
    print(f"  Target AP: {bf_results['target_ap']:.2f}")
    print(f"  Images compared: {summary['num_images']}")

    print(f"\n{'='*80}")
    print("Efficiency Comparison")
    print(f"{'='*80}")

    print(f"\nIterations per Image:")
    print(f"  Brute Force:    {summary['avg_bf_iterations']:.1f} iterations")
    print(f"  Golden-Section: {summary['avg_gs_iterations']:.1f} iterations")
    print(f"  Reduction:      {summary['avg_iteration_reduction']*100:.1f}%")

    print(f"\nTime per Image:")
    print(f"  Brute Force:    {summary['avg_bf_time']:.1f}s")
    print(f"  Golden-Section: {summary['avg_gs_time']:.1f}s")
    print(f"  Speedup:        {summary['avg_speedup']:.2f}x")

    print(f"\n{'='*80}")
    print("Accuracy Comparison")
    print(f"{'='*80}")

    print(f"\nGamma Differences (|BF - GS|):")
    print(f"  Left crossing:  {summary['avg_gamma_left_diff']:.3f} (avg), {summary['max_gamma_left_diff']:.3f} (max)")
    print(f"  Right crossing: {summary['avg_gamma_right_diff']:.3f} (avg), {summary['max_gamma_right_diff']:.3f} (max)")
    print(f"  Peak gamma:     {summary['avg_peak_gamma_diff']:.3f} (avg)")

    print(f"\n{'='*80}")
    print("Per-Image Results")
    print(f"{'='*80}")
    print(f"\n{'Image':<25} {'BF Iter':<10} {'GS Iter':<10} {'Speedup':<10} {'γ_left Δ':<12} {'γ_right Δ':<12}")
    print("-"*80)

    for c in comparisons[:20]:  # Show first 20 images
        gamma_left_str = f"{c['gamma_left_diff']:.3f}" if c['gamma_left_diff'] is not None else "N/A"
        gamma_right_str = f"{c['gamma_right_diff']:.3f}" if c['gamma_right_diff'] is not None else "N/A"
        print(f"{c['image_name']:<25} {c['bf_iterations']:<10} {c['gs_iterations']:<10} {c['speedup']:<10.2f} {gamma_left_str:<12} {gamma_right_str:<12}")

    if len(comparisons) > 20:
        print(f"... ({len(comparisons) - 20} more images)")

    print(f"\n{'='*80}")
    print("Conclusion")
    print(f"{'='*80}")

    print(f"\nGolden-section search is {summary['avg_speedup']:.2f}x faster with {summary['avg_iteration_reduction']*100:.0f}% fewer iterations.")
    print(f"Gamma accuracy: within ±{summary['avg_gamma_left_diff']:.3f} (left) and ±{summary['avg_gamma_right_diff']:.3f} (right) on average.")

    # Interpretation
    if summary['avg_gamma_left_diff'] < 0.1 and summary['avg_gamma_right_diff'] < 0.1:
        print("\n✓ Excellent agreement: Both methods find nearly identical thresholds.")
    elif summary['avg_gamma_left_diff'] < 0.2 and summary['avg_gamma_right_diff'] < 0.2:
        print("\n✓ Good agreement: Differences are small and acceptable for operational use.")
    else:
        print("\n⚠ Moderate differences: Golden-section may sacrifice some accuracy for speed.")

    print(f"\n{'='*80}\n")


def plot_comparison(comparisons, summary, output_dir, model_name):
    """Generate comparison plots."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Iterations comparison
    ax = axes[0, 0]
    bf_iters = [c['bf_iterations'] for c in comparisons]
    gs_iters = [c['gs_iterations'] for c in comparisons]
    ax.scatter(bf_iters, gs_iters, alpha=0.6, s=50)
    max_iter = max(max(bf_iters), max(gs_iters))
    ax.plot([0, max_iter], [0, max_iter], 'r--', label='Equal')
    ax.set_xlabel('Brute Force Iterations')
    ax.set_ylabel('Golden-Section Iterations')
    ax.set_title(f'{model_name} Iteration Comparison\nAvg Reduction: {summary["avg_iteration_reduction"]*100:.1f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Speedup distribution
    ax = axes[0, 1]
    speedups = [c['speedup'] for c in comparisons]
    ax.hist(speedups, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(summary['avg_speedup'], color='red', linestyle='--', label=f'Mean: {summary["avg_speedup"]:.2f}x')
    ax.set_xlabel('Speedup Factor')
    ax.set_ylabel('Frequency')
    ax.set_title('Speedup Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Time comparison
    ax = axes[0, 2]
    bf_times = [c['bf_time'] for c in comparisons]
    gs_times = [c['gs_time'] for c in comparisons]
    ax.scatter(bf_times, gs_times, alpha=0.6, s=50, color='orange')
    max_time = max(max(bf_times), max(gs_times))
    ax.plot([0, max_time], [0, max_time], 'r--', label='Equal')
    ax.set_xlabel('Brute Force Time (s)')
    ax.set_ylabel('Golden-Section Time (s)')
    ax.set_title(f'Time Comparison\nAvg Speedup: {summary["avg_speedup"]:.2f}x')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Gamma left difference
    ax = axes[1, 0]
    left_diffs = [c['gamma_left_diff'] for c in comparisons if c['gamma_left_diff'] is not None]
    ax.hist(left_diffs, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(summary['avg_gamma_left_diff'], color='red', linestyle='--',
               label=f'Mean: {summary["avg_gamma_left_diff"]:.3f}')
    ax.set_xlabel('|γ_left(BF) - γ_left(GS)|')
    ax.set_ylabel('Frequency')
    ax.set_title('Left Crossing Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Gamma right difference
    ax = axes[1, 1]
    right_diffs = [c['gamma_right_diff'] for c in comparisons if c['gamma_right_diff'] is not None]
    ax.hist(right_diffs, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(summary['avg_gamma_right_diff'], color='red', linestyle='--',
               label=f'Mean: {summary["avg_gamma_right_diff"]:.3f}')
    ax.set_xlabel('|γ_right(BF) - γ_right(GS)|')
    ax.set_ylabel('Frequency')
    ax.set_title('Right Crossing Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Peak gamma comparison
    ax = axes[1, 2]
    bf_peaks = [c['peak_gamma_bf'] for c in comparisons if c['peak_gamma_bf'] is not None]
    gs_peaks = [c['peak_gamma_gs'] for c in comparisons if c['peak_gamma_gs'] is not None]
    ax.scatter(bf_peaks, gs_peaks, alpha=0.6, s=50, color='red')
    min_peak = min(min(bf_peaks), min(gs_peaks))
    max_peak = max(max(bf_peaks), max(gs_peaks))
    ax.plot([min_peak, max_peak], [min_peak, max_peak], 'k--', label='Equal')
    ax.set_xlabel('Brute Force Peak γ')
    ax.set_ylabel('Golden-Section Peak γ')
    ax.set_title(f'Peak Gamma Comparison\nAvg Diff: {summary["avg_peak_gamma_diff"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / model_name / 'rq2a_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Brute Force vs Golden-Section search results"
    )

    parser.add_argument(
        "--brute-force",
        type=str,
        required=True,
        help="Path to brute force results.json"
    )
    parser.add_argument(
        "--golden-section",
        type=str,
        required=True,
        help="Path to golden-section results.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/results/rq2a/rq2a_comparison",
        help="Output directory for comparison report and plots (default: analysis/results/rq2a_comparison)"
    )

    args = parser.parse_args()

    # Load results
    print("Loading results...")
    bf_results = load_results(args.brute_force)
    gs_results = load_results(args.golden_section)

    # Compare
    print("Comparing methods...")
    result = compare_results(bf_results, gs_results)

    if result is None:
        print("Error: Could not compare results (no common images)")
        return

    comparisons, summary = result

    # Save comparison JSON
    output_json = Path(args.output_dir) / bf_results['model'] / 'rq2a_comparison.json'
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Print report
    print_comparison_report(bf_results, gs_results, comparisons, summary)

    comparison_data = {
        'brute_force_file': args.brute_force,
        'golden_section_file': args.golden_section,
        'summary': summary,
        'per_image_comparisons': comparisons,
    }

    with open(output_json, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    print(f"Comparison data saved to: {output_json}")

    # Generate plots
    print("\nGenerating plots...")
    plot_comparison(comparisons, summary, args.output_dir, bf_results['model'])


if __name__ == "__main__":
    main()
