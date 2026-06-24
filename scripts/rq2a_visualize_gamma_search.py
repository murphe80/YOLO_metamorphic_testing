#!/usr/bin/env python3
"""
RQ2.a: Visualize Gamma Search Results

Plots gamma vs AP curves for brute force or golden section search results.

Usage:
    python scripts/rq2a_visualize_gamma_search.py --method brute_force --model yolov8
    python scripts/rq2a_visualize_gamma_search.py --method golden_section --model yolov8
    python scripts/rq2a_visualize_gamma_search.py --method brute_force --model yolov8 --image amz_00832.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_gamma_ap_curve(
    image_data: dict,
    target_ap: float,
    method: str,
    output_path: str = None
):
    """
    Plot gamma vs AP curve for a single image.

    Args:
        image_data: Dictionary with image results
        target_ap: Target AP threshold
        method: Method name for title
        output_path: Optional path to save figure
    """
    image_name = image_data['image_name']

    # Handle skipped images
    if image_data.get('skipped', False):
        print(f"  Skipped: {image_name} - {image_data.get('skip_reason', 'No reason provided')}")
        return

    all_results = image_data.get('all_results', [])
    if not all_results:
        print(f"  No results to plot for {image_name}")
        return

    # Extract gamma and AP values
    gammas = [r[0] for r in all_results]
    aps = [r[1] for r in all_results]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot AP curve
    ax.plot(gammas, aps, 'o-', linewidth=2, markersize=4, color='#2E86AB', alpha=0.8, label='AP')

    # Target AP line
    ax.axhline(y=target_ap, color='#A23B72', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Target AP ({target_ap:.2f})')

    # Mark peak
    peak_gamma = image_data.get('peak_gamma')
    peak_ap = image_data.get('peak_ap')
    if peak_gamma is not None and peak_ap is not None:
        ax.plot(peak_gamma, peak_ap, 'o', markersize=10, color='#F18F01',
                markeredgecolor='black', markeredgewidth=1.5, label=f'Peak (γ={peak_gamma:.2f})', zorder=5)

    # Mark crossings if found
    gamma_left = image_data.get('gamma_left')
    ap_left = image_data.get('ap_left')
    if gamma_left is not None and ap_left is not None:
        ax.plot(gamma_left, ap_left, 's', markersize=8, color='#6A994E',
                markeredgecolor='black', markeredgewidth=1.5, label=f'Left crossing (γ={gamma_left:.2f})', zorder=5)

    gamma_right = image_data.get('gamma_right')
    ap_right = image_data.get('ap_right')
    if gamma_right is not None and ap_right is not None:
        ax.plot(gamma_right, ap_right, 's', markersize=8, color='#BC4749',
                markeredgecolor='black', markeredgewidth=1.5, label=f'Right crossing (γ={gamma_right:.2f})', zorder=5)

    # Labels and title
    ax.set_xlabel('Gamma (γ)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Precision (AP)', fontsize=11, fontweight='bold')
    ax.set_title(f'{image_name} — {method.replace("_", " ").title()}', fontsize=13, fontweight='bold', pad=15)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    # Legend
    ax.legend(loc='best', framealpha=0.9, fontsize=9)

    # Tight layout
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    else:
        plt.show()


def plot_all_images_grid(
    results: list,
    target_ap: float,
    method: str,
    output_path: str = None
):
    """
    Plot all images in a grid layout.

    Args:
        results: List of image result dictionaries
        target_ap: Target AP threshold
        method: Method name for title
        output_path: Optional path to save figure
    """
    # Filter out skipped images
    valid_results = [r for r in results if not r.get('skipped', False) and r.get('all_results')]

    if not valid_results:
        print("No valid results to plot")
        return

    n_images = len(valid_results)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))

    # Handle single subplot case
    if n_images == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_images > 1 else axes

    for idx, image_data in enumerate(valid_results):
        ax = axes[idx]
        image_name = image_data['image_name']
        all_results = image_data['all_results']

        # Extract gamma and AP values
        gammas = [r[0] for r in all_results]
        aps = [r[1] for r in all_results]

        # Plot AP curve
        ax.plot(gammas, aps, 'o-', linewidth=2, markersize=3, color='#2E86AB', alpha=0.8)

        # Target AP line
        ax.axhline(y=target_ap, color='#A23B72', linestyle='--', linewidth=1.5, alpha=0.7)

        # Mark peak
        peak_gamma = image_data.get('peak_gamma')
        peak_ap = image_data.get('peak_ap')
        if peak_gamma is not None and peak_ap is not None:
            ax.plot(peak_gamma, peak_ap, 'o', markersize=8, color='#F18F01',
                    markeredgecolor='black', markeredgewidth=1, zorder=5)

        # Mark crossings
        gamma_left = image_data.get('gamma_left')
        ap_left = image_data.get('ap_left')
        if gamma_left is not None and ap_left is not None:
            ax.plot(gamma_left, ap_left, 's', markersize=6, color='#6A994E',
                    markeredgecolor='black', markeredgewidth=1, zorder=5)

        gamma_right = image_data.get('gamma_right')
        ap_right = image_data.get('ap_right')
        if gamma_right is not None and ap_right is not None:
            ax.plot(gamma_right, ap_right, 's', markersize=6, color='#BC4749',
                    markeredgecolor='black', markeredgewidth=1, zorder=5)

        # Labels and title
        ax.set_xlabel('Gamma (γ)', fontsize=9)
        ax.set_ylabel('AP', fontsize=9)
        ax.set_title(image_name, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)

    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')

    # Overall title
    fig.suptitle(f'{method.replace("_", " ").title()} — Target AP: {target_ap:.2f}',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="RQ2.a: Visualize Gamma Search Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all images from brute force search
  python scripts/rq2a_visualize_gamma_search.py --method brute_force --model yolov8

  # Visualize specific image
  python scripts/rq2a_visualize_gamma_search.py --method brute_force --model yolov8 --image amz_00832.png

  # Save to file
  python scripts/rq2a_visualize_gamma_search.py --method golden_section --model yolov8 --output my_plot.png
        """
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["brute_force", "golden_section"],
        help="Search method to visualize"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8",
        choices=["yolov5", "yolov8", "yolov12"],
        help="YOLO model (default: yolov8)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Specific image name to plot (default: all images)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: display interactively)"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Custom path to results.json (default: analysis/results/rq2a_<method>_<model>/results.json)"
    )

    args = parser.parse_args()

    # Determine results file path
    if args.results_file:
        results_file = args.results_file
    else:
        results_file = f"analysis/results/rq2a_{args.method}_{args.model}/results.json"

    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)

    # Load results
    print(f"Loading results from: {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)

    method = data['method']
    target_ap = data['target_ap']
    results = data['results']
    skipped = data.get('skipped', [])

    print(f"\nMethod: {method}")
    print(f"Model: {data['model']}")
    print(f"Target AP: {target_ap}")
    print(f"Processed images: {len(results)}")
    print(f"Skipped images: {len(skipped)}")

    # Combine results and skipped for potential filtering
    all_data = results + skipped

    # Filter by image name if specified
    if args.image:
        image_data = next((r for r in all_data if r['image_name'] == args.image), None)
        if not image_data:
            print(f"\nError: Image '{args.image}' not found in results")
            print(f"Available images: {', '.join([r['image_name'] for r in all_data])}")
            sys.exit(1)

        print(f"\nPlotting: {args.image}")
        plot_gamma_ap_curve(image_data, target_ap, method, args.output)
    else:
        print(f"\nPlotting all images...")
        if args.output:
            output_path = args.output
        else:
            output_path = None

        plot_all_images_grid(results, target_ap, method, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
