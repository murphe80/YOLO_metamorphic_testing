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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_registry import get_default_model_name, get_model_names


AVAILABLE_MODELS = get_model_names()
DEFAULT_MODEL = get_default_model_name()
MAX_PLOTS_PER_FIGURE = 20


def get_default_output_dir(method: str, model: str) -> Path:
    """Return the visualisation directory for a search method and model."""
    return Path(f"analysis/results/rq2a_{method}_{model}/visualisations")


def has_complete_search_result(image_data: dict) -> bool:
    """Return whether both crossings and the peak were found."""
    required_values = ('peak_gamma', 'gamma_left', 'gamma_right')
    return all(image_data.get(key) is not None for key in required_values)


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

    if not has_complete_search_result(image_data):
        print(f"  Incomplete search result, not plotted: {image_name}")
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
    Plot images in grid layouts, with at most 20 plots per figure.

    Args:
        results: List of image result dictionaries
        target_ap: Target AP threshold
        method: Method name for title
        output_path: Optional path to save figure. When multiple figures are
            needed, page numbers are appended to the filename.
    """
    # Include only completed searches with both crossings and a peak.
    valid_results = [
        r for r in results
        if (
            not r.get('skipped', False)
            and r.get('all_results')
            and has_complete_search_result(r)
        )
    ]

    if not valid_results:
        print("No complete search results to plot")
        return

    n_pages = (len(valid_results) + MAX_PLOTS_PER_FIGURE - 1) // MAX_PLOTS_PER_FIGURE

    for page_index in range(n_pages):
        start = page_index * MAX_PLOTS_PER_FIGURE
        page_results = valid_results[start:start + MAX_PLOTS_PER_FIGURE]
        n_images = len(page_results)
        n_cols = min(3, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        for idx, image_data in enumerate(page_results):
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

        page_label = f' — Page {page_index + 1}/{n_pages}' if n_pages > 1 else ''
        fig.suptitle(
            f'{method.replace("_", " ").title()} — Target AP: {target_ap:.2f}{page_label}',
            fontsize=14,
            fontweight='bold',
            y=0.995
        )

        fig.tight_layout()

        if output_path:
            page_output_path = Path(output_path)
            if n_pages > 1:
                page_output_path = page_output_path.with_name(
                    f'{page_output_path.stem}_page_{page_index + 1:02d}{page_output_path.suffix}'
                )
            fig.savefig(page_output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {page_output_path}")
            plt.close(fig)

    if not output_path:
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

  # Save to a custom directory
  python scripts/rq2a_visualize_gamma_search.py --method golden_section --model yolov8 --output analysis/my_visualisations
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
        default=DEFAULT_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"YOLO model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Specific image name to plot (default: all images)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory (default: "
            "analysis/results/rq2a_<method>_<model>/visualisations)"
        )
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

    output_dir = args.output or get_default_output_dir(args.method, args.model)
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"--output must be a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Output directory: {output_dir}")

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
        output_path = output_dir / f"{Path(args.image).stem}_gamma_search.png"
        plot_gamma_ap_curve(image_data, target_ap, method, output_path)
    else:
        print(f"\nPlotting all images...")
        output_path = output_dir / "gamma_search_results.png"
        plot_all_images_grid(results, target_ap, method, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
