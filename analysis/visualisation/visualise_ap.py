#!/usr/bin/env python3
"""
Visualize and compare YOLO model performance across transformations.
Generates formatted tables comparing YOLOv5, YOLOv8, and YOLOv12.

Usage:
    # Basic comparison table (console output)
    python3 analysis/visualisation/visualise_ap.py
    
    # With custom transformed images directory (for config files)
    python3 analysis/visualisation/visualise_ap.py --transformed_images_dir data/transformed_images/base_metamorphic
    
    # With detailed metrics for each transformation
    python3 analysis/visualisation/visualise_ap.py --detailed
    
    # Export to vector PDF
    python3 analysis/visualisation/visualise_ap.py --export_pdf
    
    # Custom output location
    python3 analysis/visualisation/visualise_ap.py --export_pdf --pdf_output results.pdf
    
    # All options combined
    python3 analysis/visualisation/visualise_ap.py --detailed --export_pdf --transformed_images_dir data/transformed_images/base_metamorphic

Output includes:
    - Summary comparison table showing AP50 scores for each model/transformation
    - Configuration parameters used for each transformation (from config.json files)
    - Robustness ranking (which transformations are most challenging)
    - Regression analysis (where older models outperform newer ones)
    - Optional: Detailed metrics (AP50, AP75, AR@100)
    - Optional: Vector PDF export (comparison table)
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directories to path to import model_registry
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_registry import get_model_names


def load_configs(transformed_images_dir: str = None) -> Dict:
    """Load config.json or configs.json files from transformation directories."""
    configs = {}
    
    if not transformed_images_dir or not Path(transformed_images_dir).exists():
        return configs
    
    transform_base_dir = Path(transformed_images_dir)
    
    # Find all transformation directories
    for transform_dir in transform_base_dir.iterdir():
        if not transform_dir.is_dir():
            continue
        
        transform_name = transform_dir.name
        
        # Try both config.json and configs.json
        for config_name in ['config.json', 'configs.json']:
            config_file = transform_dir / config_name
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        config_data = json.load(f)
                        # If configs.json contains a dict with transform name as key, extract it
                        if isinstance(config_data, dict) and transform_name in config_data:
                            configs[transform_name] = config_data[transform_name]
                        else:
                            configs[transform_name] = config_data
                    break  # Found a config file, no need to check the other
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse {config_file}")
                    continue
    
    return configs


def load_stats(results_dir: str = "analysis/results/rq1") -> Dict:
    """Load all stats.json files for original_vs_transformed comparisons."""
    data = {}
    models = get_model_names()  # Load from model registry

    for model in models:
        data[model] = {}
        model_dir = Path(results_dir) / model
        
        if not model_dir.exists():
            continue
        
        # Find all transformation directories
        for transform_dir in model_dir.iterdir():
            if not transform_dir.is_dir():
                continue
            
            stats_file = transform_dir / "original_vs_transformed" / "stats.json"
            if stats_file.exists():
                with open(stats_file) as f:
                    stats = json.load(f)
                    if "class_shifts" in stats:
                        if stats.get("status") == "excluded":
                            print(f"{model}: {transform_dir.name}, hue {stats['hue_rotation']}: excluded")
                        for shift in stats["class_shifts"].values():
                            label = f"{transform_dir.name} (hue {stats['hue_rotation']}: {shift['mapping']})"
                            # COCO's conventional -1 indicates an unavailable metric.
                            data[model][label] = {key: (-1 if value is None else value)
                                                  for key, value in shift.items()}
                    else:
                        data[model][transform_dir.name] = stats
    
    return data


def format_config_params(config: Dict) -> str:
    """Format configuration parameters as a compact string."""
    if not config:
        return "N/A"
    
    # Format each parameter
    parts = []
    for key, value in config.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        elif isinstance(value, (int, str)):
            parts.append(f"{key}={value}")
    
    if not parts:
        return "default"
    
    return ", ".join(parts)


def print_summary_table(data: Dict, configs: Dict = None):
    """Print a summary comparison table."""
    configs = configs or {}
    
    print("\n" + "=" * 140)
    print("YOLO MODEL COMPARISON: Original vs Transformed Detection Consistency")
    print("=" * 140)
    print("\nMetamorphic Testing Results - How well does each model maintain detection accuracy after transformations?")
    print("Higher AP50 = Better robustness (1.0 = perfect consistency, 0.0 = complete failure)")
    print("AP50 = Average Precision at IoU threshold of 0.5 (most commonly used metric)\n")
    
    # Get all transformations
    all_transforms = set()
    for model_data in data.values():
        all_transforms.update(model_data.keys())
    
    transformations = sorted(all_transforms)
    
    # Header
    print(f"{'Transformation':<20} │ {'YOLOv5':<25} │ {'YOLOv8':<25} │ {'YOLOv12':<25} │ {'Winner':<10} │ {'Config':<30}")
    print("─" * 140)
    
    # Data rows
    for transform in transformations:
        v5_stats = data.get('yolov5', {}).get(transform, {})
        v8_stats = data.get('yolov8', {}).get(transform, {})
        v12_stats = data.get('yolov12', {}).get(transform, {})
        
        v5_ap = v5_stats.get('AP50', 0)
        v8_ap = v8_stats.get('AP50', 0)
        v12_ap = v12_stats.get('AP50', 0)
        
        # Determine winner
        aps = [('v5', v5_ap), ('v8', v8_ap), ('v12', v12_ap)]
        winner = max(aps, key=lambda x: x[1])[0]
        
        # Format with color indicators
        v5_str = f"{v5_ap:.4f} ({v5_ap*100:>5.1f}%)"
        v8_str = f"{v8_ap:.4f} ({v8_ap*100:>5.1f}%)"
        v12_str = f"{v12_ap:.4f} ({v12_ap*100:>5.1f}%)"
        
        # Add markers for winner
        if winner == 'v5':
            v5_str = "★ " + v5_str
        elif winner == 'v8':
            v8_str = "★ " + v8_str
        else:
            v12_str = "★ " + v12_str
        
        winner_full = {'v5': 'YOLOv5', 'v8': 'YOLOv8', 'v12': 'YOLOv12'}[winner]
        
        # Get config info
        config_str = format_config_params(configs.get(transform, {}))
        if len(config_str) > 28:
            config_str = config_str[:25] + "..."
        
        print(f"{transform:<20} │ {v5_str:<25} │ {v8_str:<25} │ {v12_str:<25} │ {winner_full:<10} │ {config_str:<30}")
    
    print("─" * 140)
    print("★ = Best performer for this transformation\n")


def print_config_details(configs: Dict):
    """Print detailed configuration parameters for each transformation."""
    if not configs:
        return
    
    print("\n" + "=" * 120)
    print("TRANSFORMATION CONFIGURATION PARAMETERS")
    print("=" * 120)
    print("\nParameters used for each transformation:\n")
    
    for transform in sorted(configs.keys()):
        config = configs[transform]
        print(f"📋 {transform}:")
        
        if not config:
            print("   (No parameters / default settings)")
        else:
            for key, value in config.items():
                if isinstance(value, float):
                    print(f"   • {key}: {value:.4f}")
                else:
                    print(f"   • {key}: {value}")
        print()


def print_detailed_metrics(data: Dict):
    """Print detailed metrics for each transformation."""
    print("\n" + "=" * 120)
    print("DETAILED METRICS BY TRANSFORMATION")
    print("=" * 120)
    
    all_transforms = set()
    for model_data in data.values():
        all_transforms.update(model_data.keys())
    
    transformations = sorted(all_transforms)
    
    for transform in transformations:
        print(f"\n{transform}")
        print("─" * 120)
        
        # Header
        print(f"{'Model':<10} │ {'AP':<12} │ {'AP50':<12} │ {'AP75':<12} │ {'AR@1':<12} │ {'AR@10':<12} │ {'AR@100':<12}")
        print("─" * 120)
        
        for model in ['yolov5', 'yolov8', 'yolov12']:
            stats = data.get(model, {}).get(transform, {})
            
            if not stats:
                print(f"{model.upper():<10} │ {'N/A':<12} │ {'N/A':<12} │ {'N/A':<12} │ {'N/A':<12} │ {'N/A':<12} │ {'N/A':<12}")
                continue
            
            ap = stats.get('AP', 0)
            ap50 = stats.get('AP50', 0)
            ap75 = stats.get('AP75', 0)
            ar1 = stats.get('AR1', 0)
            ar10 = stats.get('AR10', 0)
            ar100 = stats.get('AR100', 0)
            
            print(f"{model.upper():<10} │ {ap:>5.4f} ({ap*100:>4.1f}%) │ "
                  f"{ap50:>5.4f} ({ap50*100:>4.1f}%) │ "
                  f"{ap75:>5.4f} ({ap75*100:>4.1f}%) │ "
                  f"{ar1:>5.4f} ({ar1*100:>4.1f}%) │ "
                  f"{ar10:>5.4f} ({ar10*100:>4.1f}%) │ "
                  f"{ar100:>5.4f} ({ar100*100:>4.1f}%)")


def print_regression_analysis(data: Dict):
    """Print analysis of regressions where v5 outperforms newer models."""
    print("\n" + "=" * 120)
    print("REGRESSION ANALYSIS: Cases Where YOLOv5 Outperforms Newer Models")
    print("=" * 120)
    
    all_transforms = set()
    for model_data in data.values():
        all_transforms.update(model_data.keys())
    
    transformations = sorted(all_transforms)
    
    regressions = []
    improvements = []
    
    for transform in transformations:
        v5_ap = data.get('yolov5', {}).get(transform, {}).get('AP50', 0)
        v8_ap = data.get('yolov8', {}).get(transform, {}).get('AP50', 0)
        v12_ap = data.get('yolov12', {}).get(transform, {}).get('AP50', 0)
        
        if v5_ap > 0 and v8_ap > 0 and v12_ap > 0:
            if v5_ap > v8_ap or v5_ap > v12_ap:
                diff_v8 = (v5_ap - v8_ap) * 100
                diff_v12 = (v5_ap - v12_ap) * 100
                regressions.append({
                    'transform': transform,
                    'v5': v5_ap,
                    'v8': v8_ap,
                    'v12': v12_ap,
                    'diff_v8': diff_v8,
                    'diff_v12': diff_v12
                })
            elif v12_ap > v5_ap and v12_ap > v8_ap:
                improvements.append({
                    'transform': transform,
                    'v5': v5_ap,
                    'v8': v8_ap,
                    'v12': v12_ap,
                    'improvement': (v12_ap - v5_ap) * 100
                })
    
    if regressions:
        print("\n🚨 REGRESSIONS (YOLOv5 performs better than v8 or v12):")
        print(f"\n{'Transformation':<20} │ {'v5 AP50':<10} │ {'v8 AP50':<10} │ {'v12 AP50':<10} │ {'v8 Δ':<10} │ {'v12 Δ':<10}")
        print("─" * 90)
        
        for reg in sorted(regressions, key=lambda x: max(x['diff_v8'], x['diff_v12']), reverse=True):
            print(f"{reg['transform']:<20} │ {reg['v5']*100:>7.2f}% │ "
                  f"{reg['v8']*100:>7.2f}% │ {reg['v12']*100:>7.2f}% │ "
                  f"{reg['diff_v8']:>+7.2f}% │ {reg['diff_v12']:>+7.2f}%")
    
    if improvements:
        print("\n✓ IMPROVEMENTS (YOLOv12 performs better than v5 and v8):")
        print(f"\n{'Transformation':<20} │ {'v5 AP50':<10} │ {'v8 AP50':<10} │ {'v12 AP50':<10} │ {'Improvement':<15}")
        print("─" * 90)
        
        for imp in sorted(improvements, key=lambda x: x['improvement'], reverse=True):
            print(f"{imp['transform']:<20} │ {imp['v5']*100:>7.2f}% │ "
                  f"{imp['v8']*100:>7.2f}% │ {imp['v12']*100:>7.2f}% │ "
                  f"+{imp['improvement']:>6.2f}%")
    
    print(f"\n📊 Summary: {len(regressions)} regressions, {len(improvements)} improvements")


def print_robustness_ranking(data: Dict):
    """Print transformations ranked by robustness."""
    print("\n" + "=" * 120)
    print("TRANSFORMATION ROBUSTNESS RANKING")
    print("=" * 120)
    print("\nAverage AP50 across all three models (higher = more robust to transformation)\n")
    
    all_transforms = set()
    for model_data in data.values():
        all_transforms.update(model_data.keys())
    
    transformations = sorted(all_transforms)
    
    rankings = []
    for transform in transformations:
        v5_ap = data.get('yolov5', {}).get(transform, {}).get('AP50', 0)
        v8_ap = data.get('yolov8', {}).get(transform, {}).get('AP50', 0)
        v12_ap = data.get('yolov12', {}).get(transform, {}).get('AP50', 0)
        
        valid_aps = [ap for ap in [v5_ap, v8_ap, v12_ap] if ap > 0]
        if valid_aps:
            avg_ap = sum(valid_aps) / len(valid_aps)
            rankings.append({
                'transform': transform,
                'avg_ap': avg_ap,
                'v5': v5_ap,
                'v8': v8_ap,
                'v12': v12_ap
            })
    
    # Sort by average AP
    rankings.sort(key=lambda x: x['avg_ap'], reverse=True)
    
    print(f"{'Rank':<6} │ {'Transformation':<20} │ {'Avg AP50':<12} │ {'Category':<20} │ {'Details':<30}")
    print("─" * 120)
    
    for i, rank in enumerate(rankings, 1):
        avg_ap = rank['avg_ap']
        
        # Categorize
        if avg_ap >= 0.9:
            category = "🟢 Highly Robust"
            detail = "Minimal detection change"
        elif avg_ap >= 0.7:
            category = "🟡 Moderately Robust"
            detail = "Some detection changes"
        elif avg_ap >= 0.5:
            category = "🟠 Low Robustness"
            detail = "Significant changes"
        else:
            category = "🔴 Very Low Robustness"
            detail = "Major detection loss"
        
        print(f"{i:<6} │ {rank['transform']:<20} │ {avg_ap:>5.4f} ({avg_ap*100:>5.1f}%) │ "
              f"{category:<20} │ {detail:<30}")


def export_table_as_pdf(data: Dict, configs: Dict = None, output_file: str = "analysis/results/rq1/visualisation/comparison_table.pdf", metric: str = "AP50"):
    """Export comparison table as a vector PDF."""
    _ = configs or {}  # reserved for future columns (e.g. configuration summary)
    all_transforms = set()
    for model_data in data.values():
        all_transforms.update(model_data.keys())

    transformations = sorted(all_transforms)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    col_labels = ["Transformation", "YOLOv5", "YOLOv8", "YOLOv12", "Winner"]
    cell_text: List[List[str]] = []

    for transform in transformations:
        v5_metric = data.get("yolov5", {}).get(transform, {}).get(metric, 0)
        v8_metric = data.get("yolov8", {}).get(transform, {}).get(metric, 0)
        v12_metric = data.get("yolov12", {}).get(transform, {}).get(metric, 0)

        winner = max(
            [("v5", v5_metric), ("v8", v8_metric), ("v12", v12_metric)], key=lambda x: x[1]
        )[0]
        winner_full = {"v5": "YOLOv5", "v8": "YOLOv8", "v12": "YOLOv12"}[winner]

        v5_str = f"{v5_metric:.4f} ({v5_metric * 100:.1f}%)"
        v8_str = f"{v8_metric:.4f} ({v8_metric * 100:.1f}%)"
        v12_str = f"{v12_metric:.4f} ({v12_metric * 100:.1f}%)"

        if winner == "v5":
            v5_str = "★ " + v5_str
        elif winner == "v8":
            v8_str = "★ " + v8_str
        else:
            v12_str = "★ " + v12_str

        cell_text.append([transform, v5_str, v8_str, v12_str, winner_full])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = len(cell_text)
    ncols = len(col_labels)
    fig_h = max(4.0, 0.55 * (nrows + 1) + 1.8)
    fig_w = max(12.0, 2.2 * ncols)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)

    for j in range(ncols):
        cell = table[(0, j)]
        cell.set_facecolor("#2F5597")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")

    for i in range(1, nrows + 1):
        for j in range(ncols):
            table[(i, j)].set_edgecolor("#CCCCCC")
        if i % 2 == 0:
            for j in range(ncols):
                table[(i, j)].set_facecolor("#F2F2F2")

    fig.suptitle(
        f"YOLO model comparison : metamorphic testing ({metric.upper()})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.35,
    )
    plt.close(fig)

    print(f"\n✓ Table exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO model performance comparison"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='analysis/results/rq1',
        help='Directory containing analysis results'
    )
    parser.add_argument(
        '--transformed_images_dir',
        type=str,
        default='data/transformed_images/fsoco_test',
        help='Directory containing transformed images with config files'
    )
    parser.add_argument(
        '--export_pdf',
        action='store_true',
        help='Export comparison table as vector PDF'
    )
    parser.add_argument(
        '--pdf_output',
        type=str,
        default='analysis/results/rq1/visualisation/comparison_table.pdf',
        help='Output PDF file path'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='AP75',
        help='Metric to compare (AP50, AP75, AR1, AR10, AR100)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed metrics for all transformations'
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading analysis results...")
    data = load_stats(args.results_dir)
    
    if not data or not any(data.values()):
        print("❌ No results found. Run analysis/scripts/run_base_metamorphic_analysis.sh first")
        return
    
    # Load configs
    print("Loading transformation configurations...")
    configs = load_configs(args.transformed_images_dir)
    if configs:
        print(f"✓ Loaded configurations for {len(configs)} transformations")
    else:
        print("⚠ No configuration files found")
    
    # Print tables
    print_summary_table(data, configs)
    print_config_details(configs)
    print_robustness_ranking(data)
    print_regression_analysis(data)
    
    if args.detailed:
        print_detailed_metrics(data)
    
    # Export PDF if requested
    if args.export_pdf:
        export_table_as_pdf(data, configs, args.pdf_output, metric=args.metric)
    
    print("\n" + "=" * 120)
    print("Analysis complete!")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
