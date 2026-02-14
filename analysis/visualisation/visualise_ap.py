#!/usr/bin/env python3
"""
Visualize and compare YOLO model performance across transformations.
Generates formatted tables comparing YOLOv5, YOLOv8, and YOLOv12.

Usage:
    # Basic comparison table (console output)
    python3 analysis/visualisation/visualise_ap.py
    
    # With detailed metrics for each transformation
    python3 analysis/visualisation/visualise_ap.py --detailed
    
    # Export to PNG image
    python3 analysis/visualisation/visualise_ap.py --export_png
    
    # Custom output location
    python3 analysis/visualisation/visualise_ap.py --export_png --png_output results.png
    
    # All options combined
    python3 analysis/visualisation/visualise_ap.py --detailed --export_png

Output includes:
    - Summary comparison table showing AP50 scores for each model/transformation
    - Robustness ranking (which transformations are most challenging)
    - Regression analysis (where older models outperform newer ones)
    - Optional: Detailed metrics (AP50, AP75, AR@100)
    - Optional: PNG image export (high-resolution table)
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List


def load_stats(results_dir: str = "analysis/results") -> Dict:
    """Load all stats.json files for original_vs_transformed comparisons."""
    data = {}
    models = ['yolov5', 'yolov8', 'yolov12']
    
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
                    data[model][transform_dir.name] = json.load(f)
    
    return data


def print_summary_table(data: Dict):
    """Print a summary comparison table."""
    print("\n" + "=" * 120)
    print("YOLO MODEL COMPARISON: Original vs Transformed Detection Consistency")
    print("=" * 120)
    print("\nMetamorphic Testing Results - How well does each model maintain detection accuracy after transformations?")
    print("Higher AP50 = Better robustness (1.0 = perfect consistency, 0.0 = complete failure)")
    print("AP50 = Average Precision at IoU threshold of 0.5 (most commonly used metric)\n")
    
    # Get all transformations
    all_transforms = set()
    for model_data in data.values():
        all_transforms.update(model_data.keys())
    
    transformations = sorted(all_transforms)
    
    # Header
    print(f"{'Transformation':<20} │ {'YOLOv5':<25} │ {'YOLOv8':<25} │ {'YOLOv12':<25} │ {'Winner':<10}")
    print("─" * 120)
    
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
        
        print(f"{transform:<20} │ {v5_str:<25} │ {v8_str:<25} │ {v12_str:<25} │ {winner_full:<10}")
    
    print("─" * 120)
    print("★ = Best performer for this transformation\n")


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


def export_table_as_png(data: Dict, output_file: str = "analysis/results/comparison_table.png"):
    """Export comparison data to a formatted table file."""
    all_transforms = set()
    for model_data in data.values():
        all_transforms.update(model_data.keys())
    
    transformations = sorted(all_transforms)
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        if format == "markdown":
            # Markdown table
            f.write("# YOLO Model Comparison: Metamorphic Testing Results\n\n")
            f.write("Comparison of detection consistency across transformations (Original vs Transformed)\n\n")
            
            # Summary table
            f.write("## Summary Table (AP50 Scores)\n\n")
            f.write("| Transformation | YOLOv5 | YOLOv8 | YOLOv12 | Winner | Avg AP50 |\n")
            f.write("|---|---|---|---|---|---|\n")
            
            for transform in transformations:
                v5_stats = data.get('yolov5', {}).get(transform, {})
                v8_stats = data.get('yolov8', {}).get(transform, {})
                v12_stats = data.get('yolov12', {}).get(transform, {})
                
                v5_ap = v5_stats.get('AP50', 0)
                v8_ap = v8_stats.get('AP50', 0)
                v12_ap = v12_stats.get('AP50', 0)
                
                winner = max([('v5', v5_ap), ('v8', v8_ap), ('v12', v12_ap)], key=lambda x: x[1])[0]
                avg_ap = (v5_ap + v8_ap + v12_ap) / 3 if v5_ap > 0 and v8_ap > 0 and v12_ap > 0 else 0
                
                winner_full = {'v5': 'YOLOv5', 'v8': 'YOLOv8', 'v12': 'YOLOv12'}[winner]
                
                v5_str = f"{v5_ap:.4f} ({v5_ap*100:.1f}%)"
                v8_str = f"{v8_ap:.4f} ({v8_ap*100:.1f}%)"
                v12_str = f"{v12_ap:.4f} ({v12_ap*100:.1f}%)"
                
                # Bold the winner
                if winner == 'v5':
                    v5_str = f"**{v5_str}**"
                elif winner == 'v8':
                    v8_str = f"**{v8_str}**"
                else:
                    v12_str = f"**{v12_str}**"
                
                f.write(f"| {transform} | {v5_str} | {v8_str} | {v12_str} | {winner_full} | {avg_ap:.4f} ({avg_ap*100:.1f}%) |\n")
            
            # Detailed metrics
            f.write("\n## Detailed Metrics\n\n")
            
            for transform in transformations:
                f.write(f"\n### {transform}\n\n")
                f.write("| Model | AP50 | AP75 | AR@100 |\n")
                f.write("|---|---|---|---|\n")
                
                for model in ['yolov5', 'yolov8', 'yolov12']:
                    stats = data.get(model, {}).get(transform, {})
                    
                    if not stats:
                        f.write(f"| {model.upper()} | N/A | N/A | N/A |\n")
                        continue
                    
                    ap50 = stats.get('AP50', 0)
                    ap75 = stats.get('AP75', 0)
                    ar100 = stats.get('AR100', 0)
                    
                    f.write(f"| {model.upper()} | "
                           f"{ap50:.4f} ({ap50*100:.1f}%) | "
                           f"{ap75:.4f} ({ap75*100:.1f}%) | "
                           f"{ar100:.4f} ({ar100*100:.1f}%) |\n")
            
            # Robustness ranking
            f.write("\n## Transformation Robustness Ranking\n\n")
            f.write("| Rank | Transformation | Avg AP50 | Category |\n")
            f.write("|---|---|---|---|\n")
            
            rankings = []
            for transform in transformations:
                v5_ap = data.get('yolov5', {}).get(transform, {}).get('AP50', 0)
                v8_ap = data.get('yolov8', {}).get(transform, {}).get('AP50', 0)
                v12_ap = data.get('yolov12', {}).get(transform, {}).get('AP50', 0)
                
                valid_aps = [ap for ap in [v5_ap, v8_ap, v12_ap] if ap > 0]
                if valid_aps:
                    avg_ap = sum(valid_aps) / len(valid_aps)
                    rankings.append({'transform': transform, 'avg_ap': avg_ap})
            
            rankings.sort(key=lambda x: x['avg_ap'], reverse=True)
            
            for i, rank in enumerate(rankings, 1):
                avg_ap = rank['avg_ap']
                if avg_ap >= 0.9:
                    category = "🟢 Highly Robust"
                elif avg_ap >= 0.7:
                    category = "🟡 Moderately Robust"
                elif avg_ap >= 0.5:
                    category = "🟠 Low Robustness"
                else:
                    category = "🔴 Very Low Robustness"
                
                f.write(f"| {i} | {rank['transform']} | {avg_ap:.4f} ({avg_ap*100:.1f}%) | {category} |\n")
            
            # Key findings
            f.write("\n## Key Findings\n\n")
            
            # Count regressions
            regression_count = 0
            improvement_count = 0
            for transform in transformations:
                v5_ap = data.get('yolov5', {}).get(transform, {}).get('AP50', 0)
                v8_ap = data.get('yolov8', {}).get(transform, {}).get('AP50', 0)
                v12_ap = data.get('yolov12', {}).get(transform, {}).get('AP50', 0)
                
                if v5_ap > 0 and v8_ap > 0 and v12_ap > 0:
                    if v5_ap > v8_ap or v5_ap > v12_ap:
                        regression_count += 1
                    if v12_ap > v5_ap and v12_ap > v8_ap:
                        improvement_count += 1
            
            f.write(f"- **Regressions**: YOLOv5 outperforms newer models in {regression_count}/{len(transformations)} transformations\n")
            f.write(f"- **Improvements**: YOLOv12 outperforms older models in {improvement_count}/{len(transformations)} transformations\n")
            
            # Most challenging transformation
            if rankings:
                worst = rankings[-1]
                f.write(f"- **Most Challenging**: {worst['transform']} (Avg AP: {worst['avg_ap']*100:.1f}%)\n")
                best = rankings[0]
                f.write(f"- **Most Robust**: {best['transform']} (Avg AP: {best['avg_ap']*100:.1f}%)\n")
        
        else:  # ASCII format
            # ASCII table format
            f.write("=" * 100 + "\n")
            f.write("YOLO MODEL COMPARISON: METAMORPHIC TESTING RESULTS\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"{'Transformation':<20} | {'YOLOv5':<20} | {'YOLOv8':<20} | {'YOLOv12':<20} | {'Winner':<10}\n")
            f.write("-" * 100 + "\n")
            
            for transform in transformations:
                v5_stats = data.get('yolov5', {}).get(transform, {})
                v8_stats = data.get('yolov8', {}).get(transform, {})
                v12_stats = data.get('yolov12', {}).get(transform, {})
                
                v5_ap = v5_stats.get('AP50', 0)
                v8_ap = v8_stats.get('AP50', 0)
                v12_ap = v12_stats.get('AP50', 0)
                
                winner = max([('v5', v5_ap), ('v8', v8_ap), ('v12', v12_ap)], key=lambda x: x[1])[0]
                winner_full = {'v5': 'YOLOv5', 'v8': 'YOLOv8', 'v12': 'YOLOv12'}[winner]
                
                v5_str = f"{v5_ap:.4f} ({v5_ap*100:>5.1f}%)"
                v8_str = f"{v8_ap:.4f} ({v8_ap*100:>5.1f}%)"
                v12_str = f"{v12_ap:.4f} ({v12_ap*100:>5.1f}%)"
                
                # Add star for winner
                if winner == 'v5':
                    v5_str = "★ " + v5_str
                elif winner == 'v8':
                    v8_str = "★ " + v8_str
                else:
                    v12_str = "★ " + v12_str
                
                f.write(f"{transform:<20} | {v5_str:<20} | {v8_str:<20} | {v12_str:<20} | {winner_full:<10}\n")
            
            f.write("\n" + "=" * 100 + "\n")
    
    print(f"\n✓ Table exported to: {output_file}")
    print(f"   Format: {format.upper()}")
    if format == "markdown":
        print(f"   View in GitHub, VSCode, or any Markdown viewer for best rendering")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO model performance comparison"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='analysis/results',
        help='Directory containing analysis results'
    )
    parser.add_argument(
        '--export_png',
        action='store_true',
        help='Export comparison table as PNG image'
    )
    parser.add_argument(
        '--png_output',
        type=str,
        default='analysis/results/comparison_table.png',
        help='Output PNG file path'
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
        print("❌ No results found. Run the analysis first with run_all_analysis_matias_meta.sh")
        return
    
    # Print tables
    print_summary_table(data)
    print_robustness_ranking(data)
    print_regression_analysis(data)
    
    if args.detailed:
        print_detailed_metrics(data)
    
    # Export PNG if requested
    if args.export_png:
        export_table_as_png(data, args.png_output)
    
    print("\n" + "=" * 120)
    print("Analysis complete!")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()

