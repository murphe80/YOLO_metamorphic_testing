#!/usr/bin/env python3
"""
Configuration Analysis Module for Metamorphic Test Tuning

Analyzes tuning results, computes disparity metrics, ranks configurations,
and generates visualizations.

Usage:
    from analysis.utils.config_analysis import (
        compute_disparity_metric,
        rank_configurations,
        plot_disparity_curve,
        export_optimal_configs
    )
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add parent directories to path to import model_registry
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_registry import get_model_names


def compute_disparity_metric(ap_scores: Dict[str, float]) -> float:
    """
    Compute disparity metric quantifying model performance differences.

    Higher disparity = configuration better highlights model differences.
    Range: [0, 1] (0 = identical performance, 1 = maximum divergence)

    Args:
        ap_scores: Dictionary mapping model names to AP50 scores
                   e.g., {"yolov5": 0.85, "yolov8": 0.82, "yolov12": 0.71}

    Returns:
        Disparity score as float

    Formula:
        disparity = max pairwise difference across all models
    """
    # Get scores for all enabled models from registry
    model_names = get_model_names()
    scores = [ap_scores.get(m, 0.0) for m in model_names]

    # Handle missing models
    valid_scores = [s for s in scores if s > 0.0]
    if len(valid_scores) < 2:
        return 0.0

    # Compute pairwise differences
    disparities = []
    for i in range(len(valid_scores)):
        for j in range(i + 1, len(valid_scores)):
            disparities.append(abs(valid_scores[i] - valid_scores[j]))

    return max(disparities) if disparities else 0.0


def compute_alternative_metrics(ap_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Compute alternative disparity metrics for comparison.

    Args:
        ap_scores: Dictionary mapping model names to AP50 scores

    Returns:
        Dictionary with alternative metrics:
        - std: Standard deviation
        - range: max - min
        - cv: Coefficient of variation (std / mean)
    """
    # Get scores for all enabled models from registry
    model_names = get_model_names()
    scores = [ap_scores.get(m, 0.0) for m in model_names]
    scores = [s for s in scores if s > 0.0]  # Filter out missing models

    if not scores or len(scores) < 2:
        return {"std": 0.0, "range": 0.0, "cv": 0.0}

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    range_score = max(scores) - min(scores)
    cv_score = std_score / mean_score if mean_score > 0 else 0.0

    return {
        "std": float(std_score),
        "range": float(range_score),
        "cv": float(cv_score),
    }


def rank_configurations(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Rank configuration variants by disparity metric.

    Args:
        results: Nested dictionary structure:
                 {
                   "imgResolution_0": {
                     "config": {"size_ratio": 0.1, ...},
                     "scores": {"yolov5": 0.85, "yolov8": 0.82, "yolov12": 0.71},
                     "transformation": "imgResolution"
                   },
                   ...
                 }

    Returns:
        DataFrame with columns:
        - variant_id
        - transformation
        - disparity
        - yolov5_ap50
        - yolov8_ap50
        - yolov12_ap50
        - config (JSON string)
        - rank (1 = highest disparity)
    """
    rows = []

    for variant_id, data in results.items():
        scores = data.get("scores", {})
        config = data.get("config", {})
        transformation = data.get("transformation", "unknown")

        disparity = compute_disparity_metric(scores)
        alt_metrics = compute_alternative_metrics(scores)

        row = {
            "variant_id": variant_id,
            "transformation": transformation,
            "disparity": disparity,
            "yolov5_ap50": scores.get("yolov5", 0.0),
            "yolov8_ap50": scores.get("yolov8", 0.0),
            "yolov12_ap50": scores.get("yolov12", 0.0),
            "std": alt_metrics["std"],
            "range": alt_metrics["range"],
            "cv": alt_metrics["cv"],
            "config": json.dumps(config),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by disparity (descending) and add rank
    df = df.sort_values("disparity", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df


def export_optimal_configs(
    ranked_df: pd.DataFrame, output_path: str, top_k: int = 3
) -> Dict:
    """
    Export top-ranked configurations per transformation.

    Args:
        ranked_df: DataFrame from rank_configurations()
        output_path: Path to save JSON file
        top_k: Number of top configs to export per transformation (default: 3)

    Returns:
        Dictionary structure:
        {
          "imgResolution": {
            "rank_1": {"size_ratio": 0.3, "disparity": 0.18, "scores": {...}},
            "rank_2": {...},
            ...
          },
          ...
        }
    """
    optimal = {}

    for transformation in ranked_df["transformation"].unique():
        transform_df = ranked_df[ranked_df["transformation"] == transformation]
        transform_df = transform_df.head(top_k)

        optimal[transformation] = {}

        for i, (_, row) in enumerate(transform_df.iterrows(), 1):
            config = json.loads(row["config"])
            # Remove variant_id from config if present
            config.pop("variant_id", None)

            optimal[transformation][f"rank_{i}"] = {
                "config": config,
                "disparity": float(row["disparity"]),
                "scores": {
                    "yolov5": float(row["yolov5_ap50"]),
                    "yolov8": float(row["yolov8_ap50"]),
                    "yolov12": float(row["yolov12_ap50"]),
                },
            }

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(optimal, f, indent=2)

    return optimal


def plot_disparity_curve(
    transformation: str,
    results: List[Dict],
    parameter_name: str,
    output_path: str,
):
    """
    Plot disparity vs parameter value curve.

    Args:
        transformation: Transformation name (e.g., "imgResolution")
        results: List of result dictionaries with config, scores, disparity
        parameter_name: Name of parameter being varied (e.g., "size_ratio")
        output_path: Path to save plot image
    """
    import matplotlib.pyplot as plt

    # Extract data
    param_values = []
    disparities = []
    v5_scores = []
    v8_scores = []
    v12_scores = []

    for result in results:
        config = result.get("config", {})
        scores = result.get("scores", {})

        # Get parameter value
        if parameter_name == "none":
            param_val = 0  # Grayscale has no parameter
        else:
            param_val = config.get(parameter_name, 0)

        param_values.append(param_val)
        disparities.append(compute_disparity_metric(scores))
        v5_scores.append(scores.get("yolov5", 0.0))
        v8_scores.append(scores.get("yolov8", 0.0))
        v12_scores.append(scores.get("yolov12", 0.0))

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Disparity curve
    ax1.plot(param_values, disparities, "o-", linewidth=2, markersize=8, color="#2F5597")
    ax1.set_xlabel(parameter_name.replace("_", " ").title(), fontsize=12)
    ax1.set_ylabel("Disparity Score", fontsize=12)
    ax1.set_title(f"{transformation} - Model Disparity vs {parameter_name}", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Highlight max disparity
    max_idx = np.argmax(disparities)
    ax1.plot(param_values[max_idx], disparities[max_idx], "*", markersize=20, color="red", label="Max Disparity")
    ax1.legend()

    # Plot 2: Individual model AP50 curves
    ax2.plot(param_values, v5_scores, "o-", label="YOLOv5", linewidth=2, markersize=6)
    ax2.plot(param_values, v8_scores, "s-", label="YOLOv8", linewidth=2, markersize=6)
    ax2.plot(param_values, v12_scores, "^-", label="YOLOv12", linewidth=2, markersize=6)
    ax2.set_xlabel(parameter_name.replace("_", " ").title(), fontsize=12)
    ax2.set_ylabel("AP50 Score", fontsize=12)
    ax2.set_title("Model Performance vs Parameter", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save figure
    output_file = Path(output_path).with_suffix(".pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()


def generate_summary_report(ranked_df: pd.DataFrame, output_path: str):
    """
    Generate a markdown summary report of tuning results.

    Args:
        ranked_df: DataFrame from rank_configurations()
        output_path: Path to save markdown report
    """
    report_lines = []

    report_lines.append("# Metamorphic Configuration Tuning Report\n")
    report_lines.append("## Summary Statistics\n")
    report_lines.append(f"- Total configurations tested: {len(ranked_df)}")
    report_lines.append(f"- Transformations: {ranked_df['transformation'].nunique()}")
    report_lines.append(f"- Average disparity: {ranked_df['disparity'].mean():.4f}")
    report_lines.append(f"- Max disparity: {ranked_df['disparity'].max():.4f}")
    report_lines.append(f"- Min disparity: {ranked_df['disparity'].min():.4f}\n")

    report_lines.append("## Top 10 Configurations (Overall)\n")
    report_lines.append("| Rank | Variant | Transformation | Disparity | v5 AP50 | v8 AP50 | v12 AP50 |")
    report_lines.append("|------|---------|----------------|-----------|---------|---------|----------|")

    for _, row in ranked_df.head(10).iterrows():
        report_lines.append(
            f"| {row['rank']} | {row['variant_id']} | {row['transformation']} | "
            f"{row['disparity']:.4f} | {row['yolov5_ap50']:.4f} | "
            f"{row['yolov8_ap50']:.4f} | {row['yolov12_ap50']:.4f} |"
        )

    report_lines.append("\n## Best Configuration Per Transformation\n")

    for transformation in ranked_df["transformation"].unique():
        transform_df = ranked_df[ranked_df["transformation"] == transformation]
        best = transform_df.iloc[0]

        report_lines.append(f"\n### {transformation}")
        report_lines.append(f"- Best variant: {best['variant_id']}")
        report_lines.append(f"- Disparity: {best['disparity']:.4f}")
        report_lines.append(f"- Config: `{best['config']}`")
        report_lines.append(f"- Scores: v5={best['yolov5_ap50']:.4f}, v8={best['yolov8_ap50']:.4f}, v12={best['yolov12_ap50']:.4f}")

    # Save report
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("\n".join(report_lines))


if __name__ == "__main__":
    # Test analysis functions
    print("=" * 80)
    print("Config Analysis Module Test")
    print("=" * 80)

    # Mock data
    test_results = {
        "imgResolution_0": {
            "config": {"size_ratio": 0.1},
            "scores": {"yolov5": 0.75, "yolov8": 0.70, "yolov12": 0.65},
            "transformation": "imgResolution",
        },
        "imgResolution_5": {
            "config": {"size_ratio": 0.6},
            "scores": {"yolov5": 0.85, "yolov8": 0.65, "yolov12": 0.82},
            "transformation": "imgResolution",
        },
        "imgColorWheel_0": {
            "config": {"hue_rotation": 0},
            "scores": {"yolov5": 0.90, "yolov8": 0.88, "yolov12": 0.87},
            "transformation": "imgColorWheel",
        },
    }

    # Test disparity computation
    print("\nDisparity Metric Tests:")
    for variant_id, data in test_results.items():
        disparity = compute_disparity_metric(data["scores"])
        print(f"  {variant_id}: {disparity:.4f}")

    # Test ranking
    print("\nRanking Test:")
    ranked = rank_configurations(test_results)
    print(ranked[["variant_id", "transformation", "disparity", "rank"]])

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
