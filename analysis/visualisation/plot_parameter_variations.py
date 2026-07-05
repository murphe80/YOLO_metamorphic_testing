#!/usr/bin/env python3
"""
Plot AP vs parameter setting for each metamorphic test rule.

Generates one line chart per transformation rule, with one line per YOLO model.
Parameter values come from config/brute_force_transformations.json; AP values are
read from analysis/results/<model>/<variant_id>/<comparison>/stats.json.

Usage:
    python3 analysis/visualisation/plot_parameter_variations.py
    python3 analysis/visualisation/plot_parameter_variations.py --comparison transformed_vs_gt
    python3 analysis/visualisation/plot_parameter_variations.py --rules imgStretch imgResolution
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_registry import get_model_by_name, get_model_names

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "brute_force_transformations.json"
DEFAULT_RESULTS = PROJECT_ROOT / "analysis" / "results"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "results" / "visualisation" / "parameter_variations"

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def load_transformations_config(config_path: Path) -> Dict[str, List[dict]]:
    with open(config_path) as f:
        return json.load(f)


def get_varying_parameter(variants: List[dict]) -> Optional[str]:
    """Return the parameter key that varies across variants, if any."""
    param_keys = set()
    for variant in variants:
        param_keys.update(key for key in variant if key != "variant_id")

    varying = []
    for key in sorted(param_keys):
        values = [variant.get(key) for variant in variants]
        if len(set(values)) > 1:
            varying.append(key)

    if len(varying) == 1:
        return varying[0]
    if len(varying) > 1:
        return varying[0]
    return None


def load_metric(
    results_dir: Path,
    model: str,
    variant_id: str,
    comparison: str,
    metric: str,
) -> Optional[float]:
    stats_file = results_dir / model / variant_id / comparison / "stats.json"
    if not stats_file.exists():
        return None

    with open(stats_file) as f:
        stats = json.load(f)

    value = stats.get(metric)
    if value is None:
        return None
    return float(value)


def get_model_color(model: str) -> str:
    model_config = get_model_by_name(model)
    if model_config and "color" in model_config:
        return model_config["color"]["hex"]
    return "#888888"


def get_model_label(model: str) -> str:
    model_config = get_model_by_name(model)
    if model_config:
        return model_config.get("display_name", model)
    return model


def format_rule_name(rule: str) -> str:
    import re

    spaced = re.sub(r"([A-Z])", r" \1", rule).replace("_", " ").strip()
    return spaced.title()


def format_param_label(param_name: str) -> str:
    return param_name.replace("_", " ").title()


def collect_series(
    rule: str,
    variants: List[dict],
    models: List[str],
    results_dir: Path,
    comparison: str,
    metric: str,
) -> Tuple[Optional[str], Dict[str, List[Tuple[float, float]]], List[float]]:
    """
    Collect (parameter_value, metric) pairs per model for a rule.

    Returns:
        parameter name, {model: [(x, y), ...]}, sorted x-axis values
    """
    param_name = get_varying_parameter(variants)
    if param_name is None:
        return None, {}, []

    series: Dict[str, List[Tuple[float, float]]] = {model: [] for model in models}
    x_values = set()

    for variant in variants:
        variant_id = variant["variant_id"]
        x_value = float(variant[param_name])
        x_values.add(x_value)

        for model in models:
            metric_value = load_metric(results_dir, model, variant_id, comparison, metric)
            if metric_value is not None:
                series[model].append((x_value, metric_value))

    for model in models:
        series[model].sort(key=lambda point: point[0])

    return param_name, series, sorted(x_values)


def plot_rule(
    rule: str,
    param_name: str,
    series: Dict[str, List[Tuple[float, float]]],
    comparison: str,
    metric: str,
    output_path: Path,
) -> bool:
    has_data = any(points for points in series.values())
    if not has_data:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))

    for index, (model, points) in enumerate(sorted(series.items())):
        if not points:
            continue

        x_vals, y_vals = zip(*points)
        ax.plot(
            x_vals,
            y_vals,
            marker=MARKERS[index % len(MARKERS)],
            linewidth=2,
            markersize=6,
            color=get_model_color(model),
            label=get_model_label(model),
        )

    ax.set_xlabel(format_param_label(param_name), fontsize=11, fontweight="bold")
    ax.set_ylabel(metric, fontsize=11, fontweight="bold")
    ax.set_title(
        f"{format_rule_name(rule)} — {metric} vs {format_param_label(param_name)}\n"
        f"({comparison.replace('_', ' ')})",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def discover_models(results_dir: Path, requested_models: Optional[List[str]] = None) -> List[str]:
    if requested_models:
        return requested_models

    models = []
    for model in get_model_names():
        model_dir = results_dir / model
        if model_dir.is_dir():
            models.append(model)

    if models:
        return models

    if not results_dir.exists():
        return []

    return sorted(
        entry.name
        for entry in results_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith(".") and entry.name != "archive"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot AP vs parameter setting for metamorphic test rules"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to brute_force_transformations.json",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Directory containing analysis results",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory to save generated plots",
    )
    parser.add_argument(
        "--comparison",
        type=str,
        default="original_vs_transformed",
        choices=["original_vs_transformed", "transformed_vs_gt", "original_vs_gt"],
        help="Comparison type to plot",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="AP",
        help="Metric to plot on the y-axis (e.g. AP, AP50, AP75)",
    )
    parser.add_argument(
        "--rules",
        nargs="*",
        help="Optional list of rules to plot (default: all rules in config)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional list of models to include (default: discovered from results)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}")
        sys.exit(1)

    transformations = load_transformations_config(args.config)
    models = discover_models(args.results_dir, args.models)
    if not models:
        print("No models found in results directory.")
        sys.exit(1)

    rules = args.rules if args.rules else sorted(transformations.keys())
    generated = 0
    skipped = 0

    print(f"Models: {', '.join(models)}")
    print(f"Comparison: {args.comparison}")
    print(f"Output: {args.output_dir}\n")

    for rule in rules:
        if rule not in transformations:
            print(f"  Skip {rule}: not in transformations config")
            skipped += 1
            continue

        variants = transformations[rule]
        param_name, series, _ = collect_series(
            rule,
            variants,
            models,
            args.results_dir,
            args.comparison,
            args.metric,
        )

        if param_name is None:
            print(f"  Skip {rule}: no varying parameter across variants")
            skipped += 1
            continue

        output_path = args.output_dir / f"{rule}_{args.comparison}.png"
        if plot_rule(rule, param_name, series, args.comparison, args.metric, output_path):
            print(f"  Saved: {output_path}")
            generated += 1
        else:
            print(f"  Skip {rule}: no {args.metric} data found")
            skipped += 1

    print(f"\nDone: {generated} plot(s) generated, {skipped} skipped.")
    if generated == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
