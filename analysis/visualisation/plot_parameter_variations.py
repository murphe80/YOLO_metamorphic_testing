#!/usr/bin/env python3
"""
Plot AP and AR vs parameter setting for each metamorphic test rule.

Generates precision and AR line charts per transformation rule, with one line
per YOLO model, plus a JSON file containing the selected precision metric and
AR for every rule, setting, and model.
Parameter values come from config/brute_force_transformations.json; metric values
are read from analysis/results/<dataset>/<model>/<variant_id>/<comparison>/stats.json.
AR plots use the COCO AR100 metric and are saved under the ar_plots output
subdirectory.

Usage:
    python3 analysis/visualisation/plot_parameter_variations.py
    python3 analysis/visualisation/plot_parameter_variations.py --comparison transformed_vs_gt
    python3 analysis/visualisation/plot_parameter_variations.py --rules imgStretch imgResolution
    python3 analysis/visualisation/plot_parameter_variations.py --dataset test
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
RESULTS_ROOT = PROJECT_ROOT / "analysis" / "results"
DEFAULT_DATASET = "test"
AR_METRIC = "AR"


def dataset_results_dir(dataset: str) -> Path:
    return RESULTS_ROOT / dataset


def dataset_output_dir(dataset: str) -> Path:
    return RESULTS_ROOT / dataset / "parameter_variations"

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
    if value is None and metric == AR_METRIC:
        # COCO reports overall average recall at maxDets=100 as AR100.
        value = stats.get("AR100")
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


def collect_rule_metrics(
    variants: List[dict],
    models: List[str],
    results_dir: Path,
    comparison: str,
    precision_metric: str,
) -> List[dict]:
    """Collect the selected precision metric and AR for every setting."""
    settings = []

    for variant in variants:
        variant_id = variant["variant_id"]
        model_metrics = {}
        for model in models:
            model_metrics[model] = {
                precision_metric: load_metric(
                    results_dir,
                    model,
                    variant_id,
                    comparison,
                    precision_metric,
                ),
                "AR": load_metric(
                    results_dir, model, variant_id, comparison, AR_METRIC
                ),
            }

        settings.append(
            {
                "variant_id": variant_id,
                "setting": {
                    key: value for key, value in variant.items() if key != "variant_id"
                },
                "models": model_metrics,
            }
        )

    return settings


def write_metrics_json(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")


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
        description="Plot AP and AR vs parameter setting for metamorphic test rules"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to brute_force_transformations.json",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=["test", "val", "train_samples", "competition", "comp_labelled"],
        help="Dataset split to plot (default: test)",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Directory containing analysis results (default: analysis/results/<dataset>)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save generated plots (default: analysis/results/<dataset>/parameter_variations)",
    )
    parser.add_argument(
        "--json_output",
        type=Path,
        default=None,
        help=(
            "Path for AP/AR JSON output "
            "(default: <output_dir>/metrics_<comparison>.json)"
        ),
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

    results_dir = args.results_dir or dataset_results_dir(args.dataset)
    output_dir = args.output_dir or dataset_output_dir(args.dataset)
    json_output = args.json_output or output_dir / f"metrics_{args.comparison}.json"

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    transformations = load_transformations_config(args.config)
    models = discover_models(results_dir, args.models)
    if not models:
        print("No models found in results directory.")
        sys.exit(1)

    rules = args.rules if args.rules else sorted(transformations.keys())
    generated = 0
    skipped = 0
    ar_generated = 0
    ar_skipped = 0
    ar_output_dir = output_dir / "ar_plots"
    metrics_export = {
        "dataset": args.dataset,
        "comparison": args.comparison,
        "precision_metric": args.metric,
        "models": models,
        "rules": {},
    }

    print(f"Dataset: {args.dataset}")
    print(f"Models: {', '.join(models)}")
    print(f"Comparison: {args.comparison}")
    print(f"Output: {output_dir}")
    print(f"AR output: {ar_output_dir}\n")

    for rule in rules:
        if rule not in transformations:
            print(f"  Skip {rule}: not in transformations config")
            skipped += 1
            continue

        variants = transformations[rule]
        metrics_export["rules"][rule] = {
            "varying_parameter": get_varying_parameter(variants),
            "settings": collect_rule_metrics(
                variants,
                models,
                results_dir,
                args.comparison,
                args.metric,
            ),
        }
        param_name, series, _ = collect_series(
            rule,
            variants,
            models,
            results_dir,
            args.comparison,
            args.metric,
        )

        if param_name is None:
            print(f"  Skip {rule}: no varying parameter across variants")
            skipped += 1
            ar_skipped += 1
            continue

        output_path = output_dir / f"{rule}_{args.comparison}.png"
        if plot_rule(rule, param_name, series, args.comparison, args.metric, output_path):
            print(f"  Saved: {output_path}")
            generated += 1
        else:
            print(f"  Skip {rule}: no {args.metric} data found")
            skipped += 1

        _, ar_series, _ = collect_series(
            rule,
            variants,
            models,
            results_dir,
            args.comparison,
            AR_METRIC,
        )
        ar_output_path = ar_output_dir / f"{rule}_{args.comparison}.png"
        if plot_rule(
            rule,
            param_name,
            ar_series,
            args.comparison,
            AR_METRIC,
            ar_output_path,
        ):
            print(f"  Saved: {ar_output_path}")
            ar_generated += 1
        else:
            print(f"  Skip {rule}: no {AR_METRIC} data found")
            ar_skipped += 1

    write_metrics_json(metrics_export, json_output)
    print(f"  Saved: {json_output}")

    print(
        f"\nDone: {generated} {args.metric} plot(s) generated, {skipped} skipped; "
        f"{ar_generated} AR plot(s) generated, {ar_skipped} skipped."
    )
    has_exported_metrics = any(
        value is not None
        for rule_data in metrics_export["rules"].values()
        for setting in rule_data["settings"]
        for model_data in setting["models"].values()
        for value in model_data.values()
    )
    if generated + ar_generated == 0 and not has_exported_metrics:
        sys.exit(1)


if __name__ == "__main__":
    main()
