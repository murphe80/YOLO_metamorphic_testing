#!/usr/bin/env python3
"""Compare YOLO precision and AR across cone-colour metamorphic tests.

For every selected dataset, the script creates a single 2x2 figure containing
precision and AR100 grouped-bar charts for both supported comparisons:

* transformed_vs_gt: transformed predictions against colour-adjusted ground truth
* original_vs_transformed: transformed predictions against colour-adjusted
  original predictions

Metrics are read from ``analysis/results/<dataset>/<model>/<rule>/<comparison>/
stats.json``.  A JSON export next to the figure records the values used to make
the comparison, including unavailable model/rule/comparison combinations.

Examples:
    python3 analysis/visualisation/plot_cone_color_analysis.py --datasets test val
    python3 analysis/visualisation/plot_cone_color_analysis.py --datasets test
    python3 analysis/visualisation/plot_cone_color_analysis.py --datasets comp_labelled --precision-metric AP50
    python3 analysis/visualisation/plot_cone_color_analysis.py --datasets test --models yolov5 yolov8
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# The workspace can be mounted without a writable user configuration directory.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_metamorphic_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_registry import get_model_by_name, get_model_names


PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "analysis" / "results"
RULES = [
    "yellow_to_blue",
    "blue_to_yellow",
    "yellow_to_orange",
    "blue_to_orange",
    "orange_to_blue",
    "orange_to_yellow",
    "swap_sides",
]
COMPARISONS = ["transformed_vs_gt", "original_vs_transformed"]
DEFAULT_DATASETS = ["test"]


def model_label(model: str) -> str:
    """Return the configured display name, or a useful fallback."""
    config = get_model_by_name(model)
    return config.get("display_name", model) if config else model


def model_color(model: str) -> str:
    """Return the configured model colour, or a neutral fallback."""
    config = get_model_by_name(model)
    if config and config.get("color", {}).get("hex"):
        return config["color"]["hex"]
    return "#777777"


def discover_models(dataset_dir: Path, requested_models: Optional[List[str]]) -> List[str]:
    """Find model result directories, retaining the registry's stable ordering."""
    if requested_models:
        return requested_models

    configured = [
        model for model in get_model_names() if (dataset_dir / model).is_dir()
    ]
    extra = sorted(
        entry.name
        for entry in dataset_dir.iterdir()
        if entry.is_dir()
        and entry.name not in configured
        and any((entry / rule).is_dir() for rule in RULES)
    )
    return configured + extra


def load_metric(stats_file: Path, metric_key: str) -> Optional[float]:
    """Read one metric from a stats file, returning None when unavailable."""
    if not stats_file.is_file():
        return None
    try:
        with stats_file.open() as handle:
            value = json.load(handle).get(metric_key)
    except (json.JSONDecodeError, OSError) as error:
        print(f"Warning: cannot read {stats_file}: {error}", file=sys.stderr)
        return None
    return float(value) if value is not None else None


def collect_metrics(
    dataset_dir: Path,
    models: Iterable[str],
    precision_metric: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]]:
    """Collect the selected precision metric and AR100."""
    metric_specs = [(precision_metric, precision_metric), ("AR", "AR100")]
    results = {}
    for comparison in COMPARISONS:
        results[comparison] = {}
        for rule in RULES:
            results[comparison][rule] = {}
            for model in models:
                stats_file = dataset_dir / model / rule / comparison / "stats.json"
                results[comparison][rule][model] = {
                    export_name: load_metric(stats_file, stats_name)
                    for export_name, stats_name in metric_specs
                }
    return results


def available_rules(metrics: dict) -> List[str]:
    """Return rules for which either comparison has at least one metric."""
    return [
        rule
        for rule in RULES
        if any(
            value is not None
            for comparison in COMPARISONS
            for model_metrics in metrics[comparison][rule].values()
            for value in model_metrics.values()
        )
    ]


def humanise_comparison(comparison: str) -> str:
    return {
        "transformed_vs_gt": "Transformed vs. ground truth",
        "original_vs_transformed": "Original vs. transformed",
    }[comparison]


def humanise_rule(rule: str) -> str:
    return rule.replace("_to_", " → ").replace("_", " ")


def plot_comparison(
    ax: plt.Axes,
    metrics: dict,
    rules: List[str],
    models: List[str],
    metric: str,
    comparison: str,
) -> None:
    """Add one grouped-bar chart, leaving unavailable values unplotted."""
    positions = np.arange(len(rules))
    width = 0.8 / max(len(models), 1)
    offset = (len(models) - 1) * width / 2

    for index, model in enumerate(models):
        values = [metrics[comparison][rule][model][metric] for rule in rules]
        bars = ax.bar(
            positions + index * width - offset,
            [0 if value is None else value for value in values],
            width=width,
            color=model_color(model),
            label=model_label(model),
        )
        for bar, value in zip(bars, values):
            if value is None:
                bar.set_visible(False)

    ax.set_title(f"{metric} — {humanise_comparison(comparison)}", fontweight="bold")
    ax.set_xticks(positions, [humanise_rule(rule) for rule in rules], rotation=18, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)


def write_export(
    dataset: str,
    models: List[str],
    metrics: dict,
    precision_metric: str,
    output_path: Path,
) -> None:
    """Write the self-describing comparison used for the visualisation."""
    rules_with_data = available_rules(metrics)
    unavailable = [rule for rule in RULES if rule not in rules_with_data]
    export = {
        "dataset": dataset,
        "precision_metric": precision_metric,
        "metrics": {
            precision_metric: f"COCO average precision ({precision_metric})",
            "AR": "COCO average recall at maxDets=100 (AR100)",
        },
        "comparisons": {
            "transformed_vs_gt": "Transformed detections versus colour-adjusted ground truth.",
            "original_vs_transformed": "Transformed detections versus colour-adjusted original detections.",
        },
        "models": [
            {"name": model, "display_name": model_label(model)} for model in models
        ],
        "rules_with_data": rules_with_data,
        "rules_without_data": unavailable,
        "results": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(export, handle, indent=2)
        handle.write("\n")


def generate_dataset(
    dataset: str,
    requested_models: Optional[List[str]],
    precision_metric: str,
) -> bool:
    """Generate both deliverables for one dataset. Returns whether data was found."""
    dataset_dir = RESULTS_ROOT / dataset
    if not dataset_dir.is_dir():
        print(f"Skipping {dataset}: results directory not found: {dataset_dir}", file=sys.stderr)
        return False

    models = discover_models(dataset_dir, requested_models)
    if not models:
        print(f"Skipping {dataset}: no model directories found.", file=sys.stderr)
        return False

    metrics = collect_metrics(dataset_dir, models, precision_metric)
    rules = available_rules(metrics)
    output_dir = dataset_dir / "cone_color_analysis"
    suffix = "" if precision_metric == "AP" else f"_{precision_metric.lower()}"
    json_path = output_dir / f"cone_color_comparison{suffix}.json"
    write_export(dataset, models, metrics, precision_metric, json_path)

    if not rules:
        print(f"Skipping {dataset}: no cone-colour metrics found; saved {json_path}", file=sys.stderr)
        return False

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    for column, comparison in enumerate(COMPARISONS):
        for row, metric in enumerate((precision_metric, "AR")):
            plot_comparison(axes[row, column], metrics, rules, models, metric, comparison)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(models), frameon=False)
    figure.suptitle(
        f"Cone-colour test performance by YOLO model — {dataset}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"cone_color_comparison{suffix}.png"
    figure.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved: {png_path}")
    print(f"Saved: {json_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate four precision/AR cone-colour comparison charts per dataset."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset names to analyse (default: test).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional model names to include (default: models discovered in each dataset).",
    )
    parser.add_argument(
        "--precision-metric",
        choices=("AP", "AP50", "AP75"),
        default="AP",
        help="COCO precision metric to plot and export (default: AP).",
    )
    args = parser.parse_args()

    successful = [
        generate_dataset(dataset, args.models, args.precision_metric)
        for dataset in args.datasets
    ]
    if not any(successful):
        sys.exit(1)


if __name__ == "__main__":
    main()
