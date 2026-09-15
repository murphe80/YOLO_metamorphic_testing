#!/usr/bin/env python3
"""Plot Venn-derived metamorphic metrics across every configured rule setting.

One image is generated per metamorphic rule. Each image contains one subplot
per YOLO model, and each subplot shows grouped bars for the configured rule
settings and the five Venn metrics.

By default, metrics are read from::

    analysis/results/<dataset>/venn_diagrams/venn_statistics.json

and plots are written to::

    analysis/results/<dataset>/venn_diagrams/bar_charts/

Examples:
    python3 analysis/visualisation/plot_venn_metrics.py
    python3 analysis/visualisation/plot_venn_metrics.py --dataset val
    python3 analysis/visualisation/plot_venn_metrics.py --rules imgDarkBright
    python3 analysis/visualisation/plot_venn_metrics.py --models yolov5 yolov8
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "brute_force_transformations.json"
RESULTS_ROOT = PROJECT_ROOT / "analysis" / "results"
DEFAULT_DATASET = "test"

METRICS = [
    "tp_retention_rate",
    "fp_consistency_rate",
    "detection_loss_rate",
    "new_hallucination_rate",
    "new_tp_rate",
]

METRIC_LABELS = {
    "tp_retention_rate": "TP retention",
    "fp_consistency_rate": "FP consistency",
    "detection_loss_rate": "Detection loss",
    "new_hallucination_rate": "New hallucination",
    "new_tp_rate": "New TP",
}


def load_json(path: Path) -> dict:
    """Load a JSON object from *path*."""
    with open(path) as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def format_name(name: str) -> str:
    """Convert snake/camel case identifiers to readable title case."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return spaced.replace("_", " ").strip().title()


def varying_parameters(variants: Sequence[dict]) -> List[str]:
    """Return configuration fields whose values differ between variants."""
    keys = sorted({key for variant in variants for key in variant if key != "variant_id"})
    return [key for key in keys if len({variant.get(key) for variant in variants}) > 1]


def setting_labels(variants: Sequence[dict]) -> Tuple[List[str], str]:
    """Build x tick labels and an axis label from transformation settings."""
    varying = varying_parameters(variants)

    if len(varying) == 1:
        parameter = varying[0]
        return [str(variant.get(parameter)) for variant in variants], format_name(parameter)

    if varying:
        labels = [
            "\n".join(f"{format_name(key)}={variant.get(key)}" for key in varying)
            for variant in variants
        ]
        return labels, "Rule setting"

    labels = []
    for variant in variants:
        parameters = [(key, value) for key, value in variant.items() if key != "variant_id"]
        if parameters:
            labels.append(", ".join(f"{format_name(key)}={value}" for key, value in parameters))
        else:
            labels.append(variant["variant_id"])
    return labels, "Rule setting"


def split_statistics_key(key: str, variant_ids: Sequence[str]) -> Optional[Tuple[str, str]]:
    """Split ``<model>_<variant_id>`` while allowing underscores in either part."""
    for variant_id in sorted(variant_ids, key=len, reverse=True):
        suffix = f"_{variant_id}"
        if key.endswith(suffix):
            model = key[: -len(suffix)]
            if model:
                return model, variant_id
    return None


def registry_model_names() -> List[str]:
    """Return every model name from the registry, if it is available."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from utils.model_registry import get_model_names

        return get_model_names(enabled_only=False)
    except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError):
        return []


def index_statistics(
    statistics: Dict[str, dict], transformations: Dict[str, List[dict]]
) -> Tuple[Dict[Tuple[str, str], dict], List[str], List[str]]:
    """Index statistics and discover models and transformation variant IDs."""
    variant_ids = [
        variant["variant_id"]
        for variants in transformations.values()
        for variant in variants
    ]
    indexed: Dict[Tuple[str, str], dict] = {}
    discovered_models: List[str] = []
    discovered_variants: List[str] = []
    known_models = sorted(registry_model_names(), key=len, reverse=True)

    for key, values in statistics.items():
        split = None
        for model in known_models:
            prefix = f"{model}_"
            if key.startswith(prefix):
                split = (model, key[len(prefix) :])
                break
        if split is None:
            split = split_statistics_key(key, variant_ids)
        if split is None or not isinstance(values, dict):
            continue
        model, variant_id = split
        indexed[(model, variant_id)] = values
        if model not in discovered_models:
            discovered_models.append(model)
        if variant_id not in discovered_variants:
            discovered_variants.append(variant_id)

    return indexed, discovered_models, discovered_variants


def ordered_models(discovered: Sequence[str], requested: Optional[Sequence[str]]) -> List[str]:
    """Use requested order, otherwise registry order followed by unknown models."""
    if requested:
        return list(dict.fromkeys(requested))

    registry_order = registry_model_names()

    result = [model for model in registry_order if model in discovered]
    result.extend(sorted(model for model in discovered if model not in result))
    return result


def plot_rule(
    rule: str,
    variants: Sequence[dict],
    models: Sequence[str],
    statistics: Dict[Tuple[str, str], dict],
    output_path: Path,
    columns: int,
) -> bool:
    """Create the multi-model grouped bar chart for one rule."""
    available_models = [
        model
        for model in models
        if any((model, variant["variant_id"]) in statistics for variant in variants)
    ]
    if not available_models:
        return False

    column_count = min(columns, len(available_models))
    row_count = math.ceil(len(available_models) / column_count)
    figure_width = max(12.0, 7.0 * column_count)
    figure_height = 4.8 * row_count + 1.5
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(figure_width, figure_height),
        sharey=True,
        squeeze=False,
    )

    x = np.arange(len(variants), dtype=float)
    bar_width = 0.16
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(METRICS)))
    tick_labels, x_label = setting_labels(variants)
    legend_handles = []

    for subplot_index, model in enumerate(available_models):
        ax = axes.flat[subplot_index]
        for metric_index, (metric, color) in enumerate(zip(METRICS, colors)):
            offset = (metric_index - (len(METRICS) - 1) / 2) * bar_width
            values = []
            for variant in variants:
                record = statistics.get((model, variant["variant_id"]), {})
                value = record.get(metric)
                values.append(float(value) if value is not None else np.nan)

            bars = ax.bar(
                x + offset,
                values,
                width=bar_width,
                color=color,
                label=METRIC_LABELS[metric],
            )
            if subplot_index == 0:
                legend_handles.append(bars)

        ax.set_title(format_name(model), fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=35, ha="right")
        ax.set_xlabel(x_label, fontsize=10, fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)
        if subplot_index % column_count == 0:
            ax.set_ylabel("Rate", fontsize=10, fontweight="bold")

    for unused_index in range(len(available_models), row_count * column_count):
        axes.flat[unused_index].set_visible(False)

    fig.suptitle(
        f"{format_name(rule)} — Venn Metrics Across Rule Settings",
        fontsize=15,
        fontweight="bold",
    )
    fig.legend(
        legend_handles,
        [METRIC_LABELS[metric] for metric in METRICS],
        loc="lower center",
        ncol=len(METRICS),
        framealpha=0.9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Venn metrics across settings, with one subplot per YOLO model."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset name (default: test)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--statistics",
        type=Path,
        default=None,
        help="Statistics JSON (default: analysis/results/<dataset>/venn_diagrams/venn_statistics.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: analysis/results/<dataset>/venn_diagrams/bar_charts)",
    )
    parser.add_argument("--rules", nargs="*", help="Rules to plot (default: all configured rules)")
    parser.add_argument("--models", nargs="*", help="Models to plot (default: all found in statistics)")
    parser.add_argument(
        "--columns", type=int, default=3, help="Maximum subplot columns per image (default: 3)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    statistics_path = args.statistics or (
        RESULTS_ROOT / args.dataset / "venn_diagrams" / "venn_statistics.json"
    )
    output_dir = args.output_dir or (
        RESULTS_ROOT / args.dataset / "venn_diagrams" / "bar_charts"
    )

    if args.columns < 1:
        print("Error: --columns must be at least 1", file=sys.stderr)
        return 2

    for label, path in (("Configuration", args.config), ("Statistics", statistics_path)):
        if not path.is_file():
            print(f"Error: {label} file not found: {path}", file=sys.stderr)
            return 1

    try:
        transformations = load_json(args.config)
        raw_statistics = load_json(statistics_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    indexed_statistics, discovered_models, discovered_variants = index_statistics(
        raw_statistics, transformations
    )
    configured_variants = {
        variant["variant_id"]
        for variants in transformations.values()
        for variant in variants
    }
    for variant_id in discovered_variants:
        if variant_id not in configured_variants:
            # Rules such as blue_to_orange are not brute-force parameter sweeps;
            # the statistics contain their single setting directly.
            transformations[variant_id] = [{"variant_id": variant_id}]

    models = ordered_models(discovered_models, args.models)
    if not models:
        print("Error: no configured rule/model statistics were found", file=sys.stderr)
        return 1

    rules = args.rules or list(transformations)
    generated = 0
    skipped = 0

    print(f"Dataset: {args.dataset}")
    print(f"Statistics: {statistics_path}")
    print(f"Models: {', '.join(models)}")
    print(f"Output: {output_dir}")

    for rule in rules:
        variants = transformations.get(rule)
        if not variants:
            print(f"  Skip {rule}: rule is missing or has no settings in {args.config}")
            skipped += 1
            continue

        output_path = output_dir / f"{rule}_venn_metrics.png"
        if plot_rule(rule, variants, models, indexed_statistics, output_path, args.columns):
            print(f"  Saved: {output_path}")
            generated += 1
        else:
            print(f"  Skip {rule}: no matching statistics")
            skipped += 1

    print(f"Generated {generated} plot(s); skipped {skipped} rule(s).")
    return 0 if generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
