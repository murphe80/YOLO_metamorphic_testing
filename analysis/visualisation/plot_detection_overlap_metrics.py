#!/usr/bin/env python3
"""Plot detection-overlap metrics across up to five evenly spaced rule settings.

One image is generated per metamorphic rule. Each image contains one subplot
per YOLO model, and each subplot shows grouped bars for the configured rule
settings (including both range endpoints) and five retained, lost, or new
detection metrics. Colour-wheel plots include every reviewed class-changing
mapping, retain separate source cohorts, and include a blue-source-only plot.
Each labelled bar
is stacked by small/medium/large object size in transformed-image coordinates,
with its original image-normalized total rate preserved. Colours identify
metrics and light/medium/dark shades identify size classes.

By default, metrics are read from::

    analysis/results/rq1/<dataset>/detection_overlap_metrics/detection_overlap_metrics.json

and plots are written to::

    analysis/results/rq1/<dataset>/detection_overlap_metrics/plots/

Examples:
    python3 analysis/visualisation/plot_detection_overlap_metrics.py
    python3 analysis/visualisation/plot_detection_overlap_metrics.py --dataset val
    python3 analysis/visualisation/plot_detection_overlap_metrics.py --rules imgDarkBright
    python3 analysis/visualisation/plot_detection_overlap_metrics.py --models yolov5 yolov8
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
from matplotlib.patches import Patch
import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from utils.model_registry import get_model_display_name
from analysis.utils.color_wheel import metadata as hue_metadata
from analysis.utils.detection_matching import SIZE_BINS
from analysis.visualisation.parameter_settings import displayed_overlap_settings

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "brute_force_transformations.json"
RESULTS_ROOT = PROJECT_ROOT / "analysis" / "results" / "rq1"
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


def size_color(metric_color, size: str):
    """Keep each metric's hue while making size classes visibly distinct."""
    rgb = np.asarray(metric_color[:3])
    if size == "small":
        return rgb * 0.35 + 0.65
    if size == "large":
        return rgb * 0.5
    return rgb


def metric_size_contributions(record: dict, metric: str) -> List[float]:
    """Return additive stack heights; never infer a size from an aggregate rate."""
    value = record.get(metric)
    if value is None:
        return [np.nan] * len(SIZE_BINS)
    try:
        parts = record["size_breakdown"]["metrics"][metric]["rate_contributions"]
        values = [float(parts[size]) for size in SIZE_BINS]
    except KeyError as error:
        raise ValueError("Size breakdown missing; rerun calculate_detection_overlap_metrics.py first") from error
    if any(not math.isfinite(v) or v < 0 for v in values) or not math.isclose(
        sum(values), float(value), rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(f"Size contributions do not sum to {metric}: {values} versus {value}")
    return values


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
    if variants and "mapping_label" in variants[0]:
        return [v["mapping_label"] for v in variants], ""
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


def color_wheel_cohorts(variants, models, statistics):
    """Expand all included class changes without subsampling hue settings."""
    cohorts = []
    cohort_statistics = {}
    for variant in variants:
        record = hue_metadata(variant["variant_id"])
        for mapping in record["included_mappings"]:
            if mapping["source_category_id"] == mapping["target_category_id"]:
                continue
            key = f"{variant['variant_id']}/{mapping['key']}"
            cohorts.append({
                "variant_id": key,
                "mapping_key": mapping["key"],
                "mapping_label": f"{record['hue_rotation']}\n{mapping['mapping']}",
            })
            for model in models:
                original = statistics.get((model, variant["variant_id"]), {})
                values = original.get("class_shifts", {}).get(mapping["key"])
                if values is not None:
                    cohort_statistics[(model, key)] = values
    return cohorts, cohort_statistics


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

    column_count = min(2 if rule == "imgColorWheel" else columns, len(available_models))
    row_count = math.ceil(len(available_models) / column_count)
    figure_width = max(10.0, 1.35 * len(variants)) * column_count
    figure_height = 7.0 * row_count + 1.0
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

    for subplot_index, model in enumerate(available_models):
        ax = axes.flat[subplot_index]
        bar_positions, bar_labels = [], []
        for metric_index, metric in enumerate(METRICS):
            offset = (metric_index - (len(METRICS) - 1) / 2) * bar_width
            positions = x + offset
            contributions = np.array([
                metric_size_contributions(statistics.get((model, variant["variant_id"]), {}), metric)
                for variant in variants
            ])
            bottom = np.zeros(len(variants))
            for size_index, size in enumerate(SIZE_BINS):
                heights = contributions[:, size_index]
                ax.bar(positions, heights, bottom=bottom, width=bar_width,
                       color=size_color(colors[metric_index], size),
                       edgecolor="white", linewidth=0.4)
                bottom += heights
            bar_positions.extend(positions)
            bar_labels.extend([METRIC_LABELS[metric]] * len(variants))

        ax.set_title(get_model_display_name(model), fontsize=12, fontweight="bold")
        ax.set_xticks(bar_positions, bar_labels, rotation=90, fontsize=8)
        ax.tick_params(axis="x", length=0, pad=4)
        # A second row labels the transformation setting beneath each group.
        settings_axis = ax.secondary_xaxis("bottom")
        settings_axis.set_xticks(x, tick_labels, fontsize=9)
        settings_axis.tick_params(axis="x", length=0, pad=90)
        settings_axis.spines["bottom"].set_visible(False)
        settings_axis.set_xlabel(x_label, fontsize=10, fontweight="bold", labelpad=8)
        for boundary in x[:-1] + 0.5:
            ax.axvline(boundary, color="#DDDDDD", linewidth=0.7, zorder=0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)
        if subplot_index % column_count == 0:
            ax.set_ylabel("Rate", fontsize=10, fontweight="bold")

    for unused_index in range(len(available_models), row_count * column_count):
        axes.flat[unused_index].set_visible(False)

    source_only = any(
        statistics.get((model, variant["variant_id"]), {}).get("cohort") == "source_only"
        for model in available_models for variant in variants
    )
    fig.suptitle(
        f"{format_name(rule)} : Detection Overlap Metrics Across Rule Settings"
        + (" (source-only cohort)" if source_only else ""),
        fontsize=15,
        fontweight="bold",
    )
    fig.legend(
        [Patch(facecolor=size_color((0.5, 0.5, 0.5), size), edgecolor="white",
               label=size.title()) for size in SIZE_BINS],
        ["Small (light)", "Medium", "Large (dark)"],
        loc="lower center",
        ncol=len(SIZE_BINS),
        title="Object size in transformed image",
        framealpha=0.9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), h_pad=3.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot detection-overlap metrics with one subplot per YOLO model."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset name (default: test)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--statistics",
        type=Path,
        default=None,
        help="Metrics JSON (default: analysis/results/rq1/<dataset>/detection_overlap_metrics/detection_overlap_metrics.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: analysis/results/rq1/<dataset>/detection_overlap_metrics/plots)",
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
        RESULTS_ROOT / args.dataset / "detection_overlap_metrics" / "detection_overlap_metrics.json"
    )
    output_dir = args.output_dir or (
        RESULTS_ROOT / args.dataset / "detection_overlap_metrics" / "plots"
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

        rule_statistics = indexed_statistics
        if rule == "imgColorWheel":
            variants, rule_statistics = color_wheel_cohorts(variants, models, indexed_statistics)
        else:
            variants = displayed_overlap_settings(rule, variants)
        plots = [(rule, variants)]
        if rule == "imgColorWheel":
            blue_variants = [
                variant
                for mapping_key in ("blue_to_yellow", "blue_to_orange")
                for variant in variants
                if variant["mapping_key"] == mapping_key
            ]
            plots.append(("imgColorWheel_blue_to_yellow_and_orange", blue_variants))
        for name, plot_variants in plots:
            output_path = output_dir / f"{name}_detection_overlap_metrics.pdf"
            if plot_rule(rule, plot_variants, models, rule_statistics, output_path, args.columns):
                print(f"  Saved: {output_path}")
                generated += 1
            else:
                print(f"  Skip {name}: no matching statistics")
                skipped += 1

    print(f"Generated {generated} plot(s); skipped {skipped} rule(s).")
    return 0 if generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
