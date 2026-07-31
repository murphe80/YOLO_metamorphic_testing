#!/usr/bin/env python3
"""
Plot mean detection-confidence change across transformation settings.

For each model and setting, original and transformed detections are matched
using the same confidence-ordered, one-to-one IoU matching used by the Venn
diagram analysis. The plotted value is the mean change across matched pairs:

    transformed confidence - original confidence

Usage:
    python3 analysis/visualisation/plot_confidence_change.py
    python3 analysis/visualisation/plot_confidence_change.py --dataset test
    python3 analysis/visualisation/plot_confidence_change.py --rules imgColorWheel
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from analysis.utils.detection_matching import (
    load_detections_by_image,
    match_detection_pairs,
)
from utils.model_registry import get_model_by_name, get_model_names

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "brute_force_transformations.json"
RESULTS_ROOT = PROJECT_ROOT / "results"
ANALYSIS_RESULTS_ROOT = PROJECT_ROOT / "analysis" / "results"
DEFAULT_DATASET = "val"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def load_transformations_config(config_path: Path) -> Dict[str, List[dict]]:
    with open(config_path) as config_file:
        return json.load(config_file)


def get_varying_parameter(variants: List[dict]) -> Optional[str]:
    """Return the first parameter whose value varies across settings."""
    parameter_keys = set()
    for variant in variants:
        parameter_keys.update(key for key in variant if key != "variant_id")

    for key in sorted(parameter_keys):
        values = [variant.get(key) for variant in variants]
        if len(set(values)) > 1:
            return key
    return None


def mean_confidence_change(
    model_dir: Path,
    variant_id: str,
    rule: str,
    iou_threshold: float,
) -> Tuple[Optional[float], int]:
    """Return mean confidence change and count across matched detections."""
    original_path = model_dir / "original" / "detections_original_coco.json"
    variant_dir = model_dir / variant_id
    transformed_path = variant_dir / "detections_coco.json"

    # Geometric rules require detections mapped back to original coordinates.
    if rule in {"imgStretch", "imgResolution"}:
        transformed_path = variant_dir / "detections_coco_transformed.json"

    if not original_path.is_file() or not transformed_path.is_file():
        return None, 0

    original_by_image = load_detections_by_image(original_path)
    transformed_by_image = load_detections_by_image(transformed_path)
    changes = []
    shared_image_ids = original_by_image.keys() & transformed_by_image.keys()
    for image_id in shared_image_ids:
        original_detections = original_by_image[image_id]
        transformed_detections = transformed_by_image[image_id]
        pairs = match_detection_pairs(
            original_detections,
            transformed_detections,
            iou_threshold,
        )
        for original_index, transformed_index in pairs:
            original_score = original_detections[original_index].get("score")
            transformed_score = transformed_detections[transformed_index].get("score")
            if original_score is None or transformed_score is None:
                continue
            changes.append(float(transformed_score) - float(original_score))

    if not changes:
        return None, 0
    return sum(changes) / len(changes), len(changes)


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
    spaced = re.sub(r"([A-Z])", r" \1", rule).replace("_", " ").strip()
    return spaced.title()


def format_parameter_name(parameter: str) -> str:
    return parameter.replace("_", " ").title()


def discover_models(results_dir: Path, requested: Optional[List[str]]) -> List[str]:
    if requested:
        return requested

    registered = [
        model for model in get_model_names() if (results_dir / model).is_dir()
    ]
    if registered:
        return registered
    if not results_dir.exists():
        return []
    return sorted(path.name for path in results_dir.iterdir() if path.is_dir())


def collect_series(
    rule: str,
    variants: List[dict],
    models: List[str],
    results_dir: Path,
    iou_threshold: float,
) -> Tuple[str, Dict[str, List[Tuple[float, float]]], List[str], int]:
    """
    Collect (setting, mean confidence change) points for every model.

    Numeric parameter values are used when one parameter varies. Otherwise,
    settings are plotted by index and labelled with their variant IDs.
    """
    parameter = get_varying_parameter(variants)
    use_setting_index = parameter is None
    x_label = "Setting" if use_setting_index else format_parameter_name(parameter)
    tick_labels = [variant["variant_id"] for variant in variants]
    matched_detections = 0
    series: Dict[str, List[Tuple[float, float]]] = {model: [] for model in models}

    for setting_index, variant in enumerate(variants):
        x_value = (
            float(setting_index)
            if use_setting_index
            else float(variant[parameter])
        )
        for model in models:
            change, pair_count = mean_confidence_change(
                results_dir / model,
                variant["variant_id"],
                rule,
                iou_threshold,
            )
            matched_detections += pair_count
            if change is not None:
                series[model].append((x_value, change))

    return (
        x_label,
        series,
        tick_labels if use_setting_index else [],
        matched_detections,
    )


def plot_rule(
    rule: str,
    x_label: str,
    series: Dict[str, List[Tuple[float, float]]],
    setting_labels: List[str],
    output_path: Path,
) -> bool:
    if not any(series.values()):
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    for index, (model, points) in enumerate(sorted(series.items())):
        if not points:
            continue
        x_values, y_values = zip(*points)
        ax.plot(
            x_values,
            y_values,
            marker=MARKERS[index % len(MARKERS)],
            linewidth=2,
            markersize=6,
            color=get_model_color(model),
            label=get_model_label(model),
        )

    ax.axhline(0, color="#555555", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Confidence Change", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{format_rule_name(rule)} — Confidence Change",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    if setting_labels:
        positions = list(range(len(setting_labels)))
        ax.set_xticks(positions, setting_labels, rotation=30, ha="right")
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot confidence change across metamorphic-rule settings"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to brute_force_transformations.json",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=["test", "train_samples", "val"],
        help="Dataset split to plot (default: val)",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Detection results directory (default: results/<dataset>)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Plot directory (default: analysis/results/<dataset>/confidence_change)",
    )
    parser.add_argument(
        "--rules",
        nargs="*",
        help="Rules to plot (default: all rules in the config)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Models to include (default: models found in the results)",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="Minimum IoU for matching detections (default: 0.5)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir or RESULTS_ROOT / args.dataset
    output_dir = (
        args.output_dir
        or ANALYSIS_RESULTS_ROOT / args.dataset / "confidence_change"
    )

    if not args.config.is_file():
        print(f"Config not found: {args.config}")
        sys.exit(1)
    if not results_dir.is_dir():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    transformations = load_transformations_config(args.config)
    models = discover_models(results_dir, args.models)
    if not models:
        print("No models found in results directory.")
        sys.exit(1)

    rules = args.rules if args.rules else sorted(transformations)
    generated = 0
    skipped = 0
    print(f"Dataset: {args.dataset}")
    print(f"Models: {', '.join(models)}")
    print(f"Output: {output_dir}\n")

    for rule in rules:
        variants = transformations.get(rule)
        if variants is None:
            print(f"  Skip {rule}: not in transformations config")
            skipped += 1
            continue

        x_label, series, setting_labels, matched_detections = collect_series(
            rule,
            variants,
            models,
            results_dir,
            args.iou_threshold,
        )
        output_path = output_dir / f"{rule}.png"
        if plot_rule(rule, x_label, series, setting_labels, output_path):
            print(
                f"  Saved: {output_path} "
                f"({matched_detections} matched detections)"
            )
            generated += 1
        else:
            print(f"  Skip {rule}: no matched detections found")
            skipped += 1

    print(f"\nDone: {generated} plot(s) generated, {skipped} skipped.")
    if generated == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
