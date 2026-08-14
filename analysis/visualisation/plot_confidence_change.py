#!/usr/bin/env python3
"""Plot confidence behaviour across transformation settings.

For each model and setting, detections are classified against ground truth and
summarised in three groups:

* retained true positives: transformed confidence - original confidence
* lost true positives: original confidence - inference confidence threshold
* persistent false positives: transformed confidence - original confidence

True-positive matching is category-aware and confidence-prioritised. Persistent
false positives are spatially matched between the original and transformed
images using the repository's one-to-one IoU matcher.

Usage:
    python3 analysis/visualisation/plot_confidence_change.py
    python3 analysis/visualisation/plot_confidence_change.py --dataset test
    python3 analysis/visualisation/plot_confidence_change.py --rules imgColorWheel
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from analysis.utils.detection_matching import (
    compute_iou,
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

RETAINED_TP_CHANGE = "retained_tp_change"
LOST_TP_CHANGE = "lost_tp_change"
PERSISTENT_FP_CHANGE = "persistent_fp_change"
LOST_TP_THRESHOLD = 0.25
METRICS = (
    RETAINED_TP_CHANGE,
    LOST_TP_CHANGE,
    PERSISTENT_FP_CHANGE,
)
LINE_METRICS = (
    RETAINED_TP_CHANGE,
    PERSISTENT_FP_CHANGE,
)
METRIC_LABELS = {
    RETAINED_TP_CHANGE: "Retained TP confidence change",
    LOST_TP_CHANGE: "Lost TP confidence decrease (original − 0.25)",
    PERSISTENT_FP_CHANGE: "Persistent FP confidence change",
}


@dataclass(frozen=True)
class ConfidenceSummary:
    """Confidence observations for one model and transformation setting."""

    values: Dict[str, List[float]]


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


def match_detections_to_ground_truth(
    detections: List[dict],
    ground_truth: List[dict],
    iou_threshold: float,
) -> Dict[int, int]:
    """Return detection-to-GT matches using category-aware greedy matching."""
    ordered_detections = sorted(
        enumerate(detections),
        key=lambda item: item[1].get("score", 1.0),
        reverse=True,
    )
    matched_ground_truth = set()
    matches = {}

    for detection_index, detection in ordered_detections:
        best_iou = iou_threshold
        best_gt_index = -1
        detection_category = detection.get("category_id")
        for gt_index, gt_detection in enumerate(ground_truth):
            if gt_index in matched_ground_truth:
                continue
            if detection_category != gt_detection.get("category_id"):
                continue
            overlap = compute_iou(detection["bbox"], gt_detection["bbox"])
            if overlap > best_iou:
                best_iou = overlap
                best_gt_index = gt_index

        if best_gt_index >= 0:
            matches[detection_index] = best_gt_index
            matched_ground_truth.add(best_gt_index)

    return matches


def confidence_summary(
    model_dir: Path,
    variant_id: str,
    rule: str,
    gt_path: Path,
    iou_threshold: float,
) -> ConfidenceSummary:
    """Summarise retained TPs, lost TPs, and persistent FPs."""
    # Use GT-aligned exports throughout. The ordinary metamorphic exports can
    # use generated image IDs that do not necessarily match the GT image IDs.
    original_path = model_dir / "original" / "detections_for_gt.json"
    variant_dir = model_dir / variant_id
    transformed_path = variant_dir / "detections_for_gt.json"

    # Geometric rules require boxes mapped back to original coordinates.
    if rule in {"imgStretch", "imgResolution"}:
        transformed_path = variant_dir / "detections_for_gt_transformed.json"

    values = {metric: [] for metric in METRICS}
    if not original_path.is_file() or not transformed_path.is_file():
        return ConfidenceSummary(
            values=values,
        )

    original_by_image = load_detections_by_image(original_path)
    transformed_by_image = load_detections_by_image(transformed_path)
    gt_by_image = load_detections_by_image(gt_path)
    image_ids = (
        original_by_image.keys()
        | transformed_by_image.keys()
        | gt_by_image.keys()
    )

    for image_id in image_ids:
        original = original_by_image.get(image_id, [])
        transformed = transformed_by_image.get(image_id, [])
        ground_truth = gt_by_image.get(image_id, [])
        original_to_gt = match_detections_to_ground_truth(
            original, ground_truth, iou_threshold
        )
        transformed_to_gt = match_detections_to_ground_truth(
            transformed, ground_truth, iou_threshold
        )

        original_by_gt = {
            gt_index: detection_index
            for detection_index, gt_index in original_to_gt.items()
        }
        transformed_by_gt = {
            gt_index: detection_index
            for detection_index, gt_index in transformed_to_gt.items()
        }

        # A TP is retained only when both images contain a correct detection of
        # the same ground-truth object. A lost TP is represented by the drop
        # from its original score to the inference confidence threshold.
        for gt_index, original_index in original_by_gt.items():
            original_score = original[original_index].get("score")
            if original_score is None:
                continue
            transformed_index = transformed_by_gt.get(gt_index)
            if transformed_index is None:
                values[LOST_TP_CHANGE].append(
                    float(original_score) - LOST_TP_THRESHOLD
                )
                continue
            transformed_score = transformed[transformed_index].get("score")
            if transformed_score is not None:
                values[RETAINED_TP_CHANGE].append(
                    float(transformed_score) - float(original_score)
                )

        # Match only detections already classified as false positives so true
        # positives cannot consume a spatial match intended for an FP pair.
        original_fp_indices = [
            index for index in range(len(original)) if index not in original_to_gt
        ]
        transformed_fp_indices = [
            index
            for index in range(len(transformed))
            if index not in transformed_to_gt
        ]
        original_fps = [original[index] for index in original_fp_indices]
        transformed_fps = [transformed[index] for index in transformed_fp_indices]
        for original_fp_index, transformed_fp_index in match_detection_pairs(
            original_fps, transformed_fps, iou_threshold
        ):
            original_score = original_fps[original_fp_index].get("score")
            transformed_score = transformed_fps[transformed_fp_index].get("score")
            if original_score is not None and transformed_score is not None:
                values[PERSISTENT_FP_CHANGE].append(
                    float(transformed_score) - float(original_score)
                )

    return ConfidenceSummary(
        values=values,
    )


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
    gt_path: Path,
    iou_threshold: float,
) -> Tuple[
    str,
    Dict[str, Dict[str, List[Tuple[float, float]]]],
    Dict[str, List[Tuple[float, List[float]]]],
    List[str],
    Dict[str, int],
]:
    """Collect line means and Lost-TP distributions for every setting."""
    parameter = get_varying_parameter(variants)
    use_setting_index = parameter is None
    x_label = "Setting" if use_setting_index else format_parameter_name(parameter)
    tick_labels = [variant["variant_id"] for variant in variants]
    counts = {metric: 0 for metric in METRICS}
    line_series = {
        metric: {model: [] for model in models}
        for metric in LINE_METRICS
    }
    lost_tp_series = {model: [] for model in models}

    for setting_index, variant in enumerate(variants):
        x_value = (
            float(setting_index)
            if use_setting_index
            else float(variant[parameter])
        )
        for model in models:
            summary = confidence_summary(
                results_dir / model,
                variant["variant_id"],
                rule,
                gt_path,
                iou_threshold,
            )
            for metric in METRICS:
                observations = summary.values[metric]
                counts[metric] += len(observations)
                if metric == LOST_TP_CHANGE:
                    if observations:
                        lost_tp_series[model].append((x_value, observations))
                elif observations:
                    line_series[metric][model].append(
                        (x_value, sum(observations) / len(observations))
                    )

    return (
        x_label,
        line_series,
        lost_tp_series,
        tick_labels if use_setting_index else [],
        counts,
    )


def plot_rule(
    rule: str,
    x_label: str,
    series: Dict[str, Dict[str, List[Tuple[float, float]]]],
    setting_labels: List[str],
    output_path: Path,
) -> bool:
    if not any(points for metric in LINE_METRICS for points in series[metric].values()):
        return False

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True)
    model_order = sorted(next(iter(series.values())))
    for axis, metric in zip(axes, LINE_METRICS):
        for index, model in enumerate(model_order):
            points = series[metric][model]
            if not points:
                continue
            x_values, y_values = zip(*points)
            axis.plot(
                x_values,
                y_values,
                marker=MARKERS[index % len(MARKERS)],
                linewidth=2,
                markersize=6,
                color=get_model_color(model),
                label=get_model_label(model),
            )

        axis.axhline(
            0, color="#555555", linewidth=1, linestyle="--", alpha=0.7
        )
        axis.set_xlabel(x_label, fontsize=10, fontweight="bold")
        axis.set_ylabel(METRIC_LABELS[metric], fontsize=10, fontweight="bold")
        axis.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
        if setting_labels:
            positions = list(range(len(setting_labels)))
            axis.set_xticks(positions, setting_labels, rotation=30, ha="right")
        if not any(series[metric].values()):
            axis.text(
                0.5,
                0.5,
                "No observations",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#666666",
            )

    handles, labels = [], []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=min(len(labels), 6),
            framealpha=0.9,
            fontsize=9,
        )
    fig.suptitle(
        f"{format_rule_name(rule)} — Retained TP and Persistent FP Confidence",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def _violin_width(x_values: List[float]) -> float:
    """Return a readable violin width for numeric or indexed settings."""
    if len(x_values) < 2:
        return 0.6
    unique_values = sorted(set(x_values))
    return min(
        right - left for left, right in zip(unique_values, unique_values[1:])
    ) * 0.65


def plot_lost_tp_rule(
    rule: str,
    model: str,
    x_label: str,
    series: List[Tuple[float, List[float]]],
    setting_labels: List[str],
    output_path: Path,
) -> bool:
    """Plot Lost-TP confidence-decrease distributions for one model and rule."""
    if not series:
        return False

    x_values, distributions = zip(*series)
    fig, axis = plt.subplots(figsize=(10, 6))
    violin_parts = axis.violinplot(
        distributions,
        positions=x_values,
        widths=_violin_width(list(x_values)),
        showmeans=False,
        showmedians=True,
    )
    color = get_model_color(model)
    for body in violin_parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        violin_parts[key].set_color(color)

    axis.axhline(0, color="#555555", linewidth=1, linestyle="--", alpha=0.7)
    axis.set_xlabel(x_label, fontsize=11, fontweight="bold")
    axis.set_ylabel(METRIC_LABELS[LOST_TP_CHANGE], fontsize=11, fontweight="bold")
    axis.set_title(
        f"{format_rule_name(rule)} — {get_model_label(model)} Lost TP Confidence Decrease",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    axis.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    if setting_labels:
        positions = list(range(len(setting_labels)))
        axis.set_xticks(positions, setting_labels, rotation=30, ha="right")
    else:
        axis.set_xticks(x_values)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TP and FP confidence across metamorphic-rule settings"
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
        choices=["test", "train_samples", "val", "competition", "comp_labelled"],
        help="Dataset split to plot (default: val)",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Detection results directory (default: results/<dataset>)",
    )
    parser.add_argument(
        "--gt_path",
        type=Path,
        default=None,
        help="Ground-truth COCO JSON (default: data/base_images/<dataset>/annotations_coco.json)",
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
    gt_path = (
        args.gt_path
        or PROJECT_ROOT
        / "data"
        / "base_images"
        / args.dataset
        / "annotations_coco.json"
    )
    output_dir = (
        args.output_dir
        or ANALYSIS_RESULTS_ROOT / args.dataset / "confidence_change"
    )

    if not 0 <= args.iou_threshold <= 1:
        parser.error("--iou_threshold must be between 0 and 1")
    if not args.config.is_file():
        print(f"Config not found: {args.config}")
        sys.exit(1)
    if not results_dir.is_dir():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)
    if not gt_path.is_file():
        print(f"Ground truth not found: {gt_path}")
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
    print(f"Ground truth: {gt_path}")
    print(f"Output: {output_dir}\n")

    for rule in rules:
        variants = transformations.get(rule)
        if variants is None:
            print(f"  Skip {rule}: not in transformations config")
            skipped += 1
            continue

        x_label, line_series, lost_tp_series, setting_labels, counts = collect_series(
            rule,
            variants,
            models,
            results_dir,
            gt_path,
            args.iou_threshold,
        )
        output_path = output_dir / f"{rule}.png"
        if plot_rule(rule, x_label, line_series, setting_labels, output_path):
            print(
                f"  Saved: {output_path} "
                f"(retained TP={counts[RETAINED_TP_CHANGE]}, "
                f"lost TP={counts[LOST_TP_CHANGE]}, "
                f"persistent FP={counts[PERSISTENT_FP_CHANGE]})"
            )
            generated += 1
        else:
            print(f"  Skip {rule}: no retained TP or persistent FP detections found")
            skipped += 1

        for model in models:
            lost_output_path = output_dir / "lost_TP" / model / f"{rule}.png"
            if plot_lost_tp_rule(
                rule,
                model,
                x_label,
                lost_tp_series[model],
                setting_labels,
                lost_output_path,
            ):
                lost_count = sum(
                    len(distribution)
                    for _, distribution in lost_tp_series[model]
                )
                print(f"  Saved: {lost_output_path} ({lost_count} lost TP(s))")
                generated += 1
            else:
                print(f"  Skip {model}/{rule}: no lost TP detections found")
                skipped += 1

    print(f"\nDone: {generated} plot(s) generated, {skipped} skipped.")
    if generated == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
