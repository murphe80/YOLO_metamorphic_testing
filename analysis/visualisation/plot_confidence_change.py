#!/usr/bin/env python3
"""Plot confidence behaviour across transformation settings.

For each model and setting, detections are classified against ground truth and
summarised in four groups:

* retained true positives: transformed confidence - original confidence
* lost true positives: original confidence - inference confidence threshold
* persistent false positives: transformed confidence - original confidence
* new false positives: transformed confidence for FPs without an original FP match

The confidence-change line plots show retained true positives only.
Lost TP and new FP distributions are plotted separately.
Plots omit titles and legends; the shared model legend is saved as ``legend.pdf``.

True-positive matching is category-aware and confidence-prioritised. Persistent
false positives are spatially matched between the original and transformed
images using the repository's one-to-one IoU matcher.
Cone-colour rules share one plot with a categorical x axis. Transformed TPs
are evaluated against colour-adjusted GT, retaining the original GT object IDs.

Usage:
    python3 analysis/visualisation/plot_confidence_change.py
    python3 analysis/visualisation/plot_confidence_change.py --dataset test
    python3 analysis/visualisation/plot_confidence_change.py --rules imgColorWheel
    python3 analysis/visualisation/plot_confidence_change.py --dataset test --rules cone_colour
    python3 analysis/visualisation/plot_confidence_change.py --dataset test --new-fp-only
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from analysis.utils.detection_matching import (
    compute_iou,
    load_detections_by_image,
    match_detection_pairs,
)
from analysis.utils.analyse_cone_colours import (
    TRANSFORMATION_MAP,
    update_ground_truth_cone_color,
)
from utils.model_registry import get_model_by_name, get_model_names
from analysis.visualisation.model_legend import sorted_model_legend
from analysis.visualisation.parameter_settings import configure_resolution_axis
from analysis.utils.color_wheel import (
    metadata as hue_metadata, grouped, remap_reference, select_predictions,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "brute_force_transformations.json"
RESULTS_ROOT = PROJECT_ROOT / "results"
ANALYSIS_RESULTS_ROOT = PROJECT_ROOT / "analysis" / "results" / "rq1"
DEFAULT_DATASET = "val"
CONE_COLOUR_RULE = "cone_colour"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

RETAINED_TP_CHANGE = "retained_tp_change"
LOST_TP_CHANGE = "lost_tp_change"
PERSISTENT_FP_CHANGE = "persistent_fp_change"
NEW_FP_CONFIDENCE = "new_fp_confidence"
LOST_TP_THRESHOLD = 0.25
METRICS = (
    RETAINED_TP_CHANGE,
    LOST_TP_CHANGE,
    PERSISTENT_FP_CHANGE,
    NEW_FP_CONFIDENCE,
)
LINE_METRICS = (
    RETAINED_TP_CHANGE,
    PERSISTENT_FP_CHANGE,
)
METRIC_LABELS = {
    RETAINED_TP_CHANGE: "Retained TP confidence change",
    LOST_TP_CHANGE: "Lost TP confidence decrease (original − 0.25)",
    PERSISTENT_FP_CHANGE: "Persistent FP confidence change",
    NEW_FP_CONFIDENCE: "New FP confidence",
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
    source_mapping: Optional[dict] = None,
) -> ConfidenceSummary:
    """Summarise TP changes, persistent FP changes, and new FP confidences."""
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
    transformed_gt_by_image = gt_by_image
    if rule == "imgColorWheel":
        if source_mapping is None:
            raise ValueError("Color-wheel confidence requires a separate source mapping")
        original_data = json.loads(original_path.read_text())
        transformed_data = json.loads(transformed_path.read_text())
        gt_data = json.loads(gt_path.read_text())
        original_by_image = grouped(select_predictions(original_data, gt_data, source_mapping, transformed=False))
        transformed_by_image = grouped(select_predictions(transformed_data, gt_data, source_mapping, fallback=original_data))
        gt_by_image = grouped(remap_reference(gt_data, source_mapping, remap=False))
        transformed_gt_by_image = grouped(remap_reference(gt_data, source_mapping))
    if variant_id in TRANSFORMATION_MAP:
        # Only transformed detections use recoloured GT: original TP/FP
        # membership must still be established against the original labels.
        with TemporaryDirectory(prefix="cone_colour_confidence_") as directory:
            expected_gt = Path(directory) / "gt.json"
            update_ground_truth_cone_color(gt_path, variant_id, expected_gt)
            transformed_gt_by_image = load_detections_by_image(expected_gt)
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
            transformed, transformed_gt_by_image.get(image_id, []), iou_threshold
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
        fp_pairs = match_detection_pairs(
            original_fps, transformed_fps, iou_threshold
        )
        for original_fp_index, transformed_fp_index in fp_pairs:
            original_score = original_fps[original_fp_index].get("score")
            transformed_score = transformed_fps[transformed_fp_index].get("score")
            if original_score is not None and transformed_score is not None:
                values[PERSISTENT_FP_CHANGE].append(
                    float(transformed_score) - float(original_score)
                )
        persistent_indices = {transformed_index for _, transformed_index in fp_pairs}
        for index, detection in enumerate(transformed_fps):
            if index not in persistent_indices and detection.get("score") is not None:
                values[NEW_FP_CONFIDENCE].append(float(detection["score"]))

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
    export_records: Optional[list] = None,
    new_fp_series: Optional[Dict[str, List[Tuple[float, List[float]]]]] = None,
) -> Tuple[
    str,
    Dict[str, Dict[str, List[Tuple[float, float]]]],
    Dict[str, List[Tuple[float, List[float]]]],
    List[str],
    Dict[str, int],
]:
    """Collect line means, Lost-TP distributions, and optional new-FP distributions."""
    hue_records = []
    if rule == "imgColorWheel":
        hue_records = [{**hue_metadata(v["variant_id"]), "class_shifts": {}} for v in variants]
        variants = [
            {"variant_id": record["variant_id"], "mapping": mapping,
             "label": f"{record['hue_rotation']}\n{mapping['mapping']}"}
            for record in hue_records for mapping in record["included_mappings"]
        ]
    parameter = None if rule == "imgColorWheel" else get_varying_parameter(variants)
    use_setting_index = parameter is None
    x_label = "Setting" if use_setting_index else format_parameter_name(parameter)
    tick_labels = [variant["variant_id"] for variant in variants]
    if rule == "imgColorWheel":
        x_label = "Hue rotation / source → target (separate cohorts)"
        tick_labels = [v["label"] for v in variants]
    if rule == CONE_COLOUR_RULE:
        x_label = "Cone colour rule"
        tick_labels = [label.replace("_to_", " → ").title() for label in tick_labels]
    if export_records is not None:
        export_records.extend(hue_records)
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
                **({"source_mapping": variant["mapping"]} if rule == "imgColorWheel" else {}),
            )
            if rule == "imgColorWheel" and export_records is not None:
                record = next(r for r in export_records if r["variant_id"] == variant["variant_id"])
                shift = record.setdefault("class_shifts", {}).setdefault(
                    variant["mapping"]["key"], {**variant["mapping"], "models": {}}
                )
                shift["models"][model] = {
                    metric: {"count": len(values), "mean": sum(values) / len(values) if values else None}
                    for metric, values in summary.values.items()
                }
            for metric in METRICS:
                observations = summary.values[metric]
                counts[metric] += len(observations)
                if metric == NEW_FP_CONFIDENCE:
                    if new_fp_series is not None:
                        new_fp_series.setdefault(model, []).append((x_value, observations))
                elif metric == LOST_TP_CHANGE:
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


def plot_shared_legend(models: List[str], output_path: Path) -> None:
    """Save the existing model legend without any other text or plot elements."""
    handles = [
        Line2D(
            [], [], marker=MARKERS[index % len(MARKERS)], linewidth=2,
            markersize=6, color=get_model_color(model), label=get_model_label(model),
        )
        for index, model in enumerate(sorted(models))
    ]
    fig = plt.figure(figsize=(6, 1))
    legend = fig.legend(
        *sorted_model_legend(handles, [handle.get_label() for handle in handles]),
        loc="center", ncol=min(len(handles), 6), framealpha=0.9, fontsize=9,
    )
    fig.canvas.draw()
    bounds = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches=bounds.padded(0.05), facecolor="white")
    plt.close(fig)


def plot_rule(
    rule: str,
    x_label: str,
    series: Dict[str, Dict[str, List[Tuple[float, float]]]],
    setting_labels: List[str],
    output_path: Path,
) -> bool:
    retained_series = series[RETAINED_TP_CHANGE]
    if not any(retained_series.values()):
        return False

    fig, axis = plt.subplots(figsize=(14 if rule == "imgColorWheel" else 10, 7))
    for index, model in enumerate(sorted(retained_series)):
        points = retained_series[model]
        if not points:
            continue
        x_values, y_values = zip(*points)
        axis.plot(
            x_values,
            y_values,
            marker=MARKERS[index % len(MARKERS)],
            linewidth=2,
            linestyle="none" if rule == "imgColorWheel" else "-",
            markersize=6,
            color=get_model_color(model),
            label=get_model_label(model),
        )

    axis.axhline(0, color="#555555", linewidth=1, linestyle="--", alpha=0.7)
    axis.set_xlabel(x_label, fontsize=13, fontweight="bold")
    axis.set_ylabel(METRIC_LABELS[RETAINED_TP_CHANGE], fontsize=13, fontweight="bold")
    axis.tick_params(axis="both", labelsize=12)
    axis.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    if setting_labels:
        positions = list(range(len(setting_labels)))
        axis.set_xticks(positions, setting_labels, rotation=30, ha="right")

    configure_resolution_axis(axis, rule, [x for points in retained_series.values() for x, _ in points])
    fig.tight_layout()

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
    fig, axis = plt.subplots(figsize=(18 if rule == "imgColorWheel" else 10, 7))
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
    axis.set_xlabel(x_label, fontsize=13, fontweight="bold")
    axis.set_ylabel(METRIC_LABELS[LOST_TP_CHANGE], fontsize=13, fontweight="bold")
    axis.tick_params(axis="both", labelsize=12)
    axis.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    if setting_labels:
        positions = list(range(len(setting_labels)))
        axis.set_xticks(positions, setting_labels, rotation=30, ha="right")
    else:
        axis.set_xticks(x_values)
    configure_resolution_axis(axis, rule, x_values)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def plot_new_fp_rule(
    rule: str,
    x_label: str,
    series: Dict[str, List[Tuple[float, List[float]]]],
    setting_labels: List[str],
    output_path: Path,
) -> bool:
    """Plot absolute new-FP confidence distributions, with one panel per model."""
    if not series:
        return False
    models = list(series)
    columns = min(2, len(models))
    rows = (len(models) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, squeeze=False, sharey=True,
                             figsize=(24 if rule == "imgColorWheel" else 16, 5 * rows),
                             constrained_layout=True)
    for axis in axes.flat:
        axis.set_visible(False)
    for axis, model in zip(axes.flat, models):
        axis.set_visible(True)
        settings = series[model]
        populated = [(x, scores) for x, scores in settings if scores]
        if populated:
            positions, distributions = zip(*populated)
            axis.boxplot(distributions, positions=positions,
                         widths=_violin_width([x for x, _ in settings]),
                         showmeans=False, patch_artist=True, manage_ticks=False,
                         boxprops={"facecolor": get_model_color(model), "alpha": .55},
                         medianprops={"color": "black", "linewidth": 1.5},
                         flierprops={"marker": "o", "markersize": 2, "alpha": .4})
        else:
            axis.text(.5, .5, "No new FP observations", transform=axis.transAxes,
                      ha="center", va="center")
        positions = [x for x, _ in settings]
        if setting_labels:
            axis.set_xticks(range(len(setting_labels)), setting_labels, rotation=35, ha="right")
        else:
            axis.set_xticks(positions)
        configure_resolution_axis(axis, rule, positions)
        axis.set_ylim(0, 1)
        axis.set_ylabel("New FP confidence", fontsize=13, fontweight="bold")
        axis.set_xlabel(x_label, fontsize=13, fontweight="bold")
        axis.tick_params(axis="both", labelsize=12)
        axis.grid(axis="y", alpha=.25)
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
        help="Plot directory (default: analysis/results/rq1/<dataset>/confidence_change)",
    )
    parser.add_argument(
        "--rules",
        nargs="*",
        help="Rules to plot, including cone_colour or individual colour rules (default: all available rules)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Models to include (default: models found in the results)",
    )
    parser.add_argument(
        "--new-fp-only", action="store_true",
        help="Generate only new FP confidence plots and data under <output_dir>/new_FP",
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

    # Treat the available colour transformations as settings of one rule,
    # without adding categorical values to the numeric brute-force config.
    requested_colours = [name for name in (args.rules or []) if name in TRANSFORMATION_MAP]
    colour_names = requested_colours or list(TRANSFORMATION_MAP)
    colour_variants = [
        {"variant_id": name}
        for name in colour_names
        if any((results_dir / model / name / "detections_for_gt.json").is_file()
               for model in models)
    ]
    if colour_variants:
        transformations[CONE_COLOUR_RULE] = colour_variants
    rules = list(dict.fromkeys(
        CONE_COLOUR_RULE if name in TRANSFORMATION_MAP else name
        for name in (args.rules or sorted(transformations))
    ))
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

        hue_export = []
        new_fp_series = {model: [] for model in models}
        x_label, line_series, lost_tp_series, setting_labels, counts = collect_series(
            rule,
            variants,
            models,
            results_dir,
            gt_path,
            args.iou_threshold,
            hue_export,
            new_fp_series,
        )
        new_fp_output = output_dir / "new_FP" / f"{rule}.pdf"
        if plot_new_fp_rule(rule, x_label, new_fp_series, setting_labels, new_fp_output):
            new_fp_output.with_suffix(".json").write_text(json.dumps({
                "dataset": args.dataset, "rule": rule, "iou_threshold": args.iou_threshold,
                "definition": "transformed FP with no spatial match to an original FP; "
                              "FP membership uses category-aware GT matching; FP pairs ignore category",
                "confidence": "absolute transformed detection score, not a confidence change",
                "gt_source": str(gt_path.resolve()), "results_source": str(results_dir.resolve()),
                "x_label": x_label, "setting_labels": setting_labels,
                "models": {model: [dict(setting=x, count=len(scores), scores=scores,
                                         mean=sum(scores) / len(scores) if scores else None)
                                   for x, scores in settings]
                           for model, settings in new_fp_series.items()},
                "hue_mappings": hue_export,
            }, indent=2, allow_nan=False) + "\n")
            print(f"  Saved: {new_fp_output} (new FP={counts[NEW_FP_CONFIDENCE]})", flush=True)
            generated += 1
        if args.new_fp_only:
            continue
        if rule == "imgColorWheel":
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "imgColorWheel.json").write_text(json.dumps(hue_export, indent=2) + "\n")
        output_path = output_dir / f"{rule}.pdf"
        if plot_rule(rule, x_label, line_series, setting_labels, output_path):
            print(
                f"  Saved: {output_path} "
                f"(retained TP={counts[RETAINED_TP_CHANGE]}, "
                f"lost TP={counts[LOST_TP_CHANGE]}, "
                f"persistent FP={counts[PERSISTENT_FP_CHANGE]})"
            )
            generated += 1
        else:
            print(f"  Skip {rule}: no retained TP detections found")
            skipped += 1

        # Colour rules intentionally produce one combined figure only.
        if rule == CONE_COLOUR_RULE:
            continue

        for model in models:
            lost_output_path = output_dir / "lost_true_positives" / model / f"{rule}.pdf"
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

    if generated:
        legend_output = output_dir / "legend.pdf"
        plot_shared_legend(models, legend_output)
        print(f"  Saved: {legend_output}")

    print(f"\nDone: {generated} plot(s) generated, {skipped} skipped.")
    if generated == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
