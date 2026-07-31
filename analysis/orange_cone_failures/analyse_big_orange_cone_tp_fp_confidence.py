#!/usr/bin/env python3
"""Analyse confidence for true and false Big Orange Cone detections.

The script matches class-specific detections to COCO ground truth using
confidence-ordered, one-to-one IoU matching. It consumes the evaluation JSON
files already derived from the detection text files because those files provide
ground-truth image IDs and, for geometric transformations, bounding boxes mapped
back to the original coordinate system.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from analyse_big_orange_cone_confidence import (
    DEFAULT_MODELS,
    MODEL_LABELS,
    descriptive_statistics,
    format_number,
    format_percent,
    markdown_table,
)
from utils.detection_matching import compute_iou


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "results" / "test"
DEFAULT_GROUND_TRUTH = (
    REPOSITORY_ROOT / "data" / "base_images" / "test" / "annotations_coco.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "BIG_ORANGE_CONE_TP_FP_CONFIDENCE_ANALYSIS.md"
GEOMETRIC_FAMILIES = {"imgResolution", "imgStretch"}


@dataclass
class MatchResults:
    """Matched outcomes and diagnostics for a model or transformation family."""

    tp_confidences: list[float] = field(default_factory=list)
    fp_confidences: list[float] = field(default_factory=list)
    matched_ious: list[float] = field(default_factory=list)
    gt_instances: int = 0
    settings: int = 0
    invalid_target_detections: int = 0
    unknown_image_detections: int = 0

    @property
    def tp(self) -> int:
        return len(self.tp_confidences)

    @property
    def fp(self) -> int:
        return len(self.fp_confidences)

    @property
    def fn(self) -> int:
        return self.gt_instances - self.tp

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else math.nan

    @property
    def recall(self) -> float:
        return self.tp / self.gt_instances if self.gt_instances else math.nan

    @property
    def f1(self) -> float:
        if not self.tp:
            return 0.0 if self.fp or self.gt_instances else math.nan
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn)

    def add(self, other: "MatchResults") -> None:
        self.tp_confidences.extend(other.tp_confidences)
        self.fp_confidences.extend(other.fp_confidences)
        self.matched_ious.extend(other.matched_ious)
        self.gt_instances += other.gt_instances
        self.settings += other.settings
        self.invalid_target_detections += other.invalid_target_detections
        self.unknown_image_detections += other.unknown_image_detections


@dataclass
class ModelResults:
    overall: MatchResults = field(default_factory=MatchResults)
    by_family: dict[str, MatchResults] = field(
        default_factory=lambda: defaultdict(MatchResults)
    )
    missing_evaluation_files: list[str] = field(default_factory=list)


def family_name(setting_name: str) -> str:
    return re.sub(r"_\d+$", "", setting_name)


def load_annotations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("annotations"), list):
        return data["annotations"]
    raise ValueError(f"{path}: expected a COCO annotation list or object")


def valid_bbox(bbox: Any) -> bool:
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox)
        and bbox[2] >= 0
        and bbox[3] >= 0
    )


def load_ground_truth(
    path: Path, category_id: int
) -> tuple[dict[int, list[dict[str, Any]]], set[int], int]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ground truth must be a full COCO object")

    image_ids = {int(image["id"]) for image in data.get("images", [])}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    invalid = 0
    for annotation in data.get("annotations", []):
        if int(annotation.get("category_id", -1)) != category_id:
            continue
        if annotation.get("iscrowd", 0):
            continue
        image_id = int(annotation.get("image_id", -1))
        if image_id not in image_ids or not valid_bbox(annotation.get("bbox")):
            invalid += 1
            continue
        by_image[image_id].append(annotation)
    return dict(by_image), image_ids, invalid


def load_target_detections(
    path: Path, category_id: int
) -> tuple[dict[int, list[dict[str, Any]]], int]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    invalid = 0
    for detection in load_annotations(path):
        if int(detection.get("category_id", -1)) != category_id:
            continue
        score = detection.get("score")
        if (
            not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
            or not valid_bbox(detection.get("bbox"))
        ):
            invalid += 1
            continue
        try:
            image_id = int(detection["image_id"])
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue
        by_image[image_id].append(detection)
    return dict(by_image), invalid


def match_image(
    detections: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[float], list[float], list[float]]:
    """Greedily match one class for one image, highest confidence first."""

    ordered = sorted(detections, key=lambda item: item["score"], reverse=True)
    matched_gt: set[int] = set()
    tp_confidences: list[float] = []
    fp_confidences: list[float] = []
    matched_ious: list[float] = []

    for detection in ordered:
        best_gt_index = -1
        best_iou = iou_threshold
        for gt_index, gt_detection in enumerate(ground_truth):
            if gt_index in matched_gt:
                continue
            overlap = compute_iou(detection["bbox"], gt_detection["bbox"])
            if overlap > best_iou:
                best_iou = overlap
                best_gt_index = gt_index

        confidence = float(detection["score"])
        if best_gt_index >= 0:
            matched_gt.add(best_gt_index)
            tp_confidences.append(confidence)
            matched_ious.append(best_iou)
        else:
            fp_confidences.append(confidence)

    return tp_confidences, fp_confidences, matched_ious


def evaluation_file(setting_dir: Path) -> Path:
    family = family_name(setting_dir.name)
    if family in GEOMETRIC_FAMILIES:
        return setting_dir / "detections_for_gt_transformed.json"
    return setting_dir / "detections_for_gt.json"


def analyse_setting(
    detection_path: Path,
    gt_by_image: dict[int, list[dict[str, Any]]],
    gt_image_ids: set[int],
    detection_category_id: int,
    iou_threshold: float,
) -> MatchResults:
    detections_by_image, invalid = load_target_detections(
        detection_path, detection_category_id
    )
    result = MatchResults(
        gt_instances=sum(len(items) for items in gt_by_image.values()),
        settings=1,
        invalid_target_detections=invalid,
    )

    for image_id, detections in detections_by_image.items():
        if image_id not in gt_image_ids:
            result.unknown_image_detections += len(detections)
        tp, fp, ious = match_image(
            detections,
            gt_by_image.get(image_id, []),
            iou_threshold,
        )
        result.tp_confidences.extend(tp)
        result.fp_confidences.extend(fp)
        result.matched_ious.extend(ious)
    return result


def analyse_model(
    model_dir: Path,
    gt_by_image: dict[int, list[dict[str, Any]]],
    gt_image_ids: set[int],
    detection_category_id: int,
    iou_threshold: float,
) -> ModelResults:
    result = ModelResults()
    setting_dirs = sorted(path for path in model_dir.iterdir() if path.is_dir())
    for setting_dir in setting_dirs:
        path = evaluation_file(setting_dir)
        if not path.is_file():
            result.missing_evaluation_files.append(str(path))
            continue
        setting_result = analyse_setting(
            path,
            gt_by_image,
            gt_image_ids,
            detection_category_id,
            iou_threshold,
        )
        result.overall.add(setting_result)
        result.by_family[family_name(setting_dir.name)].add(setting_result)
    return result


def metric_at_threshold(result: MatchResults, threshold: float) -> tuple[int, int, float, float]:
    tp = sum(score >= threshold for score in result.tp_confidences)
    fp = sum(score >= threshold for score in result.fp_confidences)
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / result.gt_instances if result.gt_instances else math.nan
    return tp, fp, precision, recall


def outcome_row(model: str, outcome: str, values: Sequence[float]) -> list[str]:
    stats = descriptive_statistics(values)
    return [
        MODEL_LABELS.get(model, model),
        outcome,
        format_number(stats["n"]),
        format_number(stats.get("mean", math.nan)),
        format_number(stats.get("median", math.nan)),
        format_number(stats.get("population_variance", math.nan)),
        format_number(stats.get("population_std", math.nan)),
        format_number(stats.get("min", math.nan)),
        format_number(stats.get("p05", math.nan)),
        format_number(stats.get("q1", math.nan)),
        format_number(stats.get("q3", math.nan)),
        format_number(stats.get("p95", math.nan)),
        format_number(stats.get("max", math.nan)),
    ]


def build_report(
    results_dir: Path,
    ground_truth_path: Path,
    models: Sequence[str],
    collected: dict[str, ModelResults],
    gt_images: int,
    gt_objects_per_setting: int,
    invalid_gt: int,
    class_id: int,
    detection_category_id: int,
    gt_category_id: int,
    iou_threshold: float,
) -> str:
    lines = [
        "# Big Orange Cone True/False-Positive Confidence Analysis",
        "",
        f"Generated on {date.today().isoformat()} from `{results_dir}` using ground truth "
        f"`{ground_truth_path}`.",
        "",
        "## Scope and matching method",
        "",
        f"Big Orange Cone is model class ID `{class_id}`, mapped to COCO detection category "
        f"ID `{detection_category_id}` and ground-truth category ID `{gt_category_id}`. "
        f"The ground truth contains {gt_objects_per_setting:,} non-crowd Big Orange Cone "
        f"instances across {gt_images:,} images.",
        "",
        "For each model and transformation setting, Big Orange Cone detections were sorted "
        "from highest to lowest confidence and greedily assigned to the unmatched Big "
        f"Orange Cone ground-truth box with the greatest IoU above `{iou_threshold:.2f}`. "
        "A matched detection is a true positive (TP); an unmatched detection is a false "
        "positive (FP). Each ground-truth box can be matched once. Unmatched ground-truth "
        "boxes are false negatives (FN) and have no detection confidence.",
        "",
        "The script uses `detections_for_gt.json`, which is derived from each setting's "
        "detection `.txt` files and supplies ground-truth image IDs. For `imgStretch` and "
        "`imgResolution`, it uses `detections_for_gt_transformed.json`, whose boxes have "
        "been mapped back to original-image coordinates before IoU matching.",
        "",
        "Statistics pool the original output and every metamorphic setting. Ground-truth "
        "instances therefore appear once per evaluated setting. Variance and standard "
        "deviation are population measures (`ddof=0`).",
        "",
        "## Overall classification results",
        "",
    ]

    overall_rows = []
    for model in models:
        result = collected[model].overall
        tp_stats = descriptive_statistics(result.tp_confidences)
        fp_stats = descriptive_statistics(result.fp_confidences)
        overall_rows.append(
            [
                MODEL_LABELS.get(model, model),
                f"{result.settings:,}",
                f"{result.tp:,}",
                f"{result.fp:,}",
                f"{result.fn:,}",
                format_percent(result.precision),
                format_percent(result.recall),
                format_percent(result.f1),
                format_number(tp_stats.get("mean", math.nan)),
                format_number(fp_stats.get("mean", math.nan)),
                format_number(
                    tp_stats.get("mean", math.nan) - fp_stats.get("mean", math.nan)
                ),
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "Model",
                    "Settings",
                    "TP",
                    "FP",
                    "FN",
                    "Precision",
                    "Recall",
                    "F1",
                    "TP mean conf.",
                    "FP mean conf.",
                    "Mean gap",
                ],
                overall_rows,
            ),
            "",
            "`Mean gap` is TP mean confidence minus FP mean confidence; a larger positive "
            "value indicates greater separation between the two groups.",
            "",
            "## TP and FP confidence distributions",
            "",
        ]
    )

    outcome_rows = []
    for model in models:
        result = collected[model].overall
        outcome_rows.append(outcome_row(model, "True positive", result.tp_confidences))
        outcome_rows.append(outcome_row(model, "False positive", result.fp_confidences))
    lines.extend(
        [
            markdown_table(
                [
                    "Model",
                    "Outcome",
                    "N",
                    "Mean",
                    "Median",
                    "Variance",
                    "Std. dev.",
                    "Min",
                    "P05",
                    "P25",
                    "P75",
                    "P95",
                    "Max",
                ],
                outcome_rows,
            ),
            "",
            "## Effect of applying higher confidence thresholds",
            "",
            "TP and FP labels are fixed by the IoU matching above, then detections below "
            "each confidence threshold are removed. Recall remains relative to all "
            "ground-truth instances across the evaluated settings.",
            "",
        ]
    )

    threshold_rows = []
    for model in models:
        result = collected[model].overall
        for threshold in (0.25, 0.50, 0.75, 0.90):
            tp, fp, precision, recall = metric_at_threshold(result, threshold)
            threshold_rows.append(
                [
                    MODEL_LABELS.get(model, model),
                    f"{threshold:.2f}",
                    f"{tp:,}",
                    f"{fp:,}",
                    format_percent(precision),
                    format_percent(recall),
                ]
            )
    lines.extend(
        [
            markdown_table(
                ["Model", "Threshold", "TP retained", "FP retained", "Precision", "Recall"],
                threshold_rows,
            ),
            "",
            "## Results by transformation family",
            "",
        ]
    )

    family_rows = []
    for model in models:
        for family in sorted(collected[model].by_family):
            result = collected[model].by_family[family]
            tp_stats = descriptive_statistics(result.tp_confidences)
            fp_stats = descriptive_statistics(result.fp_confidences)
            family_rows.append(
                [
                    MODEL_LABELS.get(model, model),
                    family,
                    f"{result.settings:,}",
                    f"{result.tp:,}",
                    f"{result.fp:,}",
                    f"{result.fn:,}",
                    format_percent(result.precision),
                    format_percent(result.recall),
                    format_number(tp_stats.get("mean", math.nan)),
                    format_number(fp_stats.get("mean", math.nan)),
                ]
            )
    lines.extend(
        [
            markdown_table(
                [
                    "Model",
                    "Family",
                    "Settings",
                    "TP",
                    "FP",
                    "FN",
                    "Precision",
                    "Recall",
                    "TP mean conf.",
                    "FP mean conf.",
                ],
                family_rows,
            ),
            "",
            "## Matching diagnostics",
            "",
        ]
    )

    diagnostic_rows = []
    for model in models:
        model_result = collected[model]
        result = model_result.overall
        iou_stats = descriptive_statistics(result.matched_ious)
        diagnostic_rows.append(
            [
                MODEL_LABELS.get(model, model),
                f"{result.settings:,}",
                f"{len(model_result.missing_evaluation_files):,}",
                f"{result.invalid_target_detections:,}",
                f"{result.unknown_image_detections:,}",
                format_number(iou_stats.get("mean", math.nan)),
                format_number(iou_stats.get("median", math.nan)),
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "Model",
                    "Settings evaluated",
                    "Missing files",
                    "Invalid target detections",
                    "Unknown-image detections",
                    "Mean matched IoU",
                    "Median matched IoU",
                ],
                diagnostic_rows,
            ),
            "",
            f"Invalid Big Orange Cone ground-truth annotations excluded: `{invalid_gt}`.",
            "",
            "## Summary",
            "",
        ]
    )

    mean_gaps = {}
    for model in models:
        result = collected[model].overall
        tp_stats = descriptive_statistics(result.tp_confidences)
        fp_stats = descriptive_statistics(result.fp_confidences)
        if tp_stats["n"] and fp_stats["n"]:
            mean_gaps[model] = tp_stats["mean"] - fp_stats["mean"]
    if mean_gaps:
        best_gap = max(mean_gaps, key=mean_gaps.get)
        best_precision = max(models, key=lambda item: collected[item].overall.precision)
        best_recall = max(models, key=lambda item: collected[item].overall.recall)
        lines.extend(
            [
                f"- Largest TP-versus-FP mean confidence gap: "
                f"**{MODEL_LABELS.get(best_gap, best_gap)}** ({mean_gaps[best_gap]:.6f}).",
                f"- Highest saved-detection precision: "
                f"**{MODEL_LABELS.get(best_precision, best_precision)}** "
                f"({format_percent(collected[best_precision].overall.precision)}).",
                f"- Highest recall: **{MODEL_LABELS.get(best_recall, best_recall)}** "
                f"({format_percent(collected[best_recall].overall.recall)}).",
            ]
        )
    else:
        lines.append("Insufficient TP/FP observations for a confidence comparison.")

    lines.extend(
        [
            "",
            "A false positive here is class-specific: it can be a duplicate, a poorly "
            "localized box, a background detection, or a Big Orange Cone classification "
            "on an object belonging to another cone class. Inspecting those error types "
            "separately would require an additional class-agnostic matching analysis.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python3 analysis/analyse_big_orange_cone_tp_fp_confidence.py",
            "```",
            "",
            "Use `--iou-threshold` to change the matching criterion. Input locations, "
            "models, and category mappings are also configurable through command-line options.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare true- and false-positive Big Orange Cone confidence scores."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--class-id", type=int, default=2)
    parser.add_argument("--detection-category-id", type=int, default=4)
    parser.add_argument("--gt-category-id", type=int, default=4)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise SystemExit("--iou-threshold must be between 0 and 1")

    results_dir = args.results_dir.resolve()
    ground_truth_path = args.ground_truth.resolve()
    if not ground_truth_path.is_file():
        raise SystemExit(f"Ground truth not found: {ground_truth_path}")
    missing_models = [
        model for model in args.models if not (results_dir / model).is_dir()
    ]
    if missing_models:
        raise SystemExit("Missing model directories: " + ", ".join(missing_models))

    gt_by_image, gt_image_ids, invalid_gt = load_ground_truth(
        ground_truth_path, args.gt_category_id
    )
    gt_objects_per_setting = sum(len(items) for items in gt_by_image.values())
    collected = {
        model: analyse_model(
            results_dir / model,
            gt_by_image,
            gt_image_ids,
            args.detection_category_id,
            args.iou_threshold,
        )
        for model in args.models
    }
    report = build_report(
        results_dir=results_dir,
        ground_truth_path=ground_truth_path,
        models=args.models,
        collected=collected,
        gt_images=len(gt_image_ids),
        gt_objects_per_setting=gt_objects_per_setting,
        invalid_gt=invalid_gt,
        class_id=args.class_id,
        detection_category_id=args.detection_category_id,
        gt_category_id=args.gt_category_id,
        iou_threshold=args.iou_threshold,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")
    for model in args.models:
        result = collected[model].overall
        print(
            f"{MODEL_LABELS.get(model, model)}: TP={result.tp:,}, "
            f"FP={result.fp:,}, FN={result.fn:,}, "
            f"precision={format_percent(result.precision)}, "
            f"recall={format_percent(result.recall)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
