#!/usr/bin/env python3
"""Analyse Big Orange Cone confidence scores across YOLO model outputs.

Detection text files are expected to contain one detection per line in the
repository's standard format::

    class_id confidence x1 y1 x2 y2

By default, the script recursively scans results/test/{model}/ for the four
models used in the project and writes a Markdown report in the repository root.
Only text detections are read; derived JSON/COCO exports are intentionally not
included because they duplicate the same detections.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "results" / "test"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "BIG_ORANGE_CONE_CONFIDENCE_ANALYSIS.md"
DEFAULT_MODELS = ("yolov5", "yolov8", "yolov12", "yolov26")
MODEL_LABELS = {
    "yolov5": "YOLOv5",
    "yolov8": "YOLOv8",
    "yolov12": "YOLOv12",
    "yolov26": "YOLOv26",
}


@dataclass
class ModelData:
    """Collected confidence values and input-quality diagnostics for a model."""

    confidences: list[float] = field(default_factory=list)
    confidences_by_family: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    file_mean_confidences: list[float] = field(default_factory=list)
    files_scanned: int = 0
    empty_files: int = 0
    files_with_detections: int = 0
    files_with_target: int = 0
    valid_detection_rows: int = 0
    malformed_rows: int = 0
    out_of_range_rows: int = 0


def percentile(sorted_values: Sequence[float], percent: float) -> float:
    """Return a linearly interpolated percentile for already sorted values."""

    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def descriptive_statistics(values: Sequence[float]) -> dict[str, float | int]:
    """Compute detection-level descriptive statistics."""

    if not values:
        return {"n": 0}

    ordered = sorted(values)
    n = len(ordered)
    q1 = percentile(ordered, 25)
    q3 = percentile(ordered, 75)
    sample_variance = statistics.variance(ordered) if n > 1 else math.nan
    sample_std = statistics.stdev(ordered) if n > 1 else math.nan
    return {
        "n": n,
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "population_variance": statistics.pvariance(ordered),
        "population_std": statistics.pstdev(ordered),
        "sample_variance": sample_variance,
        "sample_std": sample_std,
        "min": ordered[0],
        "p05": percentile(ordered, 5),
        "q1": q1,
        "q3": q3,
        "p95": percentile(ordered, 95),
        "max": ordered[-1],
        "iqr": q3 - q1,
        "at_least_050": sum(value >= 0.50 for value in ordered) / n,
        "at_least_075": sum(value >= 0.75 for value in ordered) / n,
        "at_least_090": sum(value >= 0.90 for value in ordered) / n,
    }


def setting_family(relative_path: Path) -> str:
    """Map a setting directory such as imgMotionBlur_3 to imgMotionBlur."""

    setting = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"
    return re.sub(r"_\d+$", "", setting)


def collect_model_data(model_dir: Path, target_class_id: int) -> ModelData:
    """Read every detection text file below one model directory."""

    data = ModelData()
    for detection_path in sorted(model_dir.rglob("*.txt")):
        data.files_scanned += 1
        relative_path = detection_path.relative_to(model_dir)
        family = setting_family(relative_path)
        file_has_nonempty_row = False
        file_has_valid_detection = False
        file_target_confidences: list[float] = []

        with detection_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                file_has_nonempty_row = True
                fields = stripped.split()
                if len(fields) != 6:
                    data.malformed_rows += 1
                    continue
                try:
                    class_id = int(fields[0])
                    confidence = float(fields[1])
                    # Validate coordinates as well, so only complete detection rows count.
                    coordinates = [float(value) for value in fields[2:]]
                except ValueError:
                    data.malformed_rows += 1
                    continue
                if not math.isfinite(confidence) or not all(
                    math.isfinite(value) for value in coordinates
                ):
                    data.malformed_rows += 1
                    continue
                if not 0.0 <= confidence <= 1.0:
                    data.out_of_range_rows += 1
                    continue

                file_has_valid_detection = True
                data.valid_detection_rows += 1
                if class_id == target_class_id:
                    file_target_confidences.append(confidence)
                    data.confidences.append(confidence)
                    data.confidences_by_family[family].append(confidence)

        if not file_has_nonempty_row:
            data.empty_files += 1
        if file_has_valid_detection:
            data.files_with_detections += 1
        if file_target_confidences:
            data.files_with_target += 1
            data.file_mean_confidences.append(
                statistics.fmean(file_target_confidences)
            )

    return data


def format_number(value: float | int, digits: int = 6) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if math.isnan(value):
        return "N/A"
    return f"{value:.{digits}f}"


def format_percent(value: float) -> str:
    if math.isnan(value):
        return "N/A"
    return f"{100.0 * value:.2f}%"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(
    results_dir: Path,
    target_class_id: int,
    models: Sequence[str],
    collected: dict[str, ModelData],
) -> str:
    stats = {
        model: descriptive_statistics(collected[model].confidences)
        for model in models
    }
    available = [model for model in models if stats[model]["n"]]

    lines = [
        "# Big Orange Cone Confidence Analysis",
        "",
        f"Generated on {date.today().isoformat()} from `{results_dir}`.",
        "",
        "## Scope and method",
        "",
        f"Big Orange Cone was selected using zero-indexed class ID `{target_class_id}`. "
        "Every detection `.txt` file below each model directory was scanned recursively. "
        "The expected row format was `class_id confidence x1 y1 x2 y2`; JSON/COCO "
        "exports and visualization files were excluded.",
        "",
        "The main statistics are detection-weighted: every Big Orange Cone detection is "
        "one observation, including detections from the original images and all "
        "metamorphic transformation settings. Variance and standard deviation in the main "
        "tables are population measures (`ddof=0`). The detection files already reflect "
        "the confidence threshold used during inference, so this report describes saved "
        "detections rather than all pre-threshold model candidates.",
        "",
        "## Overall results",
        "",
    ]

    overall_rows = []
    for model in models:
        model_stats = stats[model]
        data = collected[model]
        overall_rows.append(
            [
                MODEL_LABELS.get(model, model),
                format_number(model_stats["n"]),
                f"{data.files_with_target:,} / {data.files_scanned:,}",
                format_number(model_stats.get("mean", math.nan)),
                format_number(model_stats.get("median", math.nan)),
                format_number(model_stats.get("population_variance", math.nan)),
                format_number(model_stats.get("population_std", math.nan)),
                format_number(model_stats.get("min", math.nan)),
                format_number(model_stats.get("max", math.nan)),
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "Model",
                    "Detections",
                    "Files with class / scanned",
                    "Mean",
                    "Median",
                    "Variance",
                    "Std. dev.",
                    "Min",
                    "Max",
                ],
                overall_rows,
            ),
            "",
            "## Distribution detail",
            "",
        ]
    )

    distribution_rows = []
    for model in models:
        model_stats = stats[model]
        distribution_rows.append(
            [
                MODEL_LABELS.get(model, model),
                format_number(model_stats.get("p05", math.nan)),
                format_number(model_stats.get("q1", math.nan)),
                format_number(model_stats.get("q3", math.nan)),
                format_number(model_stats.get("p95", math.nan)),
                format_number(model_stats.get("iqr", math.nan)),
                format_percent(model_stats.get("at_least_050", math.nan)),
                format_percent(model_stats.get("at_least_075", math.nan)),
                format_percent(model_stats.get("at_least_090", math.nan)),
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "Model",
                    "P05",
                    "P25",
                    "P75",
                    "P95",
                    "IQR",
                    "Confidence >= 0.50",
                    "Confidence >= 0.75",
                    "Confidence >= 0.90",
                ],
                distribution_rows,
            ),
            "",
            "## Results by transformation family",
            "",
            "These rows pool all numbered settings within a transformation family. "
            "`original` is kept separate as the untransformed baseline.",
            "",
        ]
    )

    family_rows = []
    for model in models:
        for family in sorted(collected[model].confidences_by_family):
            family_stats = descriptive_statistics(
                collected[model].confidences_by_family[family]
            )
            family_rows.append(
                [
                    MODEL_LABELS.get(model, model),
                    family,
                    format_number(family_stats["n"]),
                    format_number(family_stats.get("mean", math.nan)),
                    format_number(family_stats.get("median", math.nan)),
                    format_number(
                        family_stats.get("population_variance", math.nan)
                    ),
                    format_number(family_stats.get("population_std", math.nan)),
                ]
            )
    lines.extend(
        [
            markdown_table(
                [
                    "Model",
                    "Family",
                    "Detections",
                    "Mean",
                    "Median",
                    "Variance",
                    "Std. dev.",
                ],
                family_rows,
            ),
            "",
            "## File-level and parsing diagnostics",
            "",
        ]
    )

    diagnostics_rows = []
    for model in models:
        data = collected[model]
        per_file_stats = descriptive_statistics(data.file_mean_confidences)
        diagnostics_rows.append(
            [
                MODEL_LABELS.get(model, model),
                f"{data.files_scanned:,}",
                f"{data.empty_files:,}",
                f"{data.files_with_detections:,}",
                f"{data.files_with_target:,}",
                f"{data.valid_detection_rows:,}",
                f"{data.malformed_rows:,}",
                f"{data.out_of_range_rows:,}",
                format_number(per_file_stats.get("mean", math.nan)),
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "Model",
                    "Files scanned",
                    "Empty files",
                    "Files with detections",
                    "Files with class 2",
                    "Valid rows (all classes)",
                    "Malformed rows",
                    "Out-of-range rows",
                    "Mean of per-file class-2 means",
                ],
                diagnostics_rows,
            ),
            "",
            "The final column gives every class-2-positive file equal weight, unlike the "
            "detection-weighted main mean. Empty detection files are valid outputs and "
            "indicate that the model saved no detections for that image/setting.",
            "",
            "## Summary",
            "",
        ]
    )

    if available:
        highest_mean = max(available, key=lambda model: stats[model]["mean"])
        highest_median = max(available, key=lambda model: stats[model]["median"])
        lowest_variance = min(
            available, key=lambda model: stats[model]["population_variance"]
        )
        most_detections = max(available, key=lambda model: stats[model]["n"])
        lines.extend(
            [
                f"- Highest mean confidence: **{MODEL_LABELS.get(highest_mean, highest_mean)}** "
                f"({stats[highest_mean]['mean']:.6f}).",
                f"- Highest median confidence: **{MODEL_LABELS.get(highest_median, highest_median)}** "
                f"({stats[highest_median]['median']:.6f}).",
                f"- Lowest confidence variance: **{MODEL_LABELS.get(lowest_variance, lowest_variance)}** "
                f"({stats[lowest_variance]['population_variance']:.6f}).",
                f"- Most saved Big Orange Cone detections: "
                f"**{MODEL_LABELS.get(most_detections, most_detections)}** "
                f"({stats[most_detections]['n']:,}).",
            ]
        )
    else:
        lines.append("No Big Orange Cone detections were found.")

    lines.extend(
        [
            "",
            "Confidence is not accuracy: a model with a higher mean can be more confident "
            "without being better calibrated or more correct. Ground-truth matching is "
            "required to separate true positives from false positives and assess calibration.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python3 analysis/analyse_big_orange_cone_confidence.py",
            "```",
            "",
            "The input path, output path, models, and class ID can be changed with "
            "`--results-dir`, `--output`, `--models`, and `--class-id`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse Big Orange Cone confidence scores across YOLO models."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Dataset results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Markdown report path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model subdirectories to analyse",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=2,
        help="Target zero-indexed class ID (default: 2, Big Orange Cone)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    missing = [model for model in args.models if not (results_dir / model).is_dir()]
    if missing:
        raise SystemExit(
            "Missing model result directories: " + ", ".join(missing)
        )

    collected = {
        model: collect_model_data(results_dir / model, args.class_id)
        for model in args.models
    }
    report = build_report(results_dir, args.class_id, args.models, collected)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")
    for model in args.models:
        print(
            f"{MODEL_LABELS.get(model, model)}: "
            f"{len(collected[model].confidences):,} class-{args.class_id} detections"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
