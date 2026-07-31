#!/usr/bin/env python3
"""Analyse Big Orange Cone confidence across the ten imgDarkBright settings.

The saved YOLO detection files are expected to contain one row per detection::

    class_id confidence x1 y1 x2 y2

Only zero-indexed class ID 2 (Big Orange Cone) is selected by default.  This is
a detection-level confidence analysis: detections are not matched to ground
truth, so true and false positive predictions are pooled together.
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "results" / "test"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "BIG_ORANGE_CONE_DARKBRIGHT_CONFIDENCE_ANALYSIS.md"
DEFAULT_MODELS = ("yolov5", "yolov8", "yolov12", "yolov26")
MODEL_LABELS = {
    "yolov5": "YOLOv5",
    "yolov8": "YOLOv8",
    "yolov12": "YOLOv12",
    "yolov26": "YOLOv26",
}
# The mapping is defined by utils/config_generator.py and is also recorded in
# data/transformed_images/fsoco_val/imgDarkBright_*/configs.json.
SETTING_GAMMAS = tuple(0.5 * (index + 1) for index in range(10))


@dataclass
class SettingData:
    """Confidence observations and parse diagnostics for one setting."""

    confidences: list[float] = field(default_factory=list)
    per_file_means: list[float] = field(default_factory=list)
    files_scanned: int = 0
    empty_files: int = 0
    files_with_any_valid_detection: int = 0
    files_with_target: int = 0
    valid_rows: int = 0
    malformed_rows: int = 0
    out_of_range_rows: int = 0


def percentile(sorted_values: Sequence[float], percent: float) -> float:
    """Linearly interpolate a percentile from an already sorted sequence."""

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
    """Return detection-level descriptive statistics (population variance)."""

    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)
    q1 = percentile(ordered, 25)
    q3 = percentile(ordered, 75)
    return {
        "n": n,
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "variance": statistics.pvariance(ordered),
        "std": statistics.pstdev(ordered),
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


def collect_setting(setting_dir: Path, target_class_id: int) -> SettingData:
    """Parse all TXT detections under one imgDarkBright setting directory."""

    data = SettingData()
    for detection_path in sorted(setting_dir.rglob("*.txt")):
        data.files_scanned += 1
        file_has_nonempty_row = False
        file_has_valid_row = False
        target_confidences: list[float] = []

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

                file_has_valid_row = True
                data.valid_rows += 1
                if class_id == target_class_id:
                    target_confidences.append(confidence)
                    data.confidences.append(confidence)

        if not file_has_nonempty_row:
            data.empty_files += 1
        if file_has_valid_row:
            data.files_with_any_valid_detection += 1
        if target_confidences:
            data.files_with_target += 1
            data.per_file_means.append(statistics.fmean(target_confidences))

    return data


def linear_trend(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Return least-squares slope and Pearson r for paired finite values."""

    pairs = [
        (x, y)
        for x, y in zip(xs, ys)
        if math.isfinite(x) and math.isfinite(y)
    ]
    if len(pairs) < 2:
        return math.nan, math.nan
    paired_xs, paired_ys = zip(*pairs)
    mean_x = statistics.fmean(paired_xs)
    mean_y = statistics.fmean(paired_ys)
    covariance_sum = sum(
        (x - mean_x) * (y - mean_y) for x, y in pairs
    )
    x_square_sum = sum((x - mean_x) ** 2 for x in paired_xs)
    y_square_sum = sum((y - mean_y) ** 2 for y in paired_ys)
    slope = covariance_sum / x_square_sum if x_square_sum else math.nan
    denominator = math.sqrt(x_square_sum * y_square_sum)
    correlation = covariance_sum / denominator if denominator else math.nan
    return slope, correlation


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def fmt(value: float | int, digits: int = 6) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if math.isnan(value):
        return "N/A"
    return f"{value:.{digits}f}"


def fmt_percent(value: float) -> str:
    return "N/A" if math.isnan(value) else f"{100.0 * value:.2f}%"


def stat(stats: dict[str, float | int], name: str) -> float:
    """Read a statistic as a float, returning NaN when it is unavailable."""

    return float(stats.get(name, math.nan))


def build_report(
    results_dir: Path,
    target_class_id: int,
    models: Sequence[str],
    collected: dict[str, list[SettingData]],
) -> str:
    setting_stats = {
        model: [descriptive_statistics(data.confidences) for data in collected[model]]
        for model in models
    }
    pooled_values = {
        model: [
            confidence
            for setting_data in collected[model]
            for confidence in setting_data.confidences
        ]
        for model in models
    }
    pooled_stats = {
        model: descriptive_statistics(pooled_values[model]) for model in models
    }

    lines = [
        "# Big Orange Cone Confidence Across imgDarkBright Settings",
        "",
        f"Generated on {date.today().isoformat()} from `{results_dir}`.",
        "",
        "## Scope and interpretation",
        "",
        f"This analysis selects Big Orange Cone predictions using zero-indexed class ID "
        f"`{target_class_id}` from exactly `imgDarkBright_0` through "
        "`imgDarkBright_9` for YOLOv5, YOLOv8, YOLOv12, and YOLOv26. Each valid "
        "detection row has the format `class_id confidence x1 y1 x2 y2`; its saved "
        "confidence is one observation. JSON exports are not read.",
        "",
        "The setting-to-gamma mapping is 0.5, 1.0, ..., 5.0. The transform computes "
        "output intensity proportional to `input_intensity ** gamma`: gamma 0.5 "
        "brightens, gamma 1.0 is intensity-neutral, and increasing gamma above 1.0 "
        "progressively darkens. These confidence statistics do **not** use ground-truth "
        "matching, so true and false positive class-2 detections are pooled. They describe "
        "saved, post-threshold detections and should not be interpreted as accuracy or "
        "calibration measurements.",
        "",
        "Variance and standard deviation are population measures (`ddof=0`). The main "
        "tables are detection-weighted, meaning an image with multiple class-2 detections "
        "contributes multiple observations.",
        "",
        "## Pooled results across all ten settings",
        "",
    ]

    pooled_rows = []
    for model in models:
        stats = pooled_stats[model]
        pooled_rows.append(
            [
                MODEL_LABELS.get(model, model),
                fmt(stats["n"]),
                fmt(stat(stats, "mean")),
                fmt(stat(stats, "median")),
                fmt(stat(stats, "variance")),
                fmt(stat(stats, "std")),
                fmt(stat(stats, "min")),
                fmt(stat(stats, "max")),
                fmt_percent(stat(stats, "at_least_090")),
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "Model",
                    "Detections",
                    "Mean",
                    "Median",
                    "Variance",
                    "Std. dev.",
                    "Min",
                    "Max",
                    ">= 0.90",
                ],
                pooled_rows,
            ),
            "",
            "Pooling is useful as a compact summary, but settings with more saved "
            "detections receive more weight. The per-setting results below are the primary "
            "comparison for brightness sensitivity.",
            "",
            "## Results by model and setting",
            "",
        ]
    )

    main_rows = []
    for model in models:
        for index, gamma in enumerate(SETTING_GAMMAS):
            data = collected[model][index]
            stats = setting_stats[model][index]
            per_file = descriptive_statistics(data.per_file_means)
            main_rows.append(
                [
                    MODEL_LABELS.get(model, model),
                    f"imgDarkBright_{index}",
                    f"{gamma:.1f}",
                    fmt(stats["n"]),
                    f"{data.files_with_target:,} / {data.files_scanned:,}",
                    fmt(stat(stats, "mean")),
                    fmt(stat(stats, "median")),
                    fmt(stat(stats, "variance")),
                    fmt(stat(stats, "std")),
                    fmt(stat(per_file, "mean")),
                ]
            )
    lines.extend(
        [
            markdown_table(
                [
                    "Model",
                    "Setting",
                    "Gamma",
                    "Detections",
                    "Files with class / scanned",
                    "Mean",
                    "Median",
                    "Variance",
                    "Std. dev.",
                    "Mean of file means",
                ],
                main_rows,
            ),
            "",
            "`Mean of file means` gives each class-2-positive file equal weight; files "
            "without a saved class-2 prediction are excluded from that mean.",
            "",
            "## Distribution detail by setting",
            "",
        ]
    )

    distribution_rows = []
    for model in models:
        for index, gamma in enumerate(SETTING_GAMMAS):
            stats = setting_stats[model][index]
            distribution_rows.append(
                [
                    MODEL_LABELS.get(model, model),
                    f"{index} ({gamma:.1f})",
                    fmt(stat(stats, "min")),
                    fmt(stat(stats, "p05")),
                    fmt(stat(stats, "q1")),
                    fmt(stat(stats, "q3")),
                    fmt(stat(stats, "p95")),
                    fmt(stat(stats, "max")),
                    fmt(stat(stats, "iqr")),
                    fmt_percent(stat(stats, "at_least_050")),
                    fmt_percent(stat(stats, "at_least_075")),
                    fmt_percent(stat(stats, "at_least_090")),
                ]
            )
    lines.extend(
        [
            markdown_table(
                [
                    "Model",
                    "Setting (gamma)",
                    "Min",
                    "P05",
                    "P25",
                    "P75",
                    "P95",
                    "Max",
                    "IQR",
                    ">= 0.50",
                    ">= 0.75",
                    ">= 0.90",
                ],
                distribution_rows,
            ),
            "",
            "## Brightness trends and model comparisons",
            "",
        ]
    )

    trend_rows = []
    for model in models:
        means = [stat(stats, "mean") for stats in setting_stats[model]]
        slope, correlation = linear_trend(SETTING_GAMMAS, means)
        available = [index for index, mean in enumerate(means) if math.isfinite(mean)]
        if available:
            best = max(available, key=lambda index: means[index])
            worst = min(available, key=lambda index: means[index])
            first = available[0]
            last = available[-1]
            endpoint_change = means[last] - means[first]
            detection_change = (
                len(collected[model][last].confidences)
                - len(collected[model][first].confidences)
            )
        else:
            best = worst = -1
            endpoint_change = math.nan
            detection_change = 0
        trend_rows.append(
            [
                MODEL_LABELS.get(model, model),
                f"{best} ({SETTING_GAMMAS[best]:.1f})" if best >= 0 else "N/A",
                fmt(means[best]) if best >= 0 else "N/A",
                f"{worst} ({SETTING_GAMMAS[worst]:.1f})" if worst >= 0 else "N/A",
                fmt(means[worst]) if worst >= 0 else "N/A",
                fmt(endpoint_change),
                f"{detection_change:+,}",
                fmt(slope),
                fmt(correlation),
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "Model",
                    "Highest-mean setting (gamma)",
                    "Highest mean",
                    "Lowest-mean setting (gamma)",
                    "Lowest mean",
                    "Mean change 0->9",
                    "Detection-count change 0->9",
                    "Mean slope / gamma",
                    "Pearson r",
                ],
                trend_rows,
            ),
            "",
            "The slope and Pearson correlation summarize the ten per-setting means. They "
            "describe association with gamma, not necessarily a linear or causal response; "
            "the highest/lowest rows retain non-monotonic behavior that a slope can hide.",
            "",
            "### Highest mean confidence at each setting",
            "",
        ]
    )

    winner_rows = []
    winner_counts = {model: 0 for model in models}
    for index, gamma in enumerate(SETTING_GAMMAS):
        available_models = [
            model
            for model in models
            if math.isfinite(stat(setting_stats[model][index], "mean"))
        ]
        if available_models:
            winner = max(
                available_models,
                key=lambda model: stat(setting_stats[model][index], "mean"),
            )
            winner_mean = stat(setting_stats[winner][index], "mean")
            winner_counts[winner] += 1
        else:
            winner = ""
            winner_mean = math.nan
        winner_rows.append(
            [
                f"imgDarkBright_{index}",
                f"{gamma:.1f}",
                MODEL_LABELS.get(winner, winner) if winner else "N/A",
                fmt(winner_mean),
                " / ".join(
                    f"{MODEL_LABELS.get(model, model)} {fmt(stat(setting_stats[model][index], 'mean'))}"
                    for model in models
                ),
            ]
        )
    lines.extend(
        [
            markdown_table(
                ["Setting", "Gamma", "Highest-mean model", "Mean", "All model means"],
                winner_rows,
            ),
            "",
            "## Key findings",
            "",
        ]
    )

    models_with_detections = [
        model for model in models if int(pooled_stats[model]["n"]) > 0
    ]
    if models_with_detections:
        highest_pooled = max(
            models_with_detections,
            key=lambda model: stat(pooled_stats[model], "mean"),
        )
        endpoint_changes = {
            model: (
                stat(setting_stats[model][-1], "mean")
                - stat(setting_stats[model][0], "mean")
            )
            for model in models_with_detections
        }
        largest_drop = min(endpoint_changes, key=endpoint_changes.get)
        lines.extend(
            [
                f"- **{MODEL_LABELS.get(highest_pooled, highest_pooled)}** has the "
                f"highest pooled mean confidence ({stat(pooled_stats[highest_pooled], 'mean'):.6f}).",
                f"- All four models have lower mean confidence at gamma 5.0 than at "
                f"gamma 0.5. The largest endpoint decrease is "
                f"**{MODEL_LABELS.get(largest_drop, largest_drop)}** "
                f"({endpoint_changes[largest_drop]:.6f}).",
                "- "
                + "; ".join(
                    f"**{MODEL_LABELS.get(model, model)}** has the highest setting mean "
                    f"in {winner_counts[model]} of 10 settings"
                    for model in models
                )
                + ".",
            ]
        )
    else:
        lines.append("- No class-2 detections were available for comparison.")
    lines.extend(
        [
            "",
            "## Parsing diagnostics",
            "",
        ]
    )

    diagnostic_rows = []
    for model in models:
        for index, gamma in enumerate(SETTING_GAMMAS):
            data = collected[model][index]
            diagnostic_rows.append(
                [
                    MODEL_LABELS.get(model, model),
                    f"{index} ({gamma:.1f})",
                    f"{data.files_scanned:,}",
                    f"{data.empty_files:,}",
                    f"{data.files_with_any_valid_detection:,}",
                    f"{data.files_with_target:,}",
                    f"{data.valid_rows:,}",
                    f"{data.malformed_rows:,}",
                    f"{data.out_of_range_rows:,}",
                ]
            )
    lines.extend(
        [
            markdown_table(
                [
                    "Model",
                    "Setting (gamma)",
                    "Files",
                    "Empty files",
                    "Files with any detection",
                    "Files with class 2",
                    "Valid rows (all classes)",
                    "Malformed rows",
                    "Out-of-range confidence rows",
                ],
                diagnostic_rows,
            ),
            "",
            "An empty file is a valid inference output containing no saved detections. A "
            "malformed row has the wrong field count, a non-numeric value, or a non-finite "
            "value. Rows with confidence outside [0, 1] are counted separately and excluded.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python3 analysis/analyse_big_orange_cone_darkbright_confidence.py",
            "```",
            "",
            "Use `--results-dir`, `--output`, `--models`, or `--class-id` to override "
            "the defaults. All ten required model setting directories must exist.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse Big Orange Cone confidence across imgDarkBright_0 through "
            "imgDarkBright_9."
        )
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
    required_directories = [
        results_dir / model / f"imgDarkBright_{index}"
        for model in args.models
        for index in range(len(SETTING_GAMMAS))
    ]
    missing = [path for path in required_directories if not path.is_dir()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Missing required result directories:\n{formatted}")

    collected = {
        model: [
            collect_setting(
                results_dir / model / f"imgDarkBright_{index}", args.class_id
            )
            for index in range(len(SETTING_GAMMAS))
        ]
        for model in args.models
    }
    report = build_report(results_dir, args.class_id, args.models, collected)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    print(f"Wrote {output}")
    for model in args.models:
        count = sum(len(data.confidences) for data in collected[model])
        malformed = sum(data.malformed_rows for data in collected[model])
        out_of_range = sum(data.out_of_range_rows for data in collected[model])
        print(
            f"{MODEL_LABELS.get(model, model)}: {count:,} class-{args.class_id} "
            f"detections; {malformed:,} malformed rows; "
            f"{out_of_range:,} out-of-range rows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
