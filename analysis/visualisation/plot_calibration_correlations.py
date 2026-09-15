#!/usr/bin/env python3
"""Plot confidence-calibration correlations across metamorphic rules.

For every model and multi-setting rule, this script recomputes confidence
observations from the detection files and calculates the three Pearson
correlations used in Part 2 of ``analysis/results/test/venn_analysis.md``:

1. |mean retained TP confidence change| vs. (1 - TP retention rate)
2. mean lost TP confidence gap vs. TP retention rate
3. |mean persistent FP confidence change| vs. FP consistency rate

The third coefficient is sign-flipped for presentation, matching the report:
larger persistent-FP confidence changes are expected when FP consistency is
lower, so positive plotted values represent a stronger inverse relationship.

One chart is written per correlation type, with one trend line per model.
Singleton rules are skipped because Pearson correlation requires multiple
settings.

Usage:
    python3 analysis/visualisation/plot_calibration_correlations.py --dataset test
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.visualisation.plot_confidence_change import (  # noqa: E402
    LOST_TP_CHANGE,
    PERSISTENT_FP_CHANGE,
    RETAINED_TP_CHANGE,
    confidence_summary,
)
from utils.model_registry import get_model_by_name, get_model_names  # noqa: E402


DEFAULT_DATASET = "test"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "brute_force_transformations.json"
ANALYSIS_RESULTS_ROOT = PROJECT_ROOT / "analysis" / "results"
DETECTION_RESULTS_ROOT = PROJECT_ROOT / "results"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

RETAINED_CORRELATION = "retained_tp_vs_tp_retention"
LOST_CORRELATION = "lost_tp_vs_tp_retention"
PERSISTENT_FP_CORRELATION = "persistent_fp_vs_fp_consistency"

CORRELATION_SPECS = {
    RETAINED_CORRELATION: {
        "title": "Retained-TP Confidence vs. TP Failure Rate",
        "subtitle": "Pearson r: |mean retained TP confidence change| vs. (1 − TP retention)",
        "confidence_metric": RETAINED_TP_CHANGE,
        "venn_metric": "tp_retention_rate",
        "invert_venn": True,
        "absolute_confidence": True,
        "flip_correlation": False,
    },
    LOST_CORRELATION: {
        "title": "Lost-TP Confidence Gap vs. TP Retention",
        "subtitle": "Pearson r: mean lost TP confidence gap vs. TP retention",
        "confidence_metric": LOST_TP_CHANGE,
        "venn_metric": "tp_retention_rate",
        "invert_venn": False,
        "absolute_confidence": False,
        "flip_correlation": False,
    },
    PERSISTENT_FP_CORRELATION: {
        "title": "Persistent-FP Confidence vs. FP Consistency",
        "subtitle": "−Pearson r: |mean persistent FP confidence change| vs. FP consistency",
        "confidence_metric": PERSISTENT_FP_CHANGE,
        "venn_metric": "fp_consistency_rate",
        "invert_venn": False,
        "absolute_confidence": True,
        "flip_correlation": True,
    },
}


def load_json(path: Path) -> dict:
    """Load and validate a JSON object."""
    with open(path) as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def pearson_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    """Calculate Pearson's r, returning None for insufficient/constant data."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = math.sqrt(float(np.dot(x_centered, x_centered))) * math.sqrt(
        float(np.dot(y_centered, y_centered))
    )
    if denominator == 0:
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def model_color(model: str) -> str:
    """Return the configured model colour or a neutral fallback."""
    model_config = get_model_by_name(model)
    if model_config and "color" in model_config:
        return model_config["color"]["hex"]
    return "#777777"


def model_label(model: str) -> str:
    """Return the configured display name for a model."""
    model_config = get_model_by_name(model)
    if model_config:
        return model_config.get("display_name", model)
    return model


def format_rule(rule: str) -> str:
    """Format a metamorphic-rule identifier for an axis tick."""
    import re

    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", rule)
    return spaced.replace("_", " ").strip().title()


def discover_models(results_dir: Path, requested: Optional[List[str]]) -> List[str]:
    """Return requested models or registered models with result directories."""
    if requested:
        return list(dict.fromkeys(requested))
    return [model for model in get_model_names() if (results_dir / model).is_dir()]


def calculate_correlations(
    rules: Sequence[str],
    transformations: Dict[str, List[dict]],
    models: Sequence[str],
    statistics: Dict[str, dict],
    results_dir: Path,
    gt_path: Path,
    iou_threshold: float,
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """Calculate every correlation type for each model and eligible rule."""
    correlations = {
        correlation_name: {model: {} for model in models}
        for correlation_name in CORRELATION_SPECS
    }

    for rule in rules:
        for model in models:
            setting_values = {
                correlation_name: {"confidence": [], "venn": []}
                for correlation_name in CORRELATION_SPECS
            }

            for variant in transformations[rule]:
                variant_id = variant["variant_id"]
                venn_record = statistics.get(f"{model}_{variant_id}")
                if not isinstance(venn_record, dict):
                    continue

                # Matching is the expensive step. Compute it once per setting,
                # then reuse its three observation groups for all correlations.
                summary = confidence_summary(
                    results_dir / model,
                    variant_id,
                    rule,
                    gt_path,
                    iou_threshold,
                )

                for correlation_name, specification in CORRELATION_SPECS.items():
                    venn_value = venn_record.get(specification["venn_metric"])
                    observations = summary.values[specification["confidence_metric"]]
                    if venn_value is None or not observations:
                        continue

                    confidence_value = sum(observations) / len(observations)
                    if specification["absolute_confidence"]:
                        confidence_value = abs(confidence_value)

                    venn_value = float(venn_value)
                    if specification["invert_venn"]:
                        venn_value = 1.0 - venn_value

                    setting_values[correlation_name]["confidence"].append(
                        confidence_value
                    )
                    setting_values[correlation_name]["venn"].append(venn_value)

            for correlation_name, specification in CORRELATION_SPECS.items():
                values = setting_values[correlation_name]
                correlation = pearson_correlation(
                    values["confidence"], values["venn"]
                )
                if correlation is not None and specification["flip_correlation"]:
                    correlation = -correlation
                correlations[correlation_name][model][rule] = correlation

    return correlations


def plot_correlation(
    correlation_name: str,
    correlations: Dict[str, Dict[str, Optional[float]]],
    rules: Sequence[str],
    output_path: Path,
) -> bool:
    """Plot one categorical rule-to-correlation chart."""
    if not any(value is not None for model_values in correlations.values() for value in model_values.values()):
        return False

    specification = CORRELATION_SPECS[correlation_name]
    x_positions = np.arange(len(rules), dtype=float)
    fig, axis = plt.subplots(figsize=(12, 6.5))

    for index, model in enumerate(correlations):
        y_values = [
            correlations[model].get(rule)
            if correlations[model].get(rule) is not None
            else np.nan
            for rule in rules
        ]
        axis.plot(
            x_positions,
            y_values,
            marker=MARKERS[index % len(MARKERS)],
            linewidth=2,
            markersize=7,
            color=model_color(model),
            label=model_label(model),
        )

    axis.axhline(0, color="#555555", linewidth=1, linestyle="--", alpha=0.7)
    axis.set_xticks(x_positions)
    axis.set_xticklabels([format_rule(rule) for rule in rules], rotation=25, ha="right")
    axis.set_xlabel("Metamorphic rule", fontsize=11, fontweight="bold")
    axis.set_ylabel("Correlation value", fontsize=11, fontweight="bold")
    axis.set_ylim(-1.05, 1.05)
    axis.set_title(
        f"{specification['title']}\n{specification['subtitle']}",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    axis.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    axis.legend(loc="best", framealpha=0.9, fontsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Part 2 confidence-calibration correlations by metamorphic rule."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset name (default: test)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", "--results_dir", dest="results_dir", type=Path)
    parser.add_argument("--gt-path", "--gt_path", dest="gt_path", type=Path)
    parser.add_argument("--statistics", type=Path, help="Path to venn_statistics.json")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path)
    parser.add_argument("--rules", nargs="*", help="Rules to include (default: all eligible rules)")
    parser.add_argument("--models", nargs="*", help="Models to include (default: discovered models)")
    parser.add_argument(
        "--iou-threshold",
        "--iou_threshold",
        dest="iou_threshold",
        type=float,
        default=0.5,
        help="Minimum IoU used by confidence matching (default: 0.5)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = (args.results_dir or DETECTION_RESULTS_ROOT / args.dataset).resolve()
    gt_path = (
        args.gt_path
        or PROJECT_ROOT / "data" / "base_images" / args.dataset / "annotations_coco.json"
    ).resolve()
    statistics_path = (
        args.statistics
        or ANALYSIS_RESULTS_ROOT / args.dataset / "venn_diagrams" / "venn_statistics.json"
    ).resolve()
    output_dir = (
        args.output_dir
        or ANALYSIS_RESULTS_ROOT / args.dataset / "venn_diagrams" / "calibration"
    ).resolve()

    if not 0 <= args.iou_threshold <= 1:
        print("Error: --iou-threshold must be between 0 and 1", file=sys.stderr)
        return 2

    required_paths = {
        "configuration": args.config,
        "detection results": results_dir,
        "ground truth": gt_path,
        "Venn statistics": statistics_path,
    }
    for label, path in required_paths.items():
        if not path.exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            return 1

    try:
        transformations = load_json(args.config)
        statistics = load_json(statistics_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    models = discover_models(results_dir, args.models)
    if not models:
        print(f"Error: no model result directories found in {results_dir}", file=sys.stderr)
        return 1

    requested_rules = args.rules or list(transformations)
    missing_rules = [rule for rule in requested_rules if rule not in transformations]
    if missing_rules:
        print(f"Error: rules not found in configuration: {', '.join(missing_rules)}", file=sys.stderr)
        return 1

    rules = [rule for rule in requested_rules if len(transformations[rule]) >= 2]
    skipped_rules = [rule for rule in requested_rules if len(transformations[rule]) < 2]
    if not rules:
        print("Error: no rules with at least two settings were selected", file=sys.stderr)
        return 1

    if skipped_rules:
        print(f"Skipping singleton rules: {', '.join(skipped_rules)}")
    print(f"Dataset: {args.dataset}")
    print(f"Models: {', '.join(models)}")
    print(f"Rules: {', '.join(rules)}")
    print(f"Output: {output_dir}")

    correlations = calculate_correlations(
        rules,
        transformations,
        models,
        statistics,
        results_dir,
        gt_path,
        args.iou_threshold,
    )

    generated = 0
    for correlation_name, model_values in correlations.items():
        output_path = output_dir / f"{correlation_name}.png"
        if plot_correlation(correlation_name, model_values, rules, output_path):
            print(f"  Saved: {output_path}")
            generated += 1
        else:
            print(f"  Skip {correlation_name}: insufficient observations")

    print(f"Generated {generated} correlation plot(s).")
    return 0 if generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
