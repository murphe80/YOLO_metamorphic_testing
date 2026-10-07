#!/usr/bin/env python3
"""Regenerate plots and Markdown from an existing bootstrap ``results.json``.

This renderer intentionally does not rerun COCO evaluation or bootstrapping. It
allows presentation changes to be applied without altering statistical results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.utils.calculate_sample_equivalence import (
    ALL_ARMS,
    ANALYSES,
    REFERENCE_ARM,
    TARGET_ARM,
    TARGET_ARMS,
    build_summaries,
    create_response_plots,
    load_config,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    default_dir = REPO_ROOT / "analysis" / "results" / "rq3" / "sample_equivalence"
    parser = argparse.ArgumentParser(
        description="Render sample-equivalence plots and report from results.json."
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=default_dir / "results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_dir,
    )
    return parser.parse_args()


def rebuild_analysis_arm_values(parameter_rows):
    """Reconstruct the per-analysis (arm, model, variant_id, metric) -> value maps."""
    analysis_arm_values = {analysis_key: {} for analysis_key in ANALYSES}
    for row in parameter_rows:
        values = analysis_arm_values[row["analysis"]]
        common = (row["model"], row["variant_id"], row["metric"])
        reference_key = (REFERENCE_ARM, *common)
        values[reference_key] = {
            "point": row["reference_statistic"],
            "ci_low": row["reference_statistic_ci_low"],
            "ci_high": row["reference_statistic_ci_high"],
            "baseline_point": row["reference_baseline_metric"],
            "setting_point": row["reference_setting_metric"],
        }
        target_key = (row["target_arm"], *common)
        values[target_key] = {
            "point": row["target_statistic"],
            "ci_low": row["target_statistic_ci_low"],
            "ci_high": row["target_statistic_ci_high"],
            "baseline_point": row["target_baseline_metric"],
            "setting_point": row["target_setting_metric"],
        }
    return analysis_arm_values


def main() -> None:
    cli_args = parse_args()
    with cli_args.results_json.open() as handle:
        data = json.load(handle)
    metadata = data["metadata"]
    if metadata["reference_arm"] != REFERENCE_ARM:
        raise ValueError(
            f"Expected reference arm {REFERENCE_ARM!r}; received "
            f"{metadata['reference_arm']!r}."
        )
    if tuple(metadata["target_arms"]) != TARGET_ARMS:
        raise ValueError(
            f"Expected target arms {TARGET_ARMS}; received "
            f"{metadata['target_arms']}."
        )
    missing_analyses = set(ANALYSES) - set(metadata.get("analyses", {}))
    if missing_analyses:
        raise ValueError(
            f"results.json is missing analyses {sorted(missing_analyses)}; "
            "it was likely produced by an older script version. Rerun "
            "calculate_sample_equivalence.py to regenerate it."
        )

    source_config = Path(metadata["source_config"])
    config = load_config(source_config, metadata["rules"])
    render_args = SimpleNamespace(
        output_dir=cli_args.output_dir,
        models=metadata["models"],
        source_dataset=metadata["source_dataset"],
        sample_fraction=metadata["sample_fraction"],
        bootstrap_repetitions=metadata["bootstrap_repetitions"],
        confidence_level=metadata["confidence_interval"]["level"],
        equivalence_margin=metadata["equivalence_margin"],
    )
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    analysis_arm_values = rebuild_analysis_arm_values(data["parameter_results"])
    for analysis_key in ANALYSES:
        expected_keys = {
            (arm, model, row["variant_id"], metric)
            for arm in ALL_ARMS
            for model in metadata["models"]
            for row in data["parameter_results"]
            if row["analysis"] == analysis_key and row["model"] == model
            for metric in metadata["metrics"]
            if row["metric"] == metric
        }
        missing = expected_keys - set(analysis_arm_values[analysis_key])
        if missing:
            raise ValueError(
                f"Missing rendered statistic records for {analysis_key!r}: "
                f"{sorted(missing)[:5]}"
            )

    plot_paths = create_response_plots(render_args, config, analysis_arm_values)
    summaries = build_summaries(data["parameter_results"], data["auc_results"])
    image_counts = {
        arm: int(arm_data["image_count"])
        for arm, arm_data in metadata["arms"].items()
    }
    report_path = write_markdown_report(
        render_args,
        config,
        data["parameter_results"],
        data["auc_results"],
        summaries,
        data["validation"],
        image_counts,
        plot_paths,
    )

    data["summaries"] = summaries
    data["plots"] = plot_paths
    output_json = cli_args.output_dir / cli_args.results_json.name
    with output_json.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    print(f"Updated JSON: {output_json}")
    print(f"Updated report: {report_path}")
    print(f"Updated plots: {len(plot_paths)}")


if __name__ == "__main__":
    main()
