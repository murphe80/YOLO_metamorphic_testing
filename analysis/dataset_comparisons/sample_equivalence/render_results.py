#!/usr/bin/env python3
"""Regenerate plots and Markdown from an existing bootstrap ``results.json``.

This renderer intentionally does not rerun COCO evaluation or bootstrapping. It
allows presentation changes to be applied without altering statistical results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from bootstrap_equivalence import (
    ALL_ARMS,
    REFERENCE_ARM,
    TARGET_ARM,
    TARGET_ARMS,
    build_summaries,
    create_response_plots,
    load_config,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent
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


def rebuild_arm_degradations(parameter_rows):
    values = {}
    for row in parameter_rows:
        common = (row["model"], row["variant_id"], row["metric"])
        reference_key = (REFERENCE_ARM, *common)
        values[reference_key] = {
            "point": row["reference_degradation"],
            "ci_low": row["reference_degradation_ci_low"],
            "ci_high": row["reference_degradation_ci_high"],
        }
        target_key = (row["target_arm"], *common)
        values[target_key] = {
            "point": row["target_degradation"],
            "ci_low": row["target_degradation_ci_low"],
            "ci_high": row["target_degradation_ci_high"],
        }
    return values


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
    arm_degradations = rebuild_arm_degradations(data["parameter_results"])
    expected_keys = {
        (arm, model, row["variant_id"], metric)
        for arm in ALL_ARMS
        for model in metadata["models"]
        for row in data["parameter_results"]
        if row["model"] == model
        for metric in metadata["metrics"]
        if row["metric"] == metric
    }
    missing = expected_keys - set(arm_degradations)
    if missing:
        raise ValueError(f"Missing rendered degradation records: {sorted(missing)[:5]}")

    plot_paths = create_response_plots(
        render_args, config, arm_degradations
    )
    summaries = build_summaries(
        data["parameter_results"], data["auc_results"]
    )
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
