#!/usr/bin/env python3
"""
Compare original-to-transformed detection outcome distributions across datasets.

For every model and configured metamorphic-test variant, detections are matched
using the repository's existing confidence-ordered, one-to-one greedy IoU
matcher. Each image becomes one cluster with three outcome counts:

    retained: original detections matched to transformed detections
    lost:     unmatched original detections
    gained:   unmatched transformed detections

The reference and target datasets are compared with a Pearson chi-square
statistic. Its p-value is calibrated by permuting whole image clusters, rather
than individual detections, so detections from the same image are never treated
as independent observations. Benjamini-Hochberg correction is then applied
within each model/target family by default.

Typical usage:

    venv/bin/python analysis/utils/detection_distribution_tests.py

    venv/bin/python analysis/utils/detection_distribution_tests.py \
        --iou-threshold 0.6 \
        --permutations 19999 \
        --output-dir analysis/stat_tests
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from scipy.stats import chi2


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.utils.detection_matching import match_detection_pairs


OUTCOMES = ("retained", "lost", "gained")
GEOMETRIC_TRANSFORMATIONS = {"imgStretch", "imgResolution"}
DEFAULT_DATASETS = ("test", "val", "train_samples")
DEFAULT_CONFIG = REPO_ROOT / "config" / "brute_force_transformations.json"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "stat_tests"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare retained/lost/gained detection distributions in a reference "
            "dataset against one or more target datasets."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root containing <dataset>/<model>/... detection results.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Metamorphic variant configuration JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV, JSON, and Markdown outputs.",
    )
    parser.add_argument(
        "--reference-dataset",
        default="test",
        help="Reference dataset used in every comparison (default: test).",
    )
    parser.add_argument(
        "--target-datasets",
        nargs="+",
        default=["val", "train_samples"],
        help="Datasets compared with the reference.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model names. Default: models complete in every dataset.",
    )
    parser.add_argument(
        "--transformations",
        nargs="+",
        default=None,
        help="Optional base transformation names. Default: all configured tests.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help=(
            "Strict lower IoU bound for a match, matching current analysis "
            "behavior (default: 0.5, so IoU must be > 0.5)."
        ),
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=9999,
        help="Whole-image label permutations per comparison (default: 9999).",
    )
    parser.add_argument(
        "--permutation-batch-size",
        type=int,
        default=256,
        help="Number of permutations evaluated per vectorized batch.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR-adjusted significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260729,
        help="Base random seed. Each comparison receives a stable derived seed.",
    )
    parser.add_argument(
        "--correction-scope",
        choices=("model", "global", "none"),
        default="model",
        help=(
            "BH family: each model/target's settings, all models per target, "
            "or no correction (default: model)."
        ),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be between 0 and 1.")
    if args.permutations < 0:
        raise ValueError("--permutations cannot be negative.")
    if args.permutation_batch_size <= 0:
        raise ValueError("--permutation-batch-size must be positive.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1.")
    if args.reference_dataset in args.target_datasets:
        raise ValueError("The reference dataset cannot also be a target dataset.")


def _load_variant_config(
    config_path: Path,
    selected_transformations: Sequence[str] | None,
) -> List[Dict]:
    with config_path.open() as handle:
        raw_config = json.load(handle)

    selected = set(selected_transformations or [])
    variants: List[Dict] = []
    for transformation, configured_variants in raw_config.items():
        if selected and transformation not in selected:
            continue
        if isinstance(configured_variants, dict):
            configured_variants = [configured_variants]
        for position, params in enumerate(configured_variants):
            if not isinstance(params, dict):
                raise ValueError(
                    f"Expected an object for {transformation} setting {position}."
                )
            variant_id = params.get("variant_id", f"{transformation}_{position}")
            setting_text = variant_id.rsplit("_", 1)[-1]
            setting = int(setting_text) if setting_text.isdigit() else position
            variants.append(
                {
                    "transformation": transformation,
                    "variant_id": variant_id,
                    "setting": setting,
                    "parameters": {
                        key: value
                        for key, value in params.items()
                        if key != "variant_id"
                    },
                }
            )

    if selected:
        missing = selected - {item["transformation"] for item in variants}
        if missing:
            raise ValueError(
                f"Transformations not found in {config_path}: {sorted(missing)}"
            )
    if not variants:
        raise ValueError(f"No metamorphic variants selected from {config_path}.")
    return variants


def _discover_complete_models(
    results_root: Path,
    datasets: Sequence[str],
    variants: Sequence[Mapping],
) -> List[str]:
    model_sets = []
    for dataset in datasets:
        dataset_dir = results_root / dataset
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset results directory not found: {dataset_dir}")
        model_sets.append(
            {
                path.name
                for path in dataset_dir.iterdir()
                if path.is_dir()
                and (path / "original" / "detections_original_coco.json").exists()
            }
        )

    candidates = sorted(set.intersection(*model_sets))
    complete = []
    for model in candidates:
        if all(
            _transformed_path(
                results_root,
                dataset,
                model,
                variant["transformation"],
                variant["variant_id"],
            ).exists()
            for dataset in datasets
            for variant in variants
        ):
            complete.append(model)
    return complete


def _original_path(results_root: Path, dataset: str, model: str) -> Path:
    return results_root / dataset / model / "original" / "detections_original_coco.json"


def _transformed_path(
    results_root: Path,
    dataset: str,
    model: str,
    transformation: str,
    variant_id: str,
) -> Path:
    filename = (
        "detections_coco_transformed.json"
        if transformation in GEOMETRIC_TRANSFORMATIONS
        else "detections_coco.json"
    )
    return results_root / dataset / model / variant_id / filename


@lru_cache(maxsize=None)
def _load_coco_detections(path_string: str) -> Tuple[Dict[int, List[Dict]], Tuple[int, ...]]:
    path = Path(path_string)
    with path.open() as handle:
        data = json.load(handle)

    if isinstance(data, list):
        annotations = data
        declared_image_ids: Iterable[int] = ()
    elif isinstance(data, dict):
        annotations = data.get("annotations", [])
        declared_image_ids = (
            image["id"] for image in data.get("images", []) if "id" in image
        )
    else:
        raise ValueError(f"Unsupported detection JSON structure: {path}")

    by_image: Dict[int, List[Dict]] = defaultdict(list)
    for annotation in annotations:
        by_image[int(annotation["image_id"])].append(annotation)

    image_ids = tuple(sorted(set(declared_image_ids) | set(by_image)))
    return dict(by_image), image_ids


@lru_cache(maxsize=None)
def _image_outcomes(
    original_path_string: str,
    transformed_path_string: str,
    iou_threshold: float,
) -> Tuple[Tuple[int, int, int], ...]:
    original_by_image, original_image_ids = _load_coco_detections(
        original_path_string
    )
    transformed_by_image, transformed_image_ids = _load_coco_detections(
        transformed_path_string
    )
    image_ids = sorted(set(original_image_ids) | set(transformed_image_ids))

    outcomes = []
    for image_id in image_ids:
        original = original_by_image.get(image_id, [])
        transformed = transformed_by_image.get(image_id, [])
        retained = len(
            match_detection_pairs(original, transformed, iou_threshold)
        )
        outcomes.append(
            (retained, len(original) - retained, len(transformed) - retained)
        )
    return tuple(outcomes)


def _pearson_chi_square(
    first: np.ndarray,
    second: np.ndarray,
) -> Tuple[float, int]:
    table = np.vstack((first, second)).astype(float)
    nonempty_columns = table.sum(axis=0) > 0
    table = table[:, nonempty_columns]
    if table.size == 0 or table.shape[1] < 2 or table.sum() == 0:
        return 0.0, max(table.shape[1] - 1, 0)

    row_totals = table.sum(axis=1, keepdims=True)
    column_totals = table.sum(axis=0, keepdims=True)
    expected = row_totals @ column_totals / table.sum()
    valid = expected > 0
    statistic = np.sum(((table - expected) ** 2)[valid] / expected[valid])
    return float(statistic), table.shape[1] - 1


def _pearson_chi_square_batch(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Vectorized Pearson statistics for arrays shaped (batch, outcomes)."""
    pooled_columns = first + second
    keep = pooled_columns[0] > 0
    first = first[:, keep].astype(float, copy=False)
    second = second[:, keep].astype(float, copy=False)
    if first.shape[1] < 2:
        return np.zeros(first.shape[0], dtype=float)

    grand_total = (first + second).sum(axis=1)
    first_rows = first.sum(axis=1)
    second_rows = second.sum(axis=1)
    column_totals = first + second

    expected_first = (
        first_rows[:, None] * column_totals / grand_total[:, None]
    )
    expected_second = (
        second_rows[:, None] * column_totals / grand_total[:, None]
    )
    statistic = np.zeros(first.shape[0], dtype=float)
    valid_first = expected_first > 0
    valid_second = expected_second > 0
    statistic += np.sum(
        np.divide(
            (first - expected_first) ** 2,
            expected_first,
            out=np.zeros_like(expected_first),
            where=valid_first,
        ),
        axis=1,
    )
    statistic += np.sum(
        np.divide(
            (second - expected_second) ** 2,
            expected_second,
            out=np.zeros_like(expected_second),
            where=valid_second,
        ),
        axis=1,
    )
    return statistic


def _stable_rng_seed(base_seed: int, *labels: str) -> int:
    digest = hashlib.sha256(
        "|".join((str(base_seed), *labels)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _cluster_permutation_test(
    reference_outcomes: np.ndarray,
    target_outcomes: np.ndarray,
    permutations: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    reference_counts = reference_outcomes.sum(axis=0)
    target_counts = target_outcomes.sum(axis=0)
    statistic, degrees_of_freedom = _pearson_chi_square(
        reference_counts, target_counts
    )
    asymptotic_p = (
        float(chi2.sf(statistic, degrees_of_freedom))
        if degrees_of_freedom > 0
        else 1.0
    )

    if permutations == 0:
        permutation_p = asymptotic_p
    else:
        pooled = np.vstack((reference_outcomes, target_outcomes)).astype(
            np.int64, copy=False
        )
        pooled_counts = pooled.sum(axis=0)
        reference_size = len(reference_outcomes)
        exceedances = 0
        completed = 0
        while completed < permutations:
            current_batch = min(batch_size, permutations - completed)
            random_keys = rng.random((current_batch, len(pooled)))
            selected_indices = np.argpartition(
                random_keys, reference_size - 1, axis=1
            )[:, :reference_size]
            permuted_reference = pooled[selected_indices].sum(axis=1)
            permuted_target = pooled_counts - permuted_reference
            permuted_statistics = _pearson_chi_square_batch(
                permuted_reference, permuted_target
            )
            exceedances += int(
                np.count_nonzero(permuted_statistics >= statistic - 1e-12)
            )
            completed += current_batch
        permutation_p = (exceedances + 1.0) / (permutations + 1.0)

    grand_total = float(reference_counts.sum() + target_counts.sum())
    cramers_v = math.sqrt(statistic / grand_total) if grand_total else 0.0
    reference_proportions = _proportions(reference_counts)
    target_proportions = _proportions(target_counts)
    total_variation = 0.5 * float(
        np.abs(reference_proportions - target_proportions).sum()
    )

    return {
        "chi_square": statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "asymptotic_p_value": asymptotic_p,
        "permutation_p_value": permutation_p,
        "cramers_v": cramers_v,
        "total_variation": total_variation,
    }


def _proportions(counts: np.ndarray) -> np.ndarray:
    total = float(counts.sum())
    if total == 0:
        return np.zeros(len(OUTCOMES), dtype=float)
    return counts.astype(float) / total


def _benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted.tolist()


def _apply_multiple_testing_correction(
    rows: List[MutableMapping],
    scope: str,
    alpha: float,
) -> None:
    families: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if scope == "model":
            key = (row["target_dataset"], row["model"])
        elif scope == "global":
            key = (row["target_dataset"],)
        else:
            key = (row["target_dataset"], row["model"], row["variant_id"])
        families[key].append(index)

    for indices in families.values():
        p_values = [rows[index]["permutation_p_value"] for index in indices]
        q_values = (
            p_values if scope == "none" else _benjamini_hochberg(p_values)
        )
        for index, q_value in zip(indices, q_values):
            rows[index]["bh_q_value"] = q_value
            rows[index]["significant"] = bool(q_value <= alpha)
            rows[index]["result"] = (
                "significant_difference"
                if q_value <= alpha
                else "no_significant_difference_detected"
            )


def _build_result_row(
    results_root: Path,
    reference_dataset: str,
    target_dataset: str,
    model: str,
    variant: Mapping,
    iou_threshold: float,
    permutations: int,
    batch_size: int,
    base_seed: int,
) -> Dict:
    transformation = variant["transformation"]
    variant_id = variant["variant_id"]
    reference_original = _original_path(results_root, reference_dataset, model)
    target_original = _original_path(results_root, target_dataset, model)
    reference_transformed = _transformed_path(
        results_root, reference_dataset, model, transformation, variant_id
    )
    target_transformed = _transformed_path(
        results_root, target_dataset, model, transformation, variant_id
    )
    for path in (
        reference_original,
        target_original,
        reference_transformed,
        target_transformed,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required detection file not found: {path}")

    reference_outcomes = np.asarray(
        _image_outcomes(
            str(reference_original), str(reference_transformed), iou_threshold
        ),
        dtype=np.int64,
    )
    target_outcomes = np.asarray(
        _image_outcomes(
            str(target_original), str(target_transformed), iou_threshold
        ),
        dtype=np.int64,
    )
    rng = np.random.default_rng(
        _stable_rng_seed(
            base_seed, reference_dataset, target_dataset, model, variant_id
        )
    )
    test = _cluster_permutation_test(
        reference_outcomes,
        target_outcomes,
        permutations,
        batch_size,
        rng,
    )

    reference_counts = reference_outcomes.sum(axis=0)
    target_counts = target_outcomes.sum(axis=0)
    reference_proportions = _proportions(reference_counts)
    target_proportions = _proportions(target_counts)

    row = {
        "reference_dataset": reference_dataset,
        "target_dataset": target_dataset,
        "model": model,
        "transformation": transformation,
        "setting": variant["setting"],
        "variant_id": variant_id,
        "parameters": json.dumps(variant["parameters"], sort_keys=True),
        "iou_threshold": iou_threshold,
        "n_reference_images": len(reference_outcomes),
        "n_target_images": len(target_outcomes),
        "reference_zero_outcome_images": int(
            np.count_nonzero(reference_outcomes.sum(axis=1) == 0)
        ),
        "target_zero_outcome_images": int(
            np.count_nonzero(target_outcomes.sum(axis=1) == 0)
        ),
        "permutations": permutations,
        **test,
    }
    for index, outcome in enumerate(OUTCOMES):
        row[f"reference_{outcome}_count"] = int(reference_counts[index])
        row[f"target_{outcome}_count"] = int(target_counts[index])
        row[f"reference_{outcome}_proportion"] = float(
            reference_proportions[index]
        )
        row[f"target_{outcome}_proportion"] = float(target_proportions[index])
        row[f"{outcome}_difference_percentage_points"] = float(
            100.0 * (target_proportions[index] - reference_proportions[index])
        )
    return row


def _write_csv(rows: Sequence[Mapping], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_summary(rows: Sequence[Mapping], group_fields: Sequence[str]) -> List[Dict]:
    grouped: Dict[Tuple, List[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    output = []
    for key, group in sorted(grouped.items()):
        significant = sum(bool(row["significant"]) for row in group)
        record = {field: value for field, value in zip(group_fields, key)}
        record.update(
            {
                "comparisons": len(group),
                "significant_differences": significant,
                "no_significant_difference_detected": len(group) - significant,
                "significant_proportion": significant / len(group),
                "median_total_variation": float(
                    np.median([row["total_variation"] for row in group])
                ),
                "max_total_variation": max(
                    row["total_variation"] for row in group
                ),
            }
        )
        output.append(record)
    return output


def _joint_summary(
    rows: Sequence[Mapping],
    target_datasets: Sequence[str],
) -> Dict:
    if len(target_datasets) != 2:
        return {}
    first_target, second_target = target_datasets
    indexed = {
        (
            row["model"],
            row["transformation"],
            row["variant_id"],
            row["target_dataset"],
        ): row
        for row in rows
    }
    units = sorted(
        {
            (row["model"], row["transformation"], row["variant_id"])
            for row in rows
        }
    )
    counts = {
        "different_for_both": 0,
        f"different_only_for_{first_target}": 0,
        f"different_only_for_{second_target}": 0,
        "no_difference_detected_for_either": 0,
    }
    for model, transformation, variant_id in units:
        first = indexed[
            (model, transformation, variant_id, first_target)
        ]["significant"]
        second = indexed[
            (model, transformation, variant_id, second_target)
        ]["significant"]
        if first and second:
            counts["different_for_both"] += 1
        elif first:
            counts[f"different_only_for_{first_target}"] += 1
        elif second:
            counts[f"different_only_for_{second_target}"] += 1
        else:
            counts["no_difference_detected_for_either"] += 1

    return {
        "total_model_settings": len(units),
        "counts": counts,
        "proportions": {
            key: value / len(units) if units else 0.0
            for key, value in counts.items()
        },
    }


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def _write_markdown_report(
    rows: Sequence[Mapping],
    summary: Mapping,
    output_path: Path,
    args: argparse.Namespace,
    variants: Sequence[Mapping],
) -> None:
    target_summary = summary["by_target"]
    model_summary = summary["by_target_and_model"]
    transformation_summary = summary["by_target_and_transformation"]
    joint = summary.get("joint", {})

    lines = [
        "# Detection Distribution Statistical Analysis",
        "",
        "## Question and design",
        "",
        (
            f"This analysis compares `{args.reference_dataset}` with "
            f"{', '.join(f'`{name}`' for name in args.target_datasets)} for every "
            "complete model, metamorphic test, and configured setting."
        ),
        "",
        (
            "Within each image, original detections were matched to transformed "
            "detections using the repository's confidence-prioritized, one-to-one "
            f"greedy matcher with a strict IoU > {args.iou_threshold:g} criterion. "
            "Geometric detections used the existing coordinate-corrected files."
        ),
        "",
        "Each image contributed one three-count cluster:",
        "",
        "- **Retained:** matched original/transformed detections.",
        "- **Lost:** unmatched original detections.",
        "- **Gained:** unmatched transformed detections.",
        "",
        (
            "The primary test was a Pearson 2×3 homogeneity statistic calibrated "
            f"with {args.permutations:,} whole-image label permutations. Keeping "
            "each image cluster intact avoids treating detections from the same "
            "image as independent. Total variation distance and Cramér's V are "
            "reported as effect sizes."
        ),
        "",
        (
            f"Benjamini–Hochberg correction was applied with scope "
            f"`{args.correction_scope}` at q ≤ {args.alpha:g}. A non-significant "
            "result means **no difference was detected**; it does not establish "
            "that two distributions are equivalent."
        ),
        "",
        "## Candidate statistical tests",
        "",
        "1. **Image-clustered permutation Pearson test (used).** Directly tests "
        "retained/lost/gained proportions, preserves within-image clustering, and "
        "has the clearest interpretation for this question.",
        "2. **Multivariate energy-distance permutation test.** Can detect any "
        "difference in normalized per-image outcome distributions, but is less "
        "directly interpretable and needs a convention for zero-outcome images.",
        "3. **PERMANOVA on per-image compositions.** Supports multivariate outcome "
        "profiles, but significance can reflect unequal within-dataset dispersion "
        "rather than a location difference.",
        "",
        "## Coverage",
        "",
        f"- Models: {', '.join(f'`{model}`' for model in summary['models'])}",
        f"- Base metamorphic tests: {len(set(v['transformation'] for v in variants))}",
        f"- Configured settings per model: {len(variants)}",
        f"- Dataset comparisons: {len(rows)}",
        f"- Images per dataset/model/setting: {rows[0]['n_reference_images']}",
        "",
        "## Overall results",
        "",
    ]
    lines.extend(
        _markdown_table(
            (
                "Comparison",
                "Tests",
                "Different",
                "No difference detected",
                "Different proportion",
                "Median TV",
            ),
            [
                (
                    f"{item['target_dataset']} vs {args.reference_dataset}",
                    item["comparisons"],
                    item["significant_differences"],
                    item["no_significant_difference_detected"],
                    _format_percent(item["significant_proportion"]),
                    f"{item['median_total_variation']:.4f}",
                )
                for item in target_summary
            ],
        )
    )

    if joint:
        lines.extend(["", "### Joint classification", ""])
        total = joint["total_model_settings"]
        counts = joint["counts"]
        proportions = joint["proportions"]
        lines.append(
            f"The denominator is {total} model-setting combinations "
            f"({len(summary['models'])} models × {len(variants)} settings)."
        )
        lines.append("")
        lines.extend(
            _markdown_table(
                ("Classification", "Count", "Proportion"),
                [
                    (
                        key.replace("_", " "),
                        counts[key],
                        _format_percent(proportions[key]),
                    )
                    for key in counts
                ],
            )
        )

    significant_rows = [row for row in rows if row["significant"]]
    largest_by_target = {}
    for target_dataset in args.target_datasets:
        target_rows = [
            row for row in rows if row["target_dataset"] == target_dataset
        ]
        if target_rows:
            largest_by_target[target_dataset] = max(
                target_rows, key=lambda row: row["total_variation"]
            )

    lines.extend(["", "## Key findings", ""])
    for item in target_summary:
        lines.append(
            f"- **`{item['target_dataset']}` versus `{args.reference_dataset}`:** "
            f"{item['significant_differences']} of {item['comparisons']} "
            f"comparisons ({_format_percent(item['significant_proportion'])}) "
            "showed an FDR-adjusted difference."
        )
    if significant_rows:
        lines.append(
            f"- Only {len(significant_rows)} comparison"
            f"{'s' if len(significant_rows) != 1 else ''} remained significant "
            "after correction. Its effect size should be considered alongside "
            "the p/q values."
        )
    else:
        lines.append(
            "- No comparison remained significant after multiple-testing "
            "correction."
        )
    for target_dataset, row in largest_by_target.items():
        lines.append(
            f"- The largest observed TV distance for `{target_dataset}` was "
            f"{row['total_variation']:.4f} (`{row['model']}`, "
            f"`{row['variant_id']}`; q={row['bh_q_value']:.4g}); "
            f"this was {'significant' if row['significant'] else 'not significant'}."
        )

    if significant_rows:
        lines.extend(["", "### FDR-significant comparisons", ""])
        lines.extend(
            _markdown_table(
                (
                    "Target",
                    "Model",
                    "Variant",
                    "Test retained/lost/gained",
                    "Target retained/lost/gained",
                    "TV",
                    "Cramér's V",
                    "p",
                    "q",
                ),
                [
                    (
                        row["target_dataset"],
                        row["model"],
                        row["variant_id"],
                        "/".join(
                            f"{row[f'reference_{name}_proportion']:.3f}"
                            for name in OUTCOMES
                        ),
                        "/".join(
                            f"{row[f'target_{name}_proportion']:.3f}"
                            for name in OUTCOMES
                        ),
                        f"{row['total_variation']:.4f}",
                        f"{row['cramers_v']:.4f}",
                        f"{row['permutation_p_value']:.4g}",
                        f"{row['bh_q_value']:.4g}",
                    )
                    for row in sorted(
                        significant_rows,
                        key=lambda row: (
                            row["bh_q_value"],
                            -row["total_variation"],
                        ),
                    )
                ],
            )
        )
        if len(significant_rows) == 1:
            row = significant_rows[0]
            lines.extend(
                [
                    "",
                    (
                        f"For `{row['target_dataset']}` versus "
                        f"`{args.reference_dataset}`, `{row['model']}` "
                        f"`{row['variant_id']}` had "
                        f"{row['retained_difference_percentage_points']:+.2f} "
                        "percentage points retained, "
                        f"{row['lost_difference_percentage_points']:+.2f} points "
                        "lost, and "
                        f"{row['gained_difference_percentage_points']:+.2f} points "
                        "gained. The small TV and Cramér's V values indicate that "
                        "the statistically detectable difference is modest in "
                        "magnitude."
                    ),
                ]
            )

    lines.extend(["", "## Results by model", ""])
    lines.extend(
        _markdown_table(
            (
                "Target",
                "Model",
                "Different / total",
                "Different proportion",
                "Median TV",
                "Maximum TV",
            ),
            [
                (
                    item["target_dataset"],
                    item["model"],
                    f"{item['significant_differences']} / {item['comparisons']}",
                    _format_percent(item["significant_proportion"]),
                    f"{item['median_total_variation']:.4f}",
                    f"{item['max_total_variation']:.4f}",
                )
                for item in model_summary
            ],
        )
    )

    lines.extend(["", "## Results by metamorphic test", ""])
    lines.extend(
        _markdown_table(
            (
                "Target",
                "Transformation",
                "Different / total",
                "Different proportion",
                "Median TV",
            ),
            [
                (
                    item["target_dataset"],
                    item["transformation"],
                    f"{item['significant_differences']} / {item['comparisons']}",
                    _format_percent(item["significant_proportion"]),
                    f"{item['median_total_variation']:.4f}",
                )
                for item in transformation_summary
            ],
        )
    )

    example_rows = [
        row for row in rows if row["variant_id"] == "imgStretch_2"
    ]
    if example_rows:
        lines.extend(["", "## Worked example: `imgStretch_2`", ""])
        lines.append(
            "Setting 2 has `height_ratio=1.0` and `width_ratio=1.4`. Outcome "
            "proportions and corrected decisions are:"
        )
        lines.append("")
        lines.extend(
            _markdown_table(
                (
                    "Target",
                    "Model",
                    "Test retained/lost/gained",
                    "Target retained/lost/gained",
                    "TV",
                    "p",
                    "q",
                    "Decision",
                ),
                [
                    (
                        row["target_dataset"],
                        row["model"],
                        "/".join(
                            f"{row[f'reference_{name}_proportion']:.3f}"
                            for name in OUTCOMES
                        ),
                        "/".join(
                            f"{row[f'target_{name}_proportion']:.3f}"
                            for name in OUTCOMES
                        ),
                        f"{row['total_variation']:.4f}",
                        f"{row['permutation_p_value']:.4g}",
                        f"{row['bh_q_value']:.4g}",
                        (
                            "different"
                            if row["significant"]
                            else "no difference detected"
                        ),
                    )
                    for row in example_rows
                ],
            )
        )

    strongest = sorted(
        rows, key=lambda row: -row["total_variation"]
    )[:15]
    lines.extend(["", "## Largest effect sizes (regardless of significance)", ""])
    lines.append(
        "These rows are ranked by total variation distance, not by p-value. "
        "They are useful for practical follow-up but should not be called "
        "statistically different unless the final column says `different`."
    )
    lines.append("")
    lines.extend(
        _markdown_table(
            (
                "Target",
                "Model",
                "Variant",
                "TV",
                "Cramér's V",
                "p",
                "q",
                "Retained Δ pp",
                "Lost Δ pp",
                "Gained Δ pp",
                "Decision",
            ),
            [
                (
                    row["target_dataset"],
                    row["model"],
                    row["variant_id"],
                    f"{row['total_variation']:.4f}",
                    f"{row['cramers_v']:.4f}",
                    f"{row['permutation_p_value']:.4g}",
                    f"{row['bh_q_value']:.4g}",
                    f"{row['retained_difference_percentage_points']:+.2f}",
                    f"{row['lost_difference_percentage_points']:+.2f}",
                    f"{row['gained_difference_percentage_points']:+.2f}",
                    (
                        "different"
                        if row["significant"]
                        else "no difference detected"
                    ),
                )
                for row in strongest
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- The test addresses the retained/lost/gained composition, not "
            "semantic class agreement or model accuracy against ground truth.",
            "- Matching is class-agnostic because that is the behavior of the "
            "current original-to-transformed matcher.",
            "- A significant result says the dataset's metamorphic detection "
            "response differs; it does not identify dataset quality as the cause.",
            "- A non-significant result is not an equivalence result. A separate "
            "equivalence margin would be needed to claim practical sameness.",
            "- Monte Carlo p-values have minimum resolution "
            f"{1.0 / (args.permutations + 1):.6g}.",
            "",
            "## Reproduction",
            "",
            "```bash",
            (
                "venv/bin/python analysis/utils/detection_distribution_tests.py "
                f"--iou-threshold {args.iou_threshold:g} "
                f"--permutations {args.permutations} "
                f"--alpha {args.alpha:g} "
                f"--correction-scope {args.correction_scope}"
            ),
            "```",
            "",
            "Detailed results are in `detection_distribution_tests.csv`; the "
            "machine-readable aggregate summary is in "
            "`detection_distribution_summary.json`.",
            "",
            "## Statistical references",
            "",
            "- Székely, G. J. and Rizzo, M. L. (2013), "
            "[Energy statistics: A class of statistics based on "
            "distances](https://doi.org/10.1016/j.jspi.2013.03.018).",
            "- Anderson, M. J. (2001), "
            "[A new method for non-parametric multivariate analysis of "
            "variance](https://doi.org/10.1111/j.1442-9993.2001.tb00081.x).",
            "- Benjamini, Y. and Hochberg, Y. (1995), "
            "[Controlling the false discovery rate](https://doi.org/"
            "10.1111/j.2517-6161.1995.tb02031.x).",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    results_root = args.results_root.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    datasets = [args.reference_dataset, *args.target_datasets]
    variants = _load_variant_config(config_path, args.transformations)

    discovered_models = _discover_complete_models(
        results_root, datasets, variants
    )
    if args.models:
        missing_models = set(args.models) - set(discovered_models)
        if missing_models:
            raise ValueError(
                "Requested models do not have complete inputs across all selected "
                f"datasets/settings: {sorted(missing_models)}"
            )
        models = list(args.models)
    else:
        models = discovered_models
    if not models:
        raise ValueError("No models have complete inputs for the selected analysis.")

    total_comparisons = len(models) * len(variants) * len(args.target_datasets)
    print(
        f"Running {total_comparisons} comparisons: {len(models)} models × "
        f"{len(variants)} settings × {len(args.target_datasets)} targets"
    )
    print(
        f"Matcher: confidence-ordered greedy one-to-one, strict IoU > "
        f"{args.iou_threshold:g}"
    )
    print(f"Permutations per comparison: {args.permutations:,}")

    rows: List[Dict] = []
    completed = 0
    for model in models:
        for variant in variants:
            for target_dataset in args.target_datasets:
                rows.append(
                    _build_result_row(
                        results_root=results_root,
                        reference_dataset=args.reference_dataset,
                        target_dataset=target_dataset,
                        model=model,
                        variant=variant,
                        iou_threshold=args.iou_threshold,
                        permutations=args.permutations,
                        batch_size=args.permutation_batch_size,
                        base_seed=args.seed,
                    )
                )
                completed += 1
            if completed % max(1, 10 * len(args.target_datasets)) == 0:
                print(f"  completed {completed}/{total_comparisons}")

    _apply_multiple_testing_correction(
        rows, args.correction_scope, args.alpha
    )
    rows.sort(
        key=lambda row: (
            row["target_dataset"],
            row["model"],
            row["transformation"],
            row["setting"],
        )
    )

    summary = {
        "method": {
            "test": "image-clustered permutation Pearson chi-square homogeneity",
            "outcomes": list(OUTCOMES),
            "matching": "confidence-ordered greedy one-to-one IoU",
            "iou_rule": f"strictly greater than {args.iou_threshold:g}",
            "permutations": args.permutations,
            "seed": args.seed,
            "alpha": args.alpha,
            "multiple_testing": "Benjamini-Hochberg",
            "correction_scope": args.correction_scope,
        },
        "reference_dataset": args.reference_dataset,
        "target_datasets": list(args.target_datasets),
        "models": models,
        "transformations": sorted(
            {variant["transformation"] for variant in variants}
        ),
        "settings_per_model": len(variants),
        "total_comparisons": len(rows),
        "by_target": _group_summary(rows, ("target_dataset",)),
        "by_target_and_model": _group_summary(
            rows, ("target_dataset", "model")
        ),
        "by_target_and_transformation": _group_summary(
            rows, ("target_dataset", "transformation")
        ),
        "joint": _joint_summary(rows, args.target_datasets),
    }

    csv_path = output_dir / "detection_distribution_tests.csv"
    summary_path = output_dir / "detection_distribution_summary.json"
    report_path = output_dir / "DETECTION_DISTRIBUTION_ANALYSIS.md"
    _write_csv(rows, csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown_report(rows, summary, report_path, args, variants)

    print(f"Detailed results: {csv_path}")
    print(f"Summary:          {summary_path}")
    print(f"Analysis report:  {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
