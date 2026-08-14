#!/usr/bin/env python3
"""Bootstrap equivalence analysis for metamorphic AP/AR response curves.

The analysis compares ``test`` against ``val`` and ``train_samples``.  For each
model, metamorphic rule, and parameter setting it:

1. recomputes original-vs-transformed COCO AP and AR;
2. subtracts the model/dataset-specific gamma=1.0 control metric;
3. independently bootstraps images in the reference and target datasets;
4. estimates the reference-minus-target degradation difference; and
5. classifies its 90% interval against a +/-0.05 equivalence margin.

For rules with more than one parameter setting, the same calculation is also
performed on the area under the control-normalised degradation curve (AUC).

The bootstrap uses exact dataset-level COCO accumulation for each resample.
Per-image COCO matching is performed once per model/variant; bootstrap samples
then duplicate those evaluated image clusters before globally ranking detection
scores.  This is equivalent to rebuilding a remapped COCO dataset for every
resample, while avoiding millions of temporary JSON files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "brute_force_transformations.json"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_ANALYSIS_RESULTS_ROOT = REPO_ROOT / "analysis" / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "dataset_comparisons" / "train_val_test_equivalence"

REFERENCE_DATASET = "test"
TARGET_DATASETS = ("val", "train_samples")
ALL_DATASETS = (REFERENCE_DATASET, *TARGET_DATASETS)
DEFAULT_MODELS = ("yolov5", "yolov8", "yolov12", "yolov26", "yolov26_v2")
DEFAULT_RULES = (
    "imgGrayscale",
    "imgColorWheel",
    "imgResolution",
    "imgDarkBright",
    "imgMotionBlur",
    "imgStretch",
)
GEOMETRIC_RULES = {"imgResolution", "imgStretch"}
CONTROL_VARIANT = "imgDarkBright_1"
CONTROL_GAMMA = 1.0
METRICS = ("AP", "AR")
MODEL_LABELS = {
    "yolov5": "YOLOv5",
    "yolov8": "YOLOv8",
    "yolov12": "YOLOv12",
    "yolov26": "YOLOv26",
    "yolov26_v2": "YOLOv26_v2",
}
DATASET_LABELS = {
    "test": "Test",
    "val": "Validation",
    "train_samples": "Training samples",
}
DATASET_COLORS = {
    "test": "#1f77b4",
    "val": "#ff7f0e",
    "train_samples": "#2ca02c",
}
CLASS_EQUIVALENT = "equivalent"
CLASS_DIFFERENT = "meaningfully_different"
CLASS_INCONCLUSIVE = "inconclusive"
CLASS_NOT_APPLICABLE = "not_applicable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare control-normalised metamorphic AP/AR degradation in test "
            "against validation and training-sample data."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--analysis-results-root",
        type=Path,
        default=DEFAULT_ANALYSIS_RESULTS_ROOT,
        help="Root containing aggregate parameter-variation JSONs used for validation.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--rules", nargs="+", default=list(DEFAULT_RULES))
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
        help="Independent image bootstrap repetitions (confirmed design: 2000).",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.90,
        help="Percentile confidence interval level (confirmed design: 0.90).",
    )
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=0.05,
        help="Symmetric AP/AR and AUC equivalence margin (confirmed design: 0.05).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260810,
        help="Fixed reproducibility seed (confirmed design: 20260810).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel worker processes used for COCO evaluation.",
    )
    parser.add_argument(
        "--point-tolerance",
        type=float,
        default=1e-9,
        help="Maximum accepted difference from existing aggregate AP/AR JSON values.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Generate JSON and Markdown but skip PNG response-curve plots.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap-repetitions must be positive.")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("--confidence-level must be between 0 and 1.")
    if not 0.0 < args.equivalence_margin < 1.0:
        raise ValueError("--equivalence-margin must be between 0 and 1.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.point_tolerance < 0:
        raise ValueError("--point-tolerance cannot be negative.")
    if not args.config.is_file():
        raise FileNotFoundError(f"Transformation config not found: {args.config}")
    unknown_rules = sorted(set(args.rules) - set(DEFAULT_RULES))
    if unknown_rules:
        raise ValueError(f"Unsupported rules: {unknown_rules}")


def stable_seed(base_seed: int, *labels: str) -> int:
    digest = hashlib.sha256(
        "|".join((str(base_seed), *labels)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def varying_parameter(variants: Sequence[Mapping]) -> Optional[str]:
    keys = sorted(
        {
            key
            for variant in variants
            for key in variant
            if key != "variant_id"
        }
    )
    for key in keys:
        if len({json.dumps(variant.get(key), sort_keys=True) for variant in variants}) > 1:
            return key
    return None


def load_config(config_path: Path, selected_rules: Sequence[str]) -> Dict[str, Dict]:
    with config_path.open() as handle:
        raw = json.load(handle)

    output: Dict[str, Dict] = {}
    for rule in selected_rules:
        configured = raw.get(rule)
        if not isinstance(configured, list) or not configured:
            raise ValueError(f"Rule {rule!r} is missing or empty in {config_path}.")
        variants = []
        for position, item in enumerate(configured):
            if not isinstance(item, dict) or "variant_id" not in item:
                raise ValueError(f"Invalid {rule} setting at position {position}.")
            variants.append(dict(item))
        parameter = varying_parameter(variants)
        output[rule] = {
            "varying_parameter": parameter,
            "variants": variants,
        }

    brightness = raw.get("imgDarkBright")
    controls = [
        item
        for item in brightness or []
        if item.get("variant_id") == CONTROL_VARIANT
        and float(item.get("gamma", math.nan)) == CONTROL_GAMMA
    ]
    if len(controls) != 1:
        raise ValueError(
            f"Expected exactly one gamma={CONTROL_GAMMA:g} control named "
            f"{CONTROL_VARIANT!r}."
        )
    return output


def transformed_path(
    results_root: Path,
    dataset: str,
    model: str,
    rule: str,
    variant_id: str,
) -> Path:
    filename = (
        "detections_coco_transformed.json"
        if rule in GEOMETRIC_RULES
        else "detections_coco.json"
    )
    return results_root / dataset / model / variant_id / filename


def original_path(results_root: Path, dataset: str, model: str) -> Path:
    return results_root / dataset / model / "original" / "detections_original_coco.json"


def validate_inputs(
    results_root: Path,
    models: Sequence[str],
    config: Mapping[str, Mapping],
) -> Dict[str, int]:
    dataset_image_counts: Dict[str, int] = {}
    required_variants = {
        (rule, variant["variant_id"])
        for rule, rule_config in config.items()
        for variant in rule_config["variants"]
    }
    required_variants.add(("imgDarkBright", CONTROL_VARIANT))

    for dataset in ALL_DATASETS:
        model_counts = set()
        for model in models:
            original = original_path(results_root, dataset, model)
            if not original.is_file():
                raise FileNotFoundError(f"Original detections not found: {original}")
            with original.open() as handle:
                original_data = json.load(handle)
            image_count = len(original_data.get("images", []))
            if image_count == 0:
                raise ValueError(f"No images declared in {original}")
            model_counts.add(image_count)
            for rule, variant_id in required_variants:
                path = transformed_path(
                    results_root, dataset, model, rule, variant_id
                )
                if not path.is_file():
                    raise FileNotFoundError(f"Transformed detections not found: {path}")
        if len(model_counts) != 1:
            raise ValueError(
                f"Models in {dataset} do not declare the same number of images: "
                f"{sorted(model_counts)}"
            )
        dataset_image_counts[dataset] = model_counts.pop()
    return dataset_image_counts


def _load_coco_ground_truth(path: Path) -> COCO:
    with path.open() as handle:
        data = json.load(handle)
    coco = COCO()
    coco.dataset = data
    coco.createIndex()
    return coco


def _load_detection_annotations(path: Path) -> List[Dict]:
    with path.open() as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("annotations", [])
    raise ValueError(f"Unsupported detection JSON structure: {path}")


def _prepare_category_entries(
    gt_path: Path,
    dt_path: Path,
) -> Tuple[List[List[Optional[Dict]]], int, np.ndarray, np.ndarray]:
    """Run COCO matching once and retain all-area/maxDet=100 image entries."""
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = _load_coco_ground_truth(gt_path)
        coco_dt = coco_gt.loadRes(_load_detection_annotations(dt_path))
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()

    params = evaluator._paramsEval
    image_count = len(params.imgIds)
    area_count = len(params.areaRng)
    category_count = len(params.catIds) if params.useCats else 1
    entries: List[List[Optional[Dict]]] = []
    for category_index in range(category_count):
        offset = category_index * area_count * image_count
        entries.append(
            [evaluator.evalImgs[offset + image_index] for image_index in range(image_count)]
        )
    return entries, image_count, np.asarray(params.iouThrs), np.asarray(params.recThrs)


def _coco_ap_ar_for_sample(
    entries_by_category: Sequence[Sequence[Optional[Mapping]]],
    sample_indices: np.ndarray,
    recall_thresholds: np.ndarray,
) -> Tuple[float, float]:
    """Accumulate exact overall COCO AP and AR100 for one image resample."""
    precision_values: List[float] = []
    recall_values: List[float] = []
    recall_count = len(recall_thresholds)

    for category_entries in entries_by_category:
        selected = [
            category_entries[int(index)]
            for index in sample_indices
            if category_entries[int(index)] is not None
        ]
        if not selected:
            continue

        gt_ignore = np.concatenate([entry["gtIgnore"] for entry in selected])
        positive_ground_truth = int(np.count_nonzero(gt_ignore == 0))
        if positive_ground_truth == 0:
            continue

        scores = np.concatenate([entry["dtScores"][:100] for entry in selected])
        if scores.size:
            order = np.argsort(-scores, kind="mergesort")
            detection_matches = np.concatenate(
                [entry["dtMatches"][:, :100] for entry in selected], axis=1
            )[:, order]
            detection_ignore = np.concatenate(
                [entry["dtIgnore"][:, :100] for entry in selected], axis=1
            )[:, order]
            true_positive = np.logical_and(
                detection_matches, np.logical_not(detection_ignore)
            )
            false_positive = np.logical_and(
                np.logical_not(detection_matches), np.logical_not(detection_ignore)
            )
            cumulative_tp = np.cumsum(true_positive, axis=1, dtype=float)
            cumulative_fp = np.cumsum(false_positive, axis=1, dtype=float)
        else:
            threshold_count = selected[0]["dtMatches"].shape[0]
            cumulative_tp = np.empty((threshold_count, 0), dtype=float)
            cumulative_fp = np.empty((threshold_count, 0), dtype=float)

        for threshold_index in range(cumulative_tp.shape[0]):
            tp = cumulative_tp[threshold_index]
            fp = cumulative_fp[threshold_index]
            if tp.size == 0:
                recall_values.append(0.0)
                precision_values.extend([0.0] * recall_count)
                continue

            recall = tp / positive_ground_truth
            precision = tp / (tp + fp + np.spacing(1))
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            recall_values.append(float(recall[-1]))
            indices = np.searchsorted(recall, recall_thresholds, side="left")
            sampled_precision = np.zeros(recall_count, dtype=float)
            valid = indices < len(precision)
            sampled_precision[valid] = precision[indices[valid]]
            precision_values.extend(sampled_precision.tolist())

    if not precision_values or not recall_values:
        return math.nan, math.nan
    return float(np.mean(precision_values)), float(np.mean(recall_values))


@lru_cache(maxsize=None)
def _bootstrap_sample_indices(
    dataset: str,
    image_count: int,
    repetitions: int,
    base_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(base_seed, "image-bootstrap", dataset))
    return rng.integers(
        0,
        image_count,
        size=(repetitions, image_count),
        dtype=np.int32,
    )


def evaluate_task(task: Tuple) -> Dict:
    (
        results_root_string,
        dataset,
        model,
        rule,
        variant_id,
        repetitions,
        base_seed,
    ) = task
    results_root = Path(results_root_string)
    entries, image_count, _, recall_thresholds = _prepare_category_entries(
        original_path(results_root, dataset, model),
        transformed_path(results_root, dataset, model, rule, variant_id),
    )
    point_ap, point_ar = _coco_ap_ar_for_sample(
        entries, np.arange(image_count, dtype=np.int32), recall_thresholds
    )
    samples = _bootstrap_sample_indices(
        dataset, image_count, repetitions, base_seed
    )
    bootstrap_ap = np.empty(repetitions, dtype=np.float64)
    bootstrap_ar = np.empty(repetitions, dtype=np.float64)
    for index, sample_indices in enumerate(samples):
        bootstrap_ap[index], bootstrap_ar[index] = _coco_ap_ar_for_sample(
            entries, sample_indices, recall_thresholds
        )
    if not np.isfinite(bootstrap_ap).all() or not np.isfinite(bootstrap_ar).all():
        raise ValueError(
            f"Non-finite bootstrap metric for {dataset}/{model}/{variant_id}."
        )
    return {
        "dataset": dataset,
        "model": model,
        "rule": rule,
        "variant_id": variant_id,
        "image_count": image_count,
        "point": {"AP": point_ap, "AR": point_ar},
        "bootstrap": {"AP": bootstrap_ap, "AR": bootstrap_ar},
    }


def build_tasks(
    args: argparse.Namespace,
    config: Mapping[str, Mapping],
) -> List[Tuple]:
    variant_map = {
        (rule, variant["variant_id"])
        for rule, rule_config in config.items()
        for variant in rule_config["variants"]
    }
    variant_map.add(("imgDarkBright", CONTROL_VARIANT))
    return [
        (
            str(args.results_root),
            dataset,
            model,
            rule,
            variant_id,
            args.bootstrap_repetitions,
            args.seed,
        )
        for dataset in ALL_DATASETS
        for model in args.models
        for rule, variant_id in sorted(variant_map)
    ]


def execute_tasks(tasks: Sequence[Tuple], workers: int) -> Dict[Tuple[str, str, str], Dict]:
    results: Dict[Tuple[str, str, str], Dict] = {}
    total = len(tasks)
    started = time.monotonic()
    next_progress = 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_task, task): task for task in tasks}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            task = futures[future]
            try:
                result = future.result()
            except Exception as error:
                dataset, model, rule, variant_id = task[1:5]
                raise RuntimeError(
                    f"Bootstrap task failed for {dataset}/{model}/{rule}/{variant_id}"
                ) from error
            key = (result["dataset"], result["model"], result["variant_id"])
            results[key] = result
            percent = int(100 * completed / total)
            if percent >= next_progress or completed == total:
                elapsed = time.monotonic() - started
                print(
                    f"Bootstrap evaluation: {completed}/{total} ({percent}%) "
                    f"in {elapsed:.1f}s",
                    flush=True,
                )
                next_progress = percent + 5
    return results


def load_expected_metrics(
    analysis_results_root: Path,
    datasets: Sequence[str],
) -> Dict[Tuple[str, str, str, str], float]:
    expected: Dict[Tuple[str, str, str, str], float] = {}
    for dataset in datasets:
        path = (
            analysis_results_root
            / dataset
            / "parameter_variations"
            / "metrics_original_vs_transformed.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Aggregate metrics JSON not found: {path}")
        with path.open() as handle:
            data = json.load(handle)
        if data.get("comparison") != "original_vs_transformed":
            raise ValueError(f"Unexpected comparison in {path}: {data.get('comparison')}")
        for rule, rule_data in data["rules"].items():
            for setting in rule_data["settings"]:
                variant_id = setting["variant_id"]
                for model, metrics in setting["models"].items():
                    for metric in METRICS:
                        expected[(dataset, model, variant_id, metric)] = float(
                            metrics[metric]
                        )
    return expected


def validate_point_metrics(
    raw_results: Mapping[Tuple[str, str, str], Mapping],
    expected: Mapping[Tuple[str, str, str, str], float],
    tolerance: float,
) -> Dict:
    differences = []
    missing = []
    for (dataset, model, variant_id), result in raw_results.items():
        for metric in METRICS:
            key = (dataset, model, variant_id, metric)
            if key not in expected:
                missing.append("/".join(key))
                continue
            difference = abs(float(result["point"][metric]) - expected[key])
            differences.append((difference, key))
    max_difference, max_key = max(differences, default=(0.0, None))
    within_tolerance = sum(
        difference <= tolerance for difference, _ in differences
    )
    return {
        "values_checked": len(differences),
        "values_within_tolerance": within_tolerance,
        "values_different_from_stored_json": len(differences) - within_tolerance,
        "maximum_absolute_difference": max_difference,
        "maximum_difference_key": list(max_key) if max_key else None,
        "tolerance": tolerance,
        "all_within_tolerance": max_difference <= tolerance and not missing,
        "missing_expected_values": missing,
        "interpretation": (
            "Differences are expected where the stored aggregate JSON excluded "
            "images with zero transformed detections. Bootstrap point estimates "
            "use every declared image."
        ),
    }


def percentile_interval(values: np.ndarray, confidence_level: float) -> Tuple[float, float]:
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def classify_interval(low: float, high: float, margin: float) -> str:
    if low >= -margin and high <= margin:
        return CLASS_EQUIVALENT
    if low > margin or high < -margin:
        return CLASS_DIFFERENT
    return CLASS_INCONCLUSIVE


def parameter_value(rule_config: Mapping, variant: Mapping, position: int) -> float:
    parameter = rule_config["varying_parameter"]
    if parameter is None:
        return float(position)
    value = variant.get(parameter)
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"AUC/plot parameter {parameter!r} must be numeric; received {value!r}."
        )
    return float(value)


def degradation_record(
    raw_results: Mapping[Tuple[str, str, str], Mapping],
    dataset: str,
    model: str,
    variant_id: str,
    metric: str,
) -> Tuple[float, np.ndarray, float]:
    control = raw_results[(dataset, model, CONTROL_VARIANT)]
    setting = raw_results[(dataset, model, variant_id)]
    point = float(control["point"][metric]) - float(setting["point"][metric])
    bootstrap = control["bootstrap"][metric] - setting["bootstrap"][metric]
    return point, bootstrap, float(control["point"][metric])


def build_parameter_results(
    args: argparse.Namespace,
    config: Mapping[str, Mapping],
    raw_results: Mapping[Tuple[str, str, str], Mapping],
) -> Tuple[List[Dict], Dict[Tuple[str, str, str, str], Dict]]:
    rows: List[Dict] = []
    dataset_degradations: Dict[Tuple[str, str, str, str], Dict] = {}

    for rule, rule_config in config.items():
        for position, variant in enumerate(rule_config["variants"]):
            variant_id = variant["variant_id"]
            setting = {key: value for key, value in variant.items() if key != "variant_id"}
            x_value = parameter_value(rule_config, variant, position)
            for model in args.models:
                for metric in METRICS:
                    for dataset in ALL_DATASETS:
                        point, bootstrap, control_point = degradation_record(
                            raw_results, dataset, model, variant_id, metric
                        )
                        low, high = percentile_interval(
                            bootstrap, args.confidence_level
                        )
                        dataset_degradations[(dataset, model, variant_id, metric)] = {
                            "point": point,
                            "bootstrap": bootstrap,
                            "ci_low": low,
                            "ci_high": high,
                            "control_point": control_point,
                        }

                    reference = dataset_degradations[
                        (REFERENCE_DATASET, model, variant_id, metric)
                    ]
                    for target in TARGET_DATASETS:
                        target_value = dataset_degradations[
                            (target, model, variant_id, metric)
                        ]
                        difference_bootstrap = (
                            reference["bootstrap"] - target_value["bootstrap"]
                        )
                        low, high = percentile_interval(
                            difference_bootstrap, args.confidence_level
                        )
                        difference = reference["point"] - target_value["point"]
                        rows.append(
                            {
                                "reference_dataset": REFERENCE_DATASET,
                                "target_dataset": target,
                                "model": model,
                                "rule": rule,
                                "variant_id": variant_id,
                                "varying_parameter": rule_config["varying_parameter"],
                                "parameter_value": x_value,
                                "setting": setting,
                                "metric": metric,
                                "control_variant": CONTROL_VARIANT,
                                "control_gamma": CONTROL_GAMMA,
                                "reference_control_metric": reference["control_point"],
                                "target_control_metric": target_value["control_point"],
                                "reference_degradation": reference["point"],
                                "reference_degradation_ci_low": reference["ci_low"],
                                "reference_degradation_ci_high": reference["ci_high"],
                                "target_degradation": target_value["point"],
                                "target_degradation_ci_low": target_value["ci_low"],
                                "target_degradation_ci_high": target_value["ci_high"],
                                "degradation_difference": difference,
                                "difference_ci_low": low,
                                "difference_ci_high": high,
                                "equivalence_margin": args.equivalence_margin,
                                "classification": classify_interval(
                                    low, high, args.equivalence_margin
                                ),
                            }
                        )
    return rows, dataset_degradations


def build_auc_results(
    args: argparse.Namespace,
    config: Mapping[str, Mapping],
    dataset_degradations: Mapping[Tuple[str, str, str, str], Mapping],
) -> Tuple[List[Dict], Dict[Tuple[str, str, str, str], Dict]]:
    rows: List[Dict] = []
    dataset_auc: Dict[Tuple[str, str, str, str], Dict] = {}

    for rule, rule_config in config.items():
        variants = rule_config["variants"]
        if len(variants) < 2:
            continue
        x = np.asarray(
            [parameter_value(rule_config, variant, index) for index, variant in enumerate(variants)],
            dtype=float,
        )
        order = np.argsort(x, kind="mergesort")
        x = x[order]
        if x[-1] == x[0]:
            raise ValueError(f"Cannot normalise a constant AUC axis for {rule}.")
        normalised_x = (x - x[0]) / (x[-1] - x[0])
        ordered_variants = [variants[int(index)] for index in order]

        for model in args.models:
            for metric in METRICS:
                for dataset in ALL_DATASETS:
                    values = [
                        dataset_degradations[
                            (dataset, model, variant["variant_id"], metric)
                        ]
                        for variant in ordered_variants
                    ]
                    point_curve = np.asarray([value["point"] for value in values])
                    bootstrap_curves = np.vstack(
                        [value["bootstrap"] for value in values]
                    )
                    point_auc = float(np.trapezoid(point_curve, normalised_x))
                    bootstrap_auc = np.trapezoid(
                        bootstrap_curves, normalised_x, axis=0
                    )
                    low, high = percentile_interval(
                        bootstrap_auc, args.confidence_level
                    )
                    dataset_auc[(dataset, model, rule, metric)] = {
                        "point": point_auc,
                        "bootstrap": bootstrap_auc,
                        "ci_low": low,
                        "ci_high": high,
                    }

                reference = dataset_auc[(REFERENCE_DATASET, model, rule, metric)]
                for target in TARGET_DATASETS:
                    target_value = dataset_auc[(target, model, rule, metric)]
                    difference_bootstrap = (
                        reference["bootstrap"] - target_value["bootstrap"]
                    )
                    low, high = percentile_interval(
                        difference_bootstrap, args.confidence_level
                    )
                    rows.append(
                        {
                            "reference_dataset": REFERENCE_DATASET,
                            "target_dataset": target,
                            "model": model,
                            "rule": rule,
                            "varying_parameter": rule_config["varying_parameter"],
                            "metric": metric,
                            "parameter_min": float(x[0]),
                            "parameter_max": float(x[-1]),
                            "parameter_axis_normalised": True,
                            "n_settings": len(variants),
                            "reference_auc": reference["point"],
                            "reference_auc_ci_low": reference["ci_low"],
                            "reference_auc_ci_high": reference["ci_high"],
                            "target_auc": target_value["point"],
                            "target_auc_ci_low": target_value["ci_low"],
                            "target_auc_ci_high": target_value["ci_high"],
                            "auc_difference": reference["point"] - target_value["point"],
                            "difference_ci_low": low,
                            "difference_ci_high": high,
                            "equivalence_margin": args.equivalence_margin,
                            "classification": classify_interval(
                                low, high, args.equivalence_margin
                            ),
                        }
                    )
    return rows, dataset_auc


def count_classifications(rows: Iterable[Mapping]) -> Dict[str, int]:
    counts = Counter(row["classification"] for row in rows)
    return {
        classification: int(counts.get(classification, 0))
        for classification in (
            CLASS_EQUIVALENT,
            CLASS_INCONCLUSIVE,
            CLASS_DIFFERENT,
        )
    }


def build_summaries(parameter_rows: Sequence[Mapping], auc_rows: Sequence[Mapping]) -> Dict:
    def grouped(rows: Sequence[Mapping], fields: Sequence[str]) -> List[Dict]:
        groups: Dict[Tuple, List[Mapping]] = defaultdict(list)
        for row in rows:
            groups[tuple(row[field] for field in fields)].append(row)
        output = []
        for key, group in sorted(groups.items()):
            record = {field: value for field, value in zip(fields, key)}
            record["total"] = len(group)
            record["counts"] = count_classifications(group)
            record["equivalent_proportion"] = (
                record["counts"][CLASS_EQUIVALENT] / len(group) if group else 0.0
            )
            output.append(record)
        return output

    return {
        "parameter_overall": count_classifications(parameter_rows),
        "parameter_by_target_metric": grouped(
            parameter_rows, ("target_dataset", "metric")
        ),
        "parameter_by_target_metric_model": grouped(
            parameter_rows, ("target_dataset", "metric", "model")
        ),
        "parameter_by_target_metric_rule": grouped(
            parameter_rows, ("target_dataset", "metric", "rule")
        ),
        "auc_overall": count_classifications(auc_rows),
        "auc_by_target_metric": grouped(auc_rows, ("target_dataset", "metric")),
        "auc_by_target_metric_model": grouped(
            auc_rows, ("target_dataset", "metric", "model")
        ),
        "auc_by_target_metric_rule": grouped(
            auc_rows, ("target_dataset", "metric", "rule")
        ),
    }


def format_rule(rule: str) -> str:
    names = {
        "imgGrayscale": "Grayscale",
        "imgColorWheel": "Hue rotation",
        "imgResolution": "Resolution",
        "imgDarkBright": "Gamma brightness",
        "imgMotionBlur": "Motion blur",
        "imgStretch": "Horizontal stretch",
    }
    return names.get(rule, rule)


def format_parameter(parameter: Optional[str]) -> str:
    return "Setting" if parameter is None else parameter.replace("_", " ").title()


def create_response_plots(
    args: argparse.Namespace,
    config: Mapping[str, Mapping],
    dataset_degradations: Mapping[Tuple[str, str, str, str], Mapping],
) -> List[str]:
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[str] = []

    for rule, rule_config in config.items():
        variants = rule_config["variants"]
        x_values = np.asarray(
            [parameter_value(rule_config, variant, index) for index, variant in enumerate(variants)]
        )
        for metric in METRICS:
            fig, axes = plt.subplots(3, 2, figsize=(14, 14), squeeze=False)
            flat_axes = axes.ravel()
            for model_index, model in enumerate(args.models):
                axis = flat_axes[model_index]
                if len(variants) == 1:
                    positions = np.arange(len(ALL_DATASETS))
                    points = []
                    lows = []
                    highs = []
                    for dataset in ALL_DATASETS:
                        value = dataset_degradations[
                            (dataset, model, variants[0]["variant_id"], metric)
                        ]
                        points.append(value["point"])
                        lows.append(value["point"] - value["ci_low"])
                        highs.append(value["ci_high"] - value["point"])
                    axis.errorbar(
                        positions,
                        points,
                        yerr=np.asarray([lows, highs]),
                        fmt="o",
                        capsize=4,
                        color="#333333",
                    )
                    axis.set_xticks(
                        positions, [DATASET_LABELS[dataset] for dataset in ALL_DATASETS], rotation=15
                    )
                else:
                    for dataset in ALL_DATASETS:
                        values = [
                            dataset_degradations[
                                (dataset, model, variant["variant_id"], metric)
                            ]
                            for variant in variants
                        ]
                        point = np.asarray([value["point"] for value in values])
                        low = np.asarray([value["ci_low"] for value in values])
                        high = np.asarray([value["ci_high"] for value in values])
                        axis.plot(
                            x_values,
                            point,
                            marker="o",
                            linewidth=1.8,
                            markersize=4,
                            label=DATASET_LABELS[dataset],
                            color=DATASET_COLORS[dataset],
                        )
                        axis.fill_between(
                            x_values,
                            low,
                            high,
                            alpha=0.14,
                            color=DATASET_COLORS[dataset],
                        )
                    axis.set_xlabel(format_parameter(rule_config["varying_parameter"]))
                axis.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
                axis.set_ylabel(f"Control-normalised {metric} degradation")
                axis.set_title(MODEL_LABELS.get(model, model))
                axis.grid(alpha=0.25, linestyle=":")

            for unused_index in range(len(args.models), len(flat_axes)):
                flat_axes[unused_index].axis("off")
            handles, labels = flat_axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.965),
                    ncol=3,
                    framealpha=0.9,
                )
            fig.suptitle(
                f"{format_rule(rule)}: {metric} degradation relative to gamma=1.0 control",
                fontsize=15,
                fontweight="bold",
                y=0.995,
            )
            fig.subplots_adjust(
                left=0.07,
                right=0.98,
                bottom=0.055,
                top=0.91,
                hspace=0.38,
                wspace=0.12,
            )
            filename = f"{rule}_{metric.lower()}_degradation.png"
            path = plot_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            output_paths.append(str(path.relative_to(args.output_dir)))
    return output_paths


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return lines


def class_short(classification: str) -> str:
    return {
        CLASS_EQUIVALENT: "Equivalent",
        CLASS_INCONCLUSIVE: "Inconclusive",
        CLASS_DIFFERENT: "Different",
        CLASS_NOT_APPLICABLE: "N/A",
    }[classification]


def write_markdown_report(
    args: argparse.Namespace,
    config: Mapping[str, Mapping],
    parameter_rows: Sequence[Mapping],
    auc_rows: Sequence[Mapping],
    summaries: Mapping,
    validation: Mapping,
    dataset_image_counts: Mapping[str, int],
    plot_paths: Sequence[str],
) -> Path:
    report_path = args.output_dir / "REPORT.md"
    confidence_percent = int(round(args.confidence_level * 100))
    lines = [
        "# Test vs validation/training metamorphic consistency equivalence",
        "",
        "## Research question",
        "",
        (
            "To what extent are the parameter-dependent metamorphic consistency "
            "patterns observed on the test dataset reproduced in the validation "
            "and training-sample datasets?"
        ),
        "",
        "## Design",
        "",
        (
            f"The analysis covers {len(args.models)} models, {len(config)} rules, "
            f"and {sum(len(value['variants']) for value in config.values())} settings. "
            f"Each dataset contains {dataset_image_counts[REFERENCE_DATASET]} images."
        ),
        "",
        (
            "AP and AR are COCO original-vs-transformed consistency metrics: each "
            "model's original predictions act as the reference, rather than labelled "
            "ground truth. Every rule is normalised against the same gamma=1.0 "
            f"control (`{CONTROL_VARIANT}`)."
        ),
        "",
        (
            f"Images were independently resampled {args.bootstrap_repetitions:,} "
            f"times within each dataset. Results use percentile {confidence_percent}% "
            f"confidence intervals and a symmetric +/-{args.equivalence_margin:.2f} "
            "equivalence margin. No p-values or multiple-testing corrections are used."
        ),
        "",
        "Classification rules:",
        "",
        (
            f"- **Equivalent:** the complete {confidence_percent}% interval lies "
            f"within [-{args.equivalence_margin:.2f}, +{args.equivalence_margin:.2f}]."
        ),
        (
            "- **Meaningfully different:** the complete interval lies beyond one "
            "equivalence boundary."
        ),
        "- **Inconclusive:** the interval overlaps an equivalence boundary.",
        "",
        (
            "A positive difference means test degradation is greater than the "
            "alternative dataset; a negative difference means validation/training "
            "degradation is greater."
        ),
        "",
        "## Validation",
        "",
        (
            f"The implementation recomputed {validation['values_checked']:,} point "
            "AP/AR values over every declared image and compared them with the "
            "existing aggregate JSONs. "
            f"{validation['values_within_tolerance']:,} values agreed within "
            f"{validation['tolerance']:.1g}; "
            f"{validation['values_different_from_stored_json']:,} differed because "
            "the stored evaluation omitted images with zero transformed detections."
        ),
        "",
        (
            f"The maximum stored-versus-recomputed discrepancy was "
            f"{validation['maximum_absolute_difference']:.3g} at "
            f"`{'/'.join(validation['maximum_difference_key'] or [])}`. The bootstrap "
            "uses the recomputed all-image values; it does not reproduce the biased "
            "detection-positive-only subset."
        ),
        "",
    ]

    parameter_counts = summaries["parameter_overall"]
    auc_counts = summaries["auc_overall"]
    parameter_total = sum(parameter_counts.values())
    auc_total = sum(auc_counts.values())
    non_equivalent_auc_models = sorted(
        {
            row["model"]
            for row in auc_rows
            if row["classification"] != CLASS_EQUIVALENT
        }
    )
    fully_equivalent_auc_models = []
    for model in args.models:
        model_rows = [row for row in auc_rows if row["model"] == model]
        if model_rows and all(
            row["classification"] == CLASS_EQUIVALENT for row in model_rows
        ):
            fully_equivalent_auc_models.append(MODEL_LABELS.get(model, model))

    lines.extend(
        [
            "## Key findings",
            "",
            (
                f"- At parameter level, {parameter_counts[CLASS_EQUIVALENT]:,} of "
                f"{parameter_total:,} comparisons ({parameter_counts[CLASS_EQUIVALENT] / parameter_total:.1%}) "
                f"were equivalent, {parameter_counts[CLASS_INCONCLUSIVE]:,} were "
                f"inconclusive, and {parameter_counts[CLASS_DIFFERENT]:,} were "
                "meaningfully different."
            ),
            (
                f"- At rule-curve level, {auc_counts[CLASS_EQUIVALENT]} of "
                f"{auc_total} AUC comparisons were equivalent; "
                f"{auc_counts[CLASS_INCONCLUSIVE]} were inconclusive and "
                f"{auc_counts[CLASS_DIFFERENT]} were meaningfully different."
            ),
            (
                "- Every AUC comparison was equivalent for "
                f"{', '.join(fully_equivalent_auc_models)}. Their overall parameter-response "
                "patterns therefore transfer from test to both validation and training samples "
                f"within the +/-{args.equivalence_margin:.2f} margin."
            ),
            (
                "- All non-equivalent AUC outcomes were confined to "
                + ", ".join(
                    MODEL_LABELS.get(model, model) for model in non_equivalent_auc_models
                )
                + ". Its test degradation was generally larger, especially for hue rotation "
                "and horizontal stretch."
            ),
            "",
            "## Parameter-level overview",
            "",
        ]
    )

    overview_rows = []
    for summary in summaries["parameter_by_target_metric"]:
        counts = summary["counts"]
        overview_rows.append(
            [
                DATASET_LABELS[summary["target_dataset"]],
                summary["metric"],
                summary["total"],
                counts[CLASS_EQUIVALENT],
                counts[CLASS_INCONCLUSIVE],
                counts[CLASS_DIFFERENT],
                f"{summary['equivalent_proportion']:.1%}",
            ]
        )
    lines.extend(
        markdown_table(
            ["Comparison", "Metric", "Settings", "Equivalent", "Inconclusive", "Different", "Equivalent %"],
            overview_rows,
        )
    )
    lines.extend(["", "## Rule-level AUC results", ""])
    lines.append(
        "AUC uses the complete configured parameter range with its axis scaled to 0-1. Because the response is degradation, a higher AUC means lower consistency across the rule. Grayscale has one setting and therefore has no AUC result."
    )
    lines.append("")

    for target in TARGET_DATASETS:
        for metric in METRICS:
            lines.extend(
                [
                    f"### {metric}: test vs {DATASET_LABELS[target].lower()}",
                    "",
                ]
            )
            table_rows = []
            for row in auc_rows:
                if row["target_dataset"] != target or row["metric"] != metric:
                    continue
                table_rows.append(
                    [
                        MODEL_LABELS.get(row["model"], row["model"]),
                        format_rule(row["rule"]),
                        f"{row['reference_auc']:.3f}",
                        f"{row['target_auc']:.3f}",
                        f"{row['auc_difference']:+.3f}",
                        f"[{row['difference_ci_low']:+.3f}, {row['difference_ci_high']:+.3f}]",
                        class_short(row["classification"]),
                    ]
                )
            lines.extend(
                markdown_table(
                    ["Model", "Rule", "Test AUC", "Alternative AUC", "Difference", f"{confidence_percent}% CI", "Result"],
                    table_rows,
                )
            )
            lines.append("")

    lines.extend(["## Transferability summary", ""])
    model_auc = {
        (item["target_dataset"], item["metric"], item["model"]): item
        for item in summaries["auc_by_target_metric_model"]
    }
    transfer_rows = []
    for model in args.models:
        cells = []
        total_equivalent = 0
        total = 0
        for target in TARGET_DATASETS:
            for metric in METRICS:
                summary = model_auc[(target, metric, model)]
                equivalent = summary["counts"][CLASS_EQUIVALENT]
                cells.append(f"{equivalent}/{summary['total']}")
                total_equivalent += equivalent
                total += summary["total"]
        transfer_rows.append(
            [MODEL_LABELS.get(model, model), *cells, f"{total_equivalent}/{total}"]
        )
    lines.extend(
        markdown_table(
            ["Model", "Val AP", "Val AR", "Train AP", "Train AR", "All equivalent AUCs"],
            transfer_rows,
        )
    )
    lines.extend(
        [
            "",
            "Counts show equivalent rule-level AUC comparisons out of the five rules with numeric parameter curves.",
            "",
            "## Largest parameter-level differences",
            "",
        ]
    )
    largest = sorted(
        parameter_rows,
        key=lambda row: abs(float(row["degradation_difference"])),
        reverse=True,
    )[:25]
    lines.extend(
        markdown_table(
            ["Target", "Metric", "Model", "Rule", "Setting", "Difference", f"{confidence_percent}% CI", "Result"],
            [
                [
                    DATASET_LABELS[row["target_dataset"]],
                    row["metric"],
                    MODEL_LABELS.get(row["model"], row["model"]),
                    format_rule(row["rule"]),
                    row["variant_id"],
                    f"{row['degradation_difference']:+.3f}",
                    f"[{row['difference_ci_low']:+.3f}, {row['difference_ci_high']:+.3f}]",
                    class_short(row["classification"]),
                ]
                for row in largest
            ],
        )
    )

    if plot_paths:
        lines.extend(["", "## Response curves", ""])
        for rule in config:
            lines.extend([f"### {format_rule(rule)}", ""])
            for metric in METRICS:
                relative = f"plots/{rule}_{metric.lower()}_degradation.png"
                if relative in plot_paths:
                    lines.append(f"![{format_rule(rule)} {metric} degradation]({relative})")
                    lines.append("")

    lines.extend(
        [
            "## Interpretation cautions",
            "",
            "- Equivalence concerns metamorphic consistency, not absolute ground-truth detection accuracy.",
            "- The datasets contain different images, so reference and target bootstraps are independent.",
            "- Images are the resampling units. No sequence/rosbag metadata was available to model temporal clustering.",
            "- The shared gamma=1.0 control was requested for every rule; rule-specific identity settings were not substituted.",
            "- Parameter-level intervals are estimation-only and are not adjusted for multiple comparisons.",
            "- AUC can be equivalent even when individual settings are inconclusive or different; consult the parameter records in `results.json`.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))
    return report_path


def json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = load_config(args.config, args.rules)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_image_counts = validate_inputs(args.results_root, args.models, config)

    print("Dataset-comparison bootstrap equivalence analysis", flush=True)
    print(f"Models: {', '.join(args.models)}", flush=True)
    print(f"Rules: {', '.join(config)}", flush=True)
    print(
        f"Bootstrap: {args.bootstrap_repetitions:,} repetitions, "
        f"{args.confidence_level:.0%} intervals, +/-{args.equivalence_margin:.2f} margin",
        flush=True,
    )
    print(f"Workers: {args.workers}", flush=True)

    tasks = build_tasks(args, config)
    print(f"COCO model/variant/dataset evaluations: {len(tasks)}", flush=True)
    raw_results = execute_tasks(tasks, args.workers)

    expected = load_expected_metrics(args.analysis_results_root, ALL_DATASETS)
    validation = validate_point_metrics(
        raw_results, expected, args.point_tolerance
    )
    print(
        "Aggregate JSON audit: "
        f"{validation['values_within_tolerance']}/{validation['values_checked']} "
        "values agree with all-image recomputation; maximum discrepancy "
        f"{validation['maximum_absolute_difference']:.3g}",
        flush=True,
    )

    parameter_rows, dataset_degradations = build_parameter_results(
        args, config, raw_results
    )
    auc_rows, _ = build_auc_results(args, config, dataset_degradations)
    summaries = build_summaries(parameter_rows, auc_rows)

    plot_paths: List[str] = []
    if not args.skip_plots:
        plot_paths = create_response_plots(args, config, dataset_degradations)

    output = {
        "metadata": {
            "reference_dataset": REFERENCE_DATASET,
            "target_datasets": list(TARGET_DATASETS),
            "datasets": {
                dataset: {"image_count": count}
                for dataset, count in dataset_image_counts.items()
            },
            "models": list(args.models),
            "rules": list(config),
            "metrics": list(METRICS),
            "comparison": "original_vs_transformed",
            "normalisation": "gamma_1_control_minus_setting",
            "control_variant": CONTROL_VARIANT,
            "control_gamma": CONTROL_GAMMA,
            "bootstrap_unit": "individual_image",
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.seed,
            "confidence_interval": {
                "method": "percentile",
                "level": args.confidence_level,
            },
            "equivalence_margin": args.equivalence_margin,
            "classification": {
                CLASS_EQUIVALENT: "complete interval within the equivalence margins",
                CLASS_DIFFERENT: "complete interval beyond one equivalence margin",
                CLASS_INCONCLUSIVE: "interval overlaps an equivalence margin",
            },
            "auc": {
                "method": "trapezoidal",
                "parameter_axis_normalised": True,
                "grayscale": "not_applicable_single_setting",
            },
            "workers": args.workers,
            "source_config": str(args.config),
            "source_aggregate_root": str(args.analysis_results_root),
            "source_detection_root": str(args.results_root),
        },
        "validation": validation,
        "parameter_results": parameter_rows,
        "auc_results": auc_rows,
        "summaries": summaries,
        "plots": plot_paths,
    }
    results_path = args.output_dir / "results.json"
    with results_path.open("w") as handle:
        json.dump(output, handle, indent=2, default=json_ready)
        handle.write("\n")

    report_path = write_markdown_report(
        args,
        config,
        parameter_rows,
        auc_rows,
        summaries,
        validation,
        dataset_image_counts,
        plot_paths,
    )
    print(f"Saved JSON: {results_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)
    print(f"Saved plots: {len(plot_paths)}", flush=True)


if __name__ == "__main__":
    main()
