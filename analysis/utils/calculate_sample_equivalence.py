#!/usr/bin/env python3
"""Bootstrap equivalence analysis for reduced test-set sample sizes.

This mirrors the statistical design used in
the RQ3 sample-size experiment. It compares two *arms drawn from the same
physical test dataset*:

- ``full``:   every declared test image (100%).
- ``sample``: a fixed, reproducibly-drawn subset containing
  ``--sample-fraction`` (default 0.5, i.e. 50%) of the test images.

Two independent analyses are performed for every model, metamorphic rule, and
parameter setting:

1. ``original_vs_transformed_consistency``: the raw COCO AP/AR bootstrap
   distribution obtained by treating each model's *original* detections as
   the reference and the *transformed* detections as the prediction. This
   asks whether transformed-vs-original consistency is equivalent between
   the full test set and the subsample.
2. ``gt_accuracy_degradation``: the paired difference between a model's
   ``original_vs_gt`` accuracy (its untransformed detections evaluated
   against real ground truth; a per-model baseline, constant across rules)
   and its ``transformed_vs_gt`` accuracy (the transformed detections for a
   given rule/setting evaluated against real ground truth). This asks
   whether the *cost* of the transformation, measured against real
   annotations rather than the model's own original predictions, is
   equivalent between the full test set and the subsample.

For rules with more than one parameter setting, the same equivalence
calculation is also performed on the area under each statistic's response
curve (AUC).

Reviewed hue mappings are evaluated as separate source-to-target cohorts using
source-specific original-vs-GT baselines. Excluded hue settings are omitted,
and hue AUC is not computed across the excluded intervals.

The bootstrap uses exact dataset-level COCO accumulation for each resample.
Per-image COCO matching is performed once per model/variant/comparison
against the full test dataset; bootstrap samples then duplicate those
evaluated image clusters before globally ranking detection scores. The
``sample`` arm's bootstrap draws are restricted to the fixed subsample of
images, so it measures the added sampling variability of evaluating on a
smaller image set, not a comparison between independent datasets.

Point AP/AR values for the ``full`` arm are taken directly from the
already-computed
``analysis/results/rq1/<dataset>/<model>/<variant_id>/<comparison>/stats.json``
files (and the per-model ``original_vs_gt/stats.json``) rather than being
recomputed. The script still recomputes a point estimate from per-image COCO
entries for the ``full`` arm purely to sanity-check it against the stored
``stats.json`` value (see ``validate_point_metrics``); recomputation is the
only option for the ``sample`` arm's point (no ``sample``-arm stats.json
exists) and for every bootstrap confidence interval.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from analysis.utils.color_wheel import (
    metadata as hue_metadata, mapping_for_key, baseline_key,
    remap_reference, select_predictions,
)
DEFAULT_CONFIG = REPO_ROOT / "config" / "brute_force_transformations.json"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_ANALYSIS_RESULTS_ROOT = REPO_ROOT / "analysis" / "results" / "rq1"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "base_images"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "results" / "rq3" / "sample_equivalence"

SOURCE_DATASET = "test"
DEFAULT_SAMPLE_FRACTION = 0.5
REFERENCE_ARM = "full"
TARGET_ARM = "sample"
ALL_ARMS = (REFERENCE_ARM, TARGET_ARM)
TARGET_ARMS = (TARGET_ARM,)
DEFAULT_MODELS = ("yolov5", "yolov8", "yolov12", "yolov26_v2")
DEFAULT_RULES = (
    "imgGrayscale",
    "imgColorWheel",
    "imgResolution",
    "imgDarkBright",
    "imgMotionBlur",
    "imgStretch",
)
GEOMETRIC_RULES = {"imgResolution", "imgStretch"}
METRICS = ("AP", "AR")
POINT_METRIC_STATS_KEY = {"AP": "AP", "AR": "AR100"}

# Comparison types evaluated per-image via COCOeval.
COMPARISON_ORIGINAL_VS_TRANSFORMED = "original_vs_transformed"
COMPARISON_TRANSFORMED_VS_GT = "transformed_vs_gt"
COMPARISON_ORIGINAL_VS_GT = "original_vs_gt"
# Sentinel "variant" key used for the per-model, rule-independent original_vs_gt baseline.
BASELINE_KEY = "original"

# The two equivalence analyses reported. Each is a "setting" statistic
# (per rule/variant) optionally reduced by a "baseline" statistic (a
# rule-independent, per-model reference computed the same way in both arms).
ANALYSIS_CONSISTENCY = "original_vs_transformed_consistency"
ANALYSIS_GT_ACCURACY = "gt_accuracy_degradation"
ANALYSES: Dict[str, Dict[str, Optional[str]]] = {
    ANALYSIS_CONSISTENCY: {
        "setting_comparison": COMPARISON_ORIGINAL_VS_TRANSFORMED,
        "baseline_comparison": None,
        "title": "Original-vs-transformed consistency",
        "statistic_description": (
            "raw COCO AP/AR with each model's original detections as the "
            "reference and the transformed detections as the prediction "
            "(no baseline subtraction)"
        ),
        "plot_suffix": "original_vs_transformed",
        "ylabel_template": "{metric} (original-vs-transformed consistency)",
    },
    ANALYSIS_GT_ACCURACY: {
        "setting_comparison": COMPARISON_TRANSFORMED_VS_GT,
        "baseline_comparison": COMPARISON_ORIGINAL_VS_GT,
        "title": "Ground-truth accuracy degradation from transformation",
        "statistic_description": (
            "original_vs_gt AP/AR (untransformed detections vs. real "
            "ground truth; a per-model baseline, source-specific for hue mappings) "
            "minus transformed_vs_gt AP/AR (transformed detections vs. "
            "real ground truth)"
        ),
        "plot_suffix": "gt_accuracy_degradation",
        "ylabel_template": "{metric} degradation vs ground truth (original - transformed)",
    },
}
MODEL_LABELS = {
    "yolov5": "YOLOv5",
    "yolov8": "YOLOv8",
    "yolov12": "YOLOv12",
    "yolov26_v2": "YOLOv26_v2",
}
ARM_LABELS = {
    "full": "Full test set (100%)",
    "sample": "Test subsample",
}
ARM_COLORS = {
    "full": "#1f77b4",
    "sample": "#d62728",
}
CLASS_EQUIVALENT = "equivalent"
CLASS_DIFFERENT = "meaningfully_different"
CLASS_INCONCLUSIVE = "inconclusive"
CLASS_NOT_APPLICABLE = "not_applicable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare metamorphic original-vs-transformed consistency and "
            "ground-truth accuracy degradation on the full test dataset "
            "against a fixed fraction subsample of it."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--analysis-results-root",
        type=Path,
        default=DEFAULT_ANALYSIS_RESULTS_ROOT,
        help=(
            "Root containing per-variant stats.json files "
            "(<root>/<dataset>/<model>/<variant_id>/<comparison>/stats.json "
            "and <root>/<dataset>/<model>/original_vs_gt/stats.json), used "
            "as the trusted point AP/AR source for the 'full' arm."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root containing <dataset>/annotations_coco.json ground-truth files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-dataset",
        type=str,
        default=SOURCE_DATASET,
        help="Physical dataset both arms are drawn from (confirmed design: test).",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
        help="Fraction of the source dataset images used for the 'sample' arm (confirmed design: 0.5).",
    )
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
        help="Fixed reproducibility seed.",
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
        help="Maximum accepted difference from the stored stats.json AP/AR values.",
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
    if not 0.0 < args.sample_fraction < 1.0:
        raise ValueError("--sample-fraction must be between 0 and 1 (exclusive).")
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
    for rule in dict.fromkeys(name.split("__", 1)[0] for name in selected_rules):
        configured = raw.get(rule)
        if not isinstance(configured, list) or not configured:
            raise ValueError(f"Rule {rule!r} is missing or empty in {config_path}.")
        variants = []
        for position, item in enumerate(configured):
            if not isinstance(item, dict) or "variant_id" not in item:
                raise ValueError(f"Invalid {rule} setting at position {position}.")
            variants.append(dict(item))
        if rule == "imgColorWheel":
            for variant in variants:
                for mapping in hue_metadata(variant["variant_id"])["included_mappings"]:
                    name = f"imgColorWheel__{mapping['key']}"
                    if "imgColorWheel" not in selected_rules and name not in selected_rules:
                        continue
                    cohort = output.setdefault(name, {"varying_parameter": "hue_rotation", "variants": []})
                    cohort["variants"].append({
                        **variant, "variant_id": f"{variant['variant_id']}__{mapping['key']}",
                        "mapping": mapping["mapping"],
                    })
            continue
        parameter = varying_parameter(variants)
        output[rule] = {
            "varying_parameter": parameter,
            "variants": variants,
        }
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
    return results_root / dataset / model / variant_id.split("__", 1)[0] / filename


def original_path(results_root: Path, dataset: str, model: str) -> Path:
    return results_root / dataset / model / "original" / "detections_original_coco.json"


def gt_annotations_path(data_root: Path, dataset: str) -> Path:
    return data_root / dataset / "annotations_coco.json"


def original_gt_detection_path(results_root: Path, dataset: str, model: str) -> Path:
    return results_root / dataset / model / "original" / "detections_for_gt.json"


def transformed_gt_detection_path(
    results_root: Path,
    dataset: str,
    model: str,
    rule: str,
    variant_id: str,
) -> Path:
    filename = (
        "detections_for_gt_transformed.json"
        if rule in GEOMETRIC_RULES
        else "detections_for_gt.json"
    )
    return results_root / dataset / model / variant_id.split("__", 1)[0] / filename


def stats_json_path(
    analysis_results_root: Path,
    dataset: str,
    model: str,
    variant_id: str,
    comparison: str,
) -> Path:
    return analysis_results_root / dataset / model / variant_id.split("__", 1)[0] / comparison / "stats.json"


def original_vs_gt_stats_path(
    analysis_results_root: Path, dataset: str, model: str
) -> Path:
    return analysis_results_root / dataset / model / "original_vs_gt" / "stats.json"


def variant_pairs(config: Mapping[str, Mapping]) -> List[Tuple[str, str]]:
    return sorted(
        {
            (rule, variant["variant_id"])
            for rule, rule_config in config.items()
            for variant in rule_config["variants"]
        }
    )


def validate_inputs(
    results_root: Path,
    data_root: Path,
    dataset: str,
    models: Sequence[str],
    config: Mapping[str, Mapping],
) -> int:
    required_variants = variant_pairs(config)

    gt_coco_path = gt_annotations_path(data_root, dataset)
    if not gt_coco_path.is_file():
        raise FileNotFoundError(f"Ground-truth annotations not found: {gt_coco_path}")
    with gt_coco_path.open() as handle:
        gt_image_count = len(json.load(handle).get("images", []))
    if gt_image_count == 0:
        raise ValueError(f"No images declared in {gt_coco_path}")

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

        original_gt_det = original_gt_detection_path(results_root, dataset, model)
        if not original_gt_det.is_file():
            raise FileNotFoundError(
                f"Original-vs-GT detections not found: {original_gt_det}"
            )

        for rule, variant_id in required_variants:
            path = transformed_path(results_root, dataset, model, rule, variant_id)
            if not path.is_file():
                raise FileNotFoundError(f"Transformed detections not found: {path}")
            gt_det_path = transformed_gt_detection_path(
                results_root, dataset, model, rule, variant_id
            )
            if not gt_det_path.is_file():
                raise FileNotFoundError(
                    f"Transformed-vs-GT detections not found: {gt_det_path}"
                )
    if len(model_counts) != 1:
        raise ValueError(
            f"Models in {dataset} do not declare the same number of images: "
            f"{sorted(model_counts)}"
        )
    image_count = model_counts.pop()
    if image_count != gt_image_count:
        raise ValueError(
            f"Ground-truth image count ({gt_image_count}) does not match "
            f"per-model original detection image count ({image_count}) for {dataset}."
        )
    return image_count


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
    variant_key: str = "",
) -> Tuple[List[List[Optional[Dict]]], int, np.ndarray, np.ndarray]:
    """Run COCO matching once and retain all-area/maxDet=100 image entries."""
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = _load_coco_ground_truth(gt_path)
        detections = _load_detection_annotations(dt_path)
        mapping = mapping_for_key(variant_key)
        if mapping is not None:
            reference = coco_gt.dataset
            detections = select_predictions(detections, reference, mapping)
            coco_gt = COCO()
            coco_gt.dataset = remap_reference(reference, mapping)
            coco_gt.createIndex()
        if detections:
            coco_dt = coco_gt.loadRes(detections)
        else:
            coco_dt = COCO()
            coco_dt.dataset = {"images": coco_gt.dataset["images"], "categories": coco_gt.dataset["categories"], "annotations": []}
            coco_dt.createIndex()
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
def _sample_population_indices(
    source_dataset: str,
    image_count: int,
    fraction: float,
    base_seed: int,
) -> np.ndarray:
    """Draw the fixed, reproducible subsample of images for the 'sample' arm."""
    sample_size = max(1, min(image_count, int(round(image_count * fraction))))
    rng = np.random.default_rng(
        stable_seed(base_seed, "sample-population", source_dataset, f"{fraction:.6f}")
    )
    return np.sort(rng.choice(image_count, size=sample_size, replace=False)).astype(np.int32)


def _arm_populations(
    source_dataset: str,
    image_count: int,
    fraction: float,
    base_seed: int,
) -> Dict[str, np.ndarray]:
    return {
        REFERENCE_ARM: np.arange(image_count, dtype=np.int32),
        TARGET_ARM: _sample_population_indices(
            source_dataset, image_count, fraction, base_seed
        ),
    }


@lru_cache(maxsize=None)
def _bootstrap_positions(
    arm: str,
    population_size: int,
    repetitions: int,
    base_seed: int,
) -> np.ndarray:
    """Within-arm bootstrap positions (indices into that arm's own population)."""
    rng = np.random.default_rng(stable_seed(base_seed, "arm-bootstrap", arm, str(population_size)))
    return rng.integers(
        0,
        population_size,
        size=(repetitions, population_size),
        dtype=np.int32,
    )


def evaluate_task(task: Tuple) -> Dict:
    (
        source_dataset,
        comparison,
        model,
        key,
        gt_path_string,
        dt_path_string,
        repetitions,
        base_seed,
        sample_fraction,
    ) = task
    entries, image_count, _, recall_thresholds = _prepare_category_entries(
        Path(gt_path_string), Path(dt_path_string), key
    )
    populations = _arm_populations(source_dataset, image_count, sample_fraction, base_seed)

    arms: Dict[str, Dict] = {}
    for arm, population in populations.items():
        point_ap, point_ar = _coco_ap_ar_for_sample(
            entries, population, recall_thresholds
        )
        positions = _bootstrap_positions(
            arm, len(population), repetitions, base_seed
        )
        bootstrap_ap = np.empty(repetitions, dtype=np.float64)
        bootstrap_ar = np.empty(repetitions, dtype=np.float64)
        for index, position in enumerate(positions):
            sample_indices = population[position]
            bootstrap_ap[index], bootstrap_ar[index] = _coco_ap_ar_for_sample(
                entries, sample_indices, recall_thresholds
            )
        if not np.isfinite(bootstrap_ap).all() or not np.isfinite(bootstrap_ar).all():
            raise ValueError(
                f"Non-finite bootstrap metric for {arm}/{comparison}/{model}/{key}."
            )
        arms[arm] = {
            "image_count": int(len(population)),
            "point": {"AP": point_ap, "AR": point_ar},
            "bootstrap": {"AP": bootstrap_ap, "AR": bootstrap_ar},
        }
    return {
        "comparison": comparison,
        "model": model,
        "key": key,
        "arms": arms,
    }


def build_tasks(
    args: argparse.Namespace,
    config: Mapping[str, Mapping],
) -> List[Tuple]:
    pairs = variant_pairs(config)
    tasks: List[Tuple] = []
    for model in args.models:
        for rule, variant_id in pairs:
            tasks.append(
                (
                    args.source_dataset,
                    COMPARISON_ORIGINAL_VS_TRANSFORMED,
                    model,
                    variant_id,
                    str(original_path(args.results_root, args.source_dataset, model)),
                    str(
                        transformed_path(
                            args.results_root, args.source_dataset, model, rule, variant_id
                        )
                    ),
                    args.bootstrap_repetitions,
                    args.seed,
                    args.sample_fraction,
                )
            )
            tasks.append(
                (
                    args.source_dataset,
                    COMPARISON_TRANSFORMED_VS_GT,
                    model,
                    variant_id,
                    str(gt_annotations_path(args.data_root, args.source_dataset)),
                    str(
                        transformed_gt_detection_path(
                            args.results_root, args.source_dataset, model, rule, variant_id
                        )
                    ),
                    args.bootstrap_repetitions,
                    args.seed,
                    args.sample_fraction,
                )
            )
        tasks.append(
            (
                args.source_dataset,
                COMPARISON_ORIGINAL_VS_GT,
                model,
                BASELINE_KEY,
                str(gt_annotations_path(args.data_root, args.source_dataset)),
                str(
                    original_gt_detection_path(
                        args.results_root, args.source_dataset, model
                    )
                ),
                args.bootstrap_repetitions,
                args.seed,
                args.sample_fraction,
            )
        )
        if any(rule.startswith("imgColorWheel__") for rule in config):
            for mapping in hue_metadata("imgColorWheel_0")["included_mappings"]:
                task = list(tasks[-1]) if tasks[-1][1] == COMPARISON_ORIGINAL_VS_GT else None
                if task is None:
                    raise AssertionError("Expected original-vs-GT baseline task")
                task[3] = f"original__{mapping['key']}"
                tasks.append(tuple(task))
    return tasks


def execute_tasks(
    tasks: Sequence[Tuple], workers: int
) -> Dict[Tuple[str, str, str, str], Dict]:
    """Returns raw results keyed by (comparison, arm, model, key)."""
    results: Dict[Tuple[str, str, str, str], Dict] = {}
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
                source_dataset, comparison, model, key = task[0:4]
                raise RuntimeError(
                    f"Bootstrap task failed for {source_dataset}/{comparison}/{model}/{key}"
                ) from error
            for arm, arm_result in result["arms"].items():
                lookup_key = (result["comparison"], arm, result["model"], result["key"])
                results[lookup_key] = arm_result
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


def load_trusted_points(
    analysis_results_root: Path,
    dataset: str,
    models: Sequence[str],
    config: Mapping[str, Mapping],
) -> Dict[Tuple[str, str, str, str], float]:
    """Load authoritative point AP/AR values from precomputed stats.json files.

    Keyed by (comparison, model, key, metric), where ``key`` is a
    ``variant_id`` for the per-variant comparisons and ``BASELINE_KEY`` for
    ``original_vs_gt``.
    """
    trusted: Dict[Tuple[str, str, str, str], float] = {}
    missing: List[str] = []
    pairs = variant_pairs(config)

    for model in models:
        for comparison in (COMPARISON_ORIGINAL_VS_TRANSFORMED, COMPARISON_TRANSFORMED_VS_GT):
            for _, variant_id in pairs:
                path = stats_json_path(analysis_results_root, dataset, model, variant_id, comparison)
                if not path.is_file():
                    missing.append(str(path))
                    continue
                with path.open() as handle:
                    stats = json.load(handle)
                mapping = mapping_for_key(variant_id)
                if mapping is not None:
                    stats = stats["class_shifts"][mapping["key"]]
                for metric, stats_key in POINT_METRIC_STATS_KEY.items():
                    if stats_key not in stats:
                        raise ValueError(f"{path} is missing key {stats_key!r}.")
                    trusted[(comparison, model, variant_id, metric)] = float(stats[stats_key])

        path = original_vs_gt_stats_path(analysis_results_root, dataset, model)
        if not path.is_file():
            missing.append(str(path))
        else:
            with path.open() as handle:
                stats = json.load(handle)
            for metric, stats_key in POINT_METRIC_STATS_KEY.items():
                if stats_key not in stats:
                    raise ValueError(f"{path} is missing key {stats_key!r}.")
                trusted[(COMPARISON_ORIGINAL_VS_GT, model, BASELINE_KEY, metric)] = float(
                    stats[stats_key]
                )

        if any(rule.startswith("imgColorWheel__") for rule in config):
            path = original_vs_gt_stats_path(analysis_results_root, dataset, model).with_name("color_wheel_stats.json")
            if not path.is_file():
                missing.append(str(path))
            else:
                baselines = json.loads(path.read_text())["class_shifts"]
                for key, values in baselines.items():
                    for metric, stats_key in POINT_METRIC_STATS_KEY.items():
                        trusted[(COMPARISON_ORIGINAL_VS_GT, model, f"original__{key}", metric)] = float(values[stats_key])

    if missing:
        examples = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} stats.json file(s) referenced by the trusted-point "
            f"lookup are missing, for example: {examples}"
        )
    return trusted


def validate_point_metrics(
    raw_results: Mapping[Tuple[str, str, str, str], Mapping],
    trusted_points: Mapping[Tuple[str, str, str, str], float],
    tolerance: float,
) -> Dict:
    """Only the 'full' arm is checked: it matches the stored stats.json point values."""
    differences = []
    for (comparison, arm, model, key), result in raw_results.items():
        if arm != REFERENCE_ARM:
            continue
        for metric in METRICS:
            expected_key = (comparison, model, key, metric)
            if expected_key not in trusted_points:
                continue
            difference = abs(float(result["point"][metric]) - trusted_points[expected_key])
            differences.append((difference, expected_key))
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
        "all_within_tolerance": max_difference <= tolerance,
        "interpretation": (
            "Only the 'full' arm is checked against the stored per-variant "
            "stats.json AP/AR values (it uses every declared image, like the "
            "stored evaluation). The stored stats.json AP/AR values are used "
            "directly as the reported 'full' arm point estimate in the "
            "results below; this check confirms the bootstrap's own "
            "recomputation agrees with them. Differences are expected where "
            "the stored evaluation excluded images with zero detections."
        ),
    }


def resolved_point(
    raw_results: Mapping[Tuple[str, str, str, str], Mapping],
    trusted_points: Mapping[Tuple[str, str, str, str], float],
    comparison: str,
    arm: str,
    model: str,
    key: str,
    metric: str,
) -> float:
    if arm == REFERENCE_ARM:
        trusted = trusted_points.get((comparison, model, key, metric))
        if trusted is not None:
            return trusted
    return float(raw_results[(comparison, arm, model, key)]["point"][metric])


def resolved_bootstrap(
    raw_results: Mapping[Tuple[str, str, str, str], Mapping],
    comparison: str,
    arm: str,
    model: str,
    key: str,
    metric: str,
) -> np.ndarray:
    return raw_results[(comparison, arm, model, key)]["bootstrap"][metric]


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


def statistic_record(
    raw_results: Mapping[Tuple[str, str, str, str], Mapping],
    trusted_points: Mapping[Tuple[str, str, str, str], float],
    analysis_key: str,
    arm: str,
    model: str,
    variant_id: str,
    metric: str,
) -> Dict:
    """Compute one analysis's per-arm statistic (point + bootstrap array).

    For ``ANALYSIS_CONSISTENCY`` the statistic is the raw setting-comparison
    AP/AR. For ``ANALYSIS_GT_ACCURACY`` it is baseline-comparison AP/AR minus
    setting-comparison AP/AR (both evaluated against real ground truth).
    """
    spec = ANALYSES[analysis_key]
    setting_comparison = spec["setting_comparison"]
    setting_point = resolved_point(
        raw_results, trusted_points, setting_comparison, arm, model, variant_id, metric
    )
    setting_bootstrap = resolved_bootstrap(
        raw_results, setting_comparison, arm, model, variant_id, metric
    )

    baseline_comparison = spec["baseline_comparison"]
    if baseline_comparison is None:
        return {
            "point": setting_point,
            "bootstrap": setting_bootstrap,
            "baseline_point": None,
            "setting_point": setting_point,
        }

    baseline_point = resolved_point(
        raw_results, trusted_points, baseline_comparison, arm, model, baseline_key(variant_id), metric
    )
    baseline_bootstrap = resolved_bootstrap(
        raw_results, baseline_comparison, arm, model, baseline_key(variant_id), metric
    )
    return {
        "point": baseline_point - setting_point,
        "bootstrap": baseline_bootstrap - setting_bootstrap,
        "baseline_point": baseline_point,
        "setting_point": setting_point,
    }


def build_statistic_results(
    args: argparse.Namespace,
    config: Mapping[str, Mapping],
    analysis_key: str,
    raw_results: Mapping[Tuple[str, str, str, str], Mapping],
    trusted_points: Mapping[Tuple[str, str, str, str], float],
) -> Tuple[List[Dict], Dict[Tuple[str, str, str, str], Dict]]:
    spec = ANALYSES[analysis_key]
    rows: List[Dict] = []
    arm_values: Dict[Tuple[str, str, str, str], Dict] = {}

    for rule, rule_config in config.items():
        for position, variant in enumerate(rule_config["variants"]):
            variant_id = variant["variant_id"]
            setting = {key: value for key, value in variant.items() if key != "variant_id"}
            x_value = parameter_value(rule_config, variant, position)
            for model in args.models:
                for metric in METRICS:
                    for arm in ALL_ARMS:
                        record = statistic_record(
                            raw_results, trusted_points, analysis_key, arm, model, variant_id, metric
                        )
                        low, high = percentile_interval(
                            record["bootstrap"], args.confidence_level
                        )
                        arm_values[(arm, model, variant_id, metric)] = {
                            "point": record["point"],
                            "bootstrap": record["bootstrap"],
                            "ci_low": low,
                            "ci_high": high,
                            "baseline_point": record["baseline_point"],
                            "setting_point": record["setting_point"],
                        }

                    reference = arm_values[(REFERENCE_ARM, model, variant_id, metric)]
                    for target in TARGET_ARMS:
                        target_value = arm_values[(target, model, variant_id, metric)]
                        difference_bootstrap = (
                            reference["bootstrap"] - target_value["bootstrap"]
                        )
                        low, high = percentile_interval(
                            difference_bootstrap, args.confidence_level
                        )
                        difference = reference["point"] - target_value["point"]
                        rows.append(
                            {
                                "analysis": analysis_key,
                                "setting_comparison": spec["setting_comparison"],
                                "baseline_comparison": spec["baseline_comparison"],
                                "reference_arm": REFERENCE_ARM,
                                "target_arm": target,
                                "model": model,
                                "rule": rule,
                                "variant_id": variant_id,
                                "varying_parameter": rule_config["varying_parameter"],
                                "parameter_value": x_value,
                                "setting": setting,
                                "metric": metric,
                                "reference_baseline_metric": reference["baseline_point"],
                                "target_baseline_metric": target_value["baseline_point"],
                                "reference_setting_metric": reference["setting_point"],
                                "target_setting_metric": target_value["setting_point"],
                                "reference_statistic": reference["point"],
                                "reference_statistic_ci_low": reference["ci_low"],
                                "reference_statistic_ci_high": reference["ci_high"],
                                "target_statistic": target_value["point"],
                                "target_statistic_ci_low": target_value["ci_low"],
                                "target_statistic_ci_high": target_value["ci_high"],
                                "statistic_difference": difference,
                                "difference_ci_low": low,
                                "difference_ci_high": high,
                                "equivalence_margin": args.equivalence_margin,
                                "classification": classify_interval(
                                    low, high, args.equivalence_margin
                                ),
                            }
                        )
    return rows, arm_values


def build_auc_results(
    args: argparse.Namespace,
    config: Mapping[str, Mapping],
    analysis_key: str,
    arm_values: Mapping[Tuple[str, str, str, str], Mapping],
) -> Tuple[List[Dict], Dict[Tuple[str, str, str, str], Dict]]:
    spec = ANALYSES[analysis_key]
    rows: List[Dict] = []
    arm_auc: Dict[Tuple[str, str, str, str], Dict] = {}

    for rule, rule_config in config.items():
        variants = rule_config["variants"]
        if len(variants) < 2 or rule.startswith("imgColorWheel__"):
            # Do not integrate across manually excluded hue intervals.
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
                for arm in ALL_ARMS:
                    values = [
                        arm_values[
                            (arm, model, variant["variant_id"], metric)
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
                    arm_auc[(arm, model, rule, metric)] = {
                        "point": point_auc,
                        "bootstrap": bootstrap_auc,
                        "ci_low": low,
                        "ci_high": high,
                    }

                reference = arm_auc[(REFERENCE_ARM, model, rule, metric)]
                for target in TARGET_ARMS:
                    target_value = arm_auc[(target, model, rule, metric)]
                    difference_bootstrap = (
                        reference["bootstrap"] - target_value["bootstrap"]
                    )
                    low, high = percentile_interval(
                        difference_bootstrap, args.confidence_level
                    )
                    rows.append(
                        {
                            "analysis": analysis_key,
                            "setting_comparison": spec["setting_comparison"],
                            "baseline_comparison": spec["baseline_comparison"],
                            "reference_arm": REFERENCE_ARM,
                            "target_arm": target,
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
    return rows, arm_auc


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


def _grouped(rows: Sequence[Mapping], fields: Sequence[str]) -> List[Dict]:
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


def build_summaries(parameter_rows: Sequence[Mapping], auc_rows: Sequence[Mapping]) -> Dict:
    summaries: Dict[str, Dict] = {}
    for analysis_key in ANALYSES:
        p_rows = [row for row in parameter_rows if row["analysis"] == analysis_key]
        a_rows = [row for row in auc_rows if row["analysis"] == analysis_key]
        summaries[analysis_key] = {
            "parameter_overall": count_classifications(p_rows),
            "parameter_by_target_metric": _grouped(p_rows, ("target_arm", "metric")),
            "parameter_by_target_metric_model": _grouped(
                p_rows, ("target_arm", "metric", "model")
            ),
            "parameter_by_target_metric_rule": _grouped(
                p_rows, ("target_arm", "metric", "rule")
            ),
            "auc_overall": count_classifications(a_rows),
            "auc_by_target_metric": _grouped(a_rows, ("target_arm", "metric")),
            "auc_by_target_metric_model": _grouped(
                a_rows, ("target_arm", "metric", "model")
            ),
            "auc_by_target_metric_rule": _grouped(
                a_rows, ("target_arm", "metric", "rule")
            ),
        }
    return summaries


def format_rule(rule: str) -> str:
    if rule.startswith("imgColorWheel__"):
        return "Hue rotation: " + rule.split("__", 1)[1].replace("_to_", " → ").replace("_", " ")
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
    analysis_arm_values: Mapping[str, Mapping[Tuple[str, str, str, str], Mapping]],
) -> List[str]:
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[str] = []

    for analysis_key, arm_values in analysis_arm_values.items():
        spec = ANALYSES[analysis_key]
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
                        positions = np.arange(len(ALL_ARMS))
                        points = []
                        lows = []
                        highs = []
                        for arm in ALL_ARMS:
                            value = arm_values[
                                (arm, model, variants[0]["variant_id"], metric)
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
                            positions, [ARM_LABELS[arm] for arm in ALL_ARMS], rotation=15
                        )
                    else:
                        for arm in ALL_ARMS:
                            values = [
                                arm_values[
                                    (arm, model, variant["variant_id"], metric)
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
                                linestyle="none" if rule.startswith("imgColorWheel__") else "-",
                                markersize=4,
                                label=ARM_LABELS[arm],
                                color=ARM_COLORS[arm],
                            )
                            if rule.startswith("imgColorWheel__"):
                                axis.vlines(x_values, low, high, color=ARM_COLORS[arm], alpha=0.6)
                            else:
                                axis.fill_between(x_values, low, high, alpha=0.14, color=ARM_COLORS[arm])
                        axis.set_xlabel(format_parameter(rule_config["varying_parameter"]))
                    if len(variants) == 1 and rule.startswith("imgColorWheel__"):
                        axis.set_xlabel(f"Hue rotation: {x_values[0]:g}")
                    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
                    axis.set_ylabel(spec["ylabel_template"].format(metric=metric))
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
                    f"{format_rule(rule)}: {metric} \u2014 {spec['title']}",
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
                filename = f"{rule}_{metric.lower()}_{spec['plot_suffix']}.png"
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


def _render_analysis_section(
    args: argparse.Namespace,
    config: Mapping[str, Mapping],
    analysis_key: str,
    parameter_rows: Sequence[Mapping],
    auc_rows: Sequence[Mapping],
    summaries: Mapping,
    plot_paths: Sequence[str],
) -> List[str]:
    spec = ANALYSES[analysis_key]
    sample_percent = int(round(args.sample_fraction * 100))
    confidence_percent = int(round(args.confidence_level * 100))
    lines: List[str] = [f"## {spec['title']}", ""]
    lines.append(f"Statistic: {spec['statistic_description']}.")
    lines.append("")

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

    if parameter_total:
        equivalent_share = parameter_counts[CLASS_EQUIVALENT] / parameter_total
        key_finding_line = (
            f"- At parameter level, {parameter_counts[CLASS_EQUIVALENT]:,} of "
            f"{parameter_total:,} comparisons ({equivalent_share:.1%}) "
            f"were equivalent, {parameter_counts[CLASS_INCONCLUSIVE]:,} were "
            f"inconclusive, and {parameter_counts[CLASS_DIFFERENT]:,} were "
            "meaningfully different."
        )
    else:
        key_finding_line = "- No parameter-level comparisons were computed."
    lines.extend(["### Key findings", "", key_finding_line])
    if auc_total:
        lines.append(
            f"- At rule-curve level, {auc_counts[CLASS_EQUIVALENT]} of "
            f"{auc_total} AUC comparisons were equivalent; "
            f"{auc_counts[CLASS_INCONCLUSIVE]} were inconclusive and "
            f"{auc_counts[CLASS_DIFFERENT]} were meaningfully different."
        )
    if fully_equivalent_auc_models:
        lines.append(
            "- Every AUC comparison was equivalent for "
            f"{', '.join(fully_equivalent_auc_models)}. Their overall parameter-response "
            f"patterns therefore transfer from the full test set to the {sample_percent}% "
            f"subsample within the +/-{args.equivalence_margin:.2f} margin."
        )
    if non_equivalent_auc_models:
        lines.append(
            "- All non-equivalent AUC outcomes were confined to "
            + ", ".join(
                MODEL_LABELS.get(model, model) for model in non_equivalent_auc_models
            )
            + "."
        )

    lines.extend(["", "### Parameter-level overview", ""])
    overview_rows = []
    for summary in summaries["parameter_by_target_metric"]:
        counts = summary["counts"]
        overview_rows.append(
            [
                ARM_LABELS[summary["target_arm"]],
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

    lines.extend(["", "### Rule-level AUC results", ""])
    lines.append(
        "AUC uses the complete configured parameter range with its axis scaled to 0-1. "
        "Rules with a single setting (Grayscale) and reviewed hue mappings with excluded intervals have no AUC result."
    )
    lines.append("")
    for target in TARGET_ARMS:
        for metric in METRICS:
            lines.extend(
                [
                    f"#### {metric}: full test set vs {sample_percent}% subsample",
                    "",
                ]
            )
            table_rows = []
            for row in auc_rows:
                if row["target_arm"] != target or row["metric"] != metric:
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
                    ["Model", "Rule", "Full AUC", "Sample AUC", "Difference", f"{confidence_percent}% CI", "Result"],
                    table_rows,
                )
            )
            lines.append("")

    lines.extend(["### Sample-size equivalence summary", ""])
    model_auc = {
        (item["target_arm"], item["metric"], item["model"]): item
        for item in summaries["auc_by_target_metric_model"]
    }
    transfer_rows = []
    for model in args.models:
        cells = []
        total_equivalent = 0
        total = 0
        for target in TARGET_ARMS:
            for metric in METRICS:
                summary = model_auc.get((target, metric, model))
                if summary is None:
                    cells.append("n/a")
                    continue
                equivalent = summary["counts"][CLASS_EQUIVALENT]
                cells.append(f"{equivalent}/{summary['total']}")
                total_equivalent += equivalent
                total += summary["total"]
        transfer_rows.append(
            [MODEL_LABELS.get(model, model), *cells, f"{total_equivalent}/{total}" if total else "n/a"]
        )
    lines.extend(
        markdown_table(
            ["Model", "Sample AP", "Sample AR", "All equivalent AUCs"],
            transfer_rows,
        )
    )
    lines.extend(
        [
            "",
            "Counts show equivalent rule-level AUC comparisons out of the rules with numeric parameter curves.",
            "",
            "### Largest parameter-level differences",
            "",
        ]
    )
    largest = sorted(
        parameter_rows,
        key=lambda row: abs(float(row["statistic_difference"])),
        reverse=True,
    )[:25]
    lines.extend(
        markdown_table(
            ["Target", "Metric", "Model", "Rule", "Setting", "Difference", f"{confidence_percent}% CI", "Result"],
            [
                [
                    ARM_LABELS[row["target_arm"]],
                    row["metric"],
                    MODEL_LABELS.get(row["model"], row["model"]),
                    format_rule(row["rule"]),
                    row["variant_id"],
                    f"{row['statistic_difference']:+.3f}",
                    f"[{row['difference_ci_low']:+.3f}, {row['difference_ci_high']:+.3f}]",
                    class_short(row["classification"]),
                ]
                for row in largest
            ],
        )
    )

    relevant_plots = [
        path for path in plot_paths if path.endswith(f"_{spec['plot_suffix']}.png")
    ]
    if relevant_plots:
        lines.extend(["", "### Response curves", ""])
        for rule in config:
            rule_plots = [
                f"plots/{rule}_{metric.lower()}_{spec['plot_suffix']}.png" for metric in METRICS
            ]
            rule_plots = [path for path in rule_plots if path in relevant_plots]
            if not rule_plots:
                continue
            lines.extend([f"#### {format_rule(rule)}", ""])
            for metric, relative in zip(METRICS, rule_plots):
                lines.append(f"![{format_rule(rule)} {metric} {spec['plot_suffix']}]({relative})")
                lines.append("")

    lines.append("")
    return lines


def write_markdown_report(
    args: argparse.Namespace,
    config: Mapping[str, Mapping],
    parameter_rows: Sequence[Mapping],
    auc_rows: Sequence[Mapping],
    summaries: Mapping,
    validation: Mapping,
    image_counts: Mapping[str, int],
    plot_paths: Sequence[str],
) -> Path:
    report_path = args.output_dir / "report.md"
    confidence_percent = int(round(args.confidence_level * 100))
    sample_percent = int(round(args.sample_fraction * 100))
    lines = [
        "# Test-set sample-size metamorphic consistency equivalence",
        "",
        "## Research question",
        "",
        (
            f"If only {sample_percent}% of the test dataset were used for "
            "metamorphic assessment, would (a) the observed original-vs-transformed "
            "consistency and (b) the observed ground-truth accuracy degradation "
            "caused by each transformation be equivalent to using the full test dataset?"
        ),
        "",
        "## Design",
        "",
        (
            f"The analysis covers {len(args.models)} models, {len(config)} rules, "
            f"and {sum(len(value['variants']) for value in config.values())} settings, "
            f"drawn entirely from the `{args.source_dataset}` dataset "
            f"({image_counts[REFERENCE_ARM]} images)."
        ),
        "",
        (
            f"The `{TARGET_ARM}` arm uses a single fixed, reproducibly-drawn "
            f"subsample containing {image_counts[TARGET_ARM]} images "
            f"({sample_percent}% of {image_counts[REFERENCE_ARM]}, rounded to the "
            "nearest image). The same subsample is reused across every model, "
            "rule, setting, and analysis below."
        ),
        "",
        "Two independent statistics are tested for equivalence between the two arms:",
        "",
    ]
    for analysis_key, spec in ANALYSES.items():
        lines.append(f"- **{spec['title']}** ({analysis_key}): {spec['statistic_description']}.")
    lines.extend(
        [
            "",
            (
                f"Each arm is independently resampled {args.bootstrap_repetitions:,} "
                "times with replacement from within its own population: the `full` arm "
                f"resamples from all {image_counts[REFERENCE_ARM]} images, while the "
                f"`{TARGET_ARM}` arm resamples only from its "
                f"{image_counts[TARGET_ARM]}-image subsample. Results use percentile "
                f"{confidence_percent}% confidence intervals and a symmetric "
                f"+/-{args.equivalence_margin:.2f} equivalence margin. No p-values or "
                "multiple-testing corrections are used."
            ),
            "",
            (
                "Point AP/AR values reported for the `full` arm are read directly from "
                "the existing `analysis/results/rq1/<dataset>/<model>/<variant_id>/<comparison>/"
                "stats.json` files (and the per-model `original_vs_gt/stats.json`), not "
                "recomputed; only the `sample` arm's point and both arms' bootstrap "
                "confidence intervals are computed by this script (see Validation)."
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
                f"A positive difference means the full-dataset statistic is greater than "
                f"the {sample_percent}% subsample; a negative difference means the "
                "subsample statistic is greater."
            ),
            "",
            "## Validation",
            "",
            (
                f"The implementation recomputed {validation['values_checked']:,} point "
                "AP/AR values for the `full` arm over every declared image and compared "
                "them with the stored per-variant `stats.json` files. "
                f"{validation['values_within_tolerance']:,} values agreed within "
                f"{validation['tolerance']:.1g}; "
                f"{validation['values_different_from_stored_json']:,} differed."
                if validation['values_within_tolerance'] is not None
                and validation['values_different_from_stored_json'] is not None
                else (
                    f"The retained evidence covers {validation['values_checked']:,} "
                    "full-arm point AP/AR checks. Aggregate agreement and disagreement "
                    "counts are unavailable: per-model recomputation discrepancies "
                    "were not saved. No evaluation or bootstrap was rerun."
                )
            ),
            "",
        ]
    )
    if validation["maximum_difference_key"]:
        lines.extend(
            [
                (
                    f"The maximum stored-versus-recomputed discrepancy was "
                    f"{validation['maximum_absolute_difference']:.3g} at "
                    f"`{'/'.join(validation['maximum_difference_key'])}`. Reported `full` "
                    "arm points use the stored stats.json values, not this recomputation."
                ),
                "",
            ]
        )

    for analysis_key in ANALYSES:
        p_rows = [row for row in parameter_rows if row["analysis"] == analysis_key]
        a_rows = [row for row in auc_rows if row["analysis"] == analysis_key]
        lines.extend(
            _render_analysis_section(
                args, config, analysis_key, p_rows, a_rows, summaries[analysis_key], plot_paths
            )
        )

    lines.extend(
        [
            "## Interpretation cautions",
            "",
            "- Original-vs-transformed consistency concerns metamorphic consistency, not absolute ground-truth detection accuracy.",
            "- Ground-truth accuracy degradation concerns real detection accuracy loss and is directly comparable across rules on the same AP/AR scale.",
            (
                f"- Unlike a comparison between independent datasets, the `{TARGET_ARM}` "
                f"arm's images are a strict subset of the `{REFERENCE_ARM}` arm's images, "
                "so the two bootstrap distributions are correlated rather than independent. "
                "The comparison estimates the added sampling variability/bias from using "
                "fewer images, not generalisation across distinct datasets."
            ),
            "- Reviewed hue mappings use separate source-class baselines and results. Excluded settings are omitted, and hue AUC is not computed across excluded intervals.",
            "- Images are the resampling units. No sequence/rosbag metadata was available to model temporal clustering.",
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
    source_image_count = validate_inputs(
        args.results_root, args.data_root, args.source_dataset, args.models, config
    )

    print("Sample-size equivalence bootstrap analysis", flush=True)
    print(f"Source dataset: {args.source_dataset} ({source_image_count} images)", flush=True)
    print(f"Sample fraction: {args.sample_fraction:.0%}", flush=True)
    print(f"Models: {', '.join(args.models)}", flush=True)
    print(f"Rules: {', '.join(config)}", flush=True)
    print(f"Analyses: {', '.join(ANALYSES)}", flush=True)
    print(
        f"Bootstrap: {args.bootstrap_repetitions:,} repetitions, "
        f"{args.confidence_level:.0%} intervals, +/-{args.equivalence_margin:.2f} margin",
        flush=True,
    )
    print(f"Workers: {args.workers}", flush=True)

    tasks = build_tasks(args, config)
    print(f"COCO model/variant/comparison evaluations: {len(tasks)}", flush=True)
    raw_results = execute_tasks(tasks, args.workers)

    sample_image_count = int(
        len(_sample_population_indices(
            args.source_dataset, source_image_count, args.sample_fraction, args.seed
        ))
    )
    image_counts = {REFERENCE_ARM: source_image_count, TARGET_ARM: sample_image_count}

    trusted_points = load_trusted_points(
        args.analysis_results_root, args.source_dataset, args.models, config
    )
    validation = validate_point_metrics(raw_results, trusted_points, args.point_tolerance)
    print(
        "stats.json audit (full arm only): "
        f"{validation['values_within_tolerance']}/{validation['values_checked']} "
        "values agree with all-image recomputation; maximum discrepancy "
        f"{validation['maximum_absolute_difference']:.3g}",
        flush=True,
    )

    parameter_rows: List[Dict] = []
    auc_rows: List[Dict] = []
    analysis_arm_values: Dict[str, Dict] = {}
    for analysis_key in ANALYSES:
        rows, arm_values = build_statistic_results(
            args, config, analysis_key, raw_results, trusted_points
        )
        parameter_rows.extend(rows)
        analysis_arm_values[analysis_key] = arm_values
        auc_analysis_rows, _ = build_auc_results(args, config, analysis_key, arm_values)
        auc_rows.extend(auc_analysis_rows)

    summaries = build_summaries(parameter_rows, auc_rows)

    plot_paths: List[str] = []
    if not args.skip_plots:
        plot_paths = create_response_plots(args, config, analysis_arm_values)

    output = {
        "metadata": {
            "source_dataset": args.source_dataset,
            "reference_arm": REFERENCE_ARM,
            "target_arms": list(TARGET_ARMS),
            "sample_fraction": args.sample_fraction,
            "arms": {
                arm: {"image_count": count}
                for arm, count in image_counts.items()
            },
            "models": list(args.models),
            "rules": list(config),
            "color_wheel_reviewed_settings": [hue_metadata(f"imgColorWheel_{i}") for i in range(10)] if "imgColorWheel" in args.rules else [],
            "metrics": list(METRICS),
            "analyses": {
                analysis_key: {
                    "title": spec["title"],
                    "setting_comparison": spec["setting_comparison"],
                    "baseline_comparison": spec["baseline_comparison"],
                    "statistic_description": spec["statistic_description"],
                }
                for analysis_key, spec in ANALYSES.items()
            },
            "point_source": {
                "full_arm": "stats_json",
                "sample_arm": "recomputed_bootstrap_point",
                "stats_json_root": str(args.analysis_results_root),
            },
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
                "color_wheel": "not_applicable_excluded_intervals_and_separate_source_mappings",
            },
            "workers": args.workers,
            "source_config": str(args.config),
            "source_analysis_results_root": str(args.analysis_results_root),
            "source_detection_root": str(args.results_root),
            "source_data_root": str(args.data_root),
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
        image_counts,
        plot_paths,
    )
    print(f"Saved JSON: {results_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)
    print(f"Saved plots: {len(plot_paths)}", flush=True)


if __name__ == "__main__":
    main()
