#!/usr/bin/env python3
"""Validate fixed RQ2b brightness ranges on gamma-transformed holdout images.

Only images whose existing original detections have per-image AP >= 0.5 are
eligible. For each eligible image, the script searches for bright and dark
gamma failures and retains points spread across the AP interval below 0.5.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq2b_validation_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.rq2.rq2a_brute_force_search import (  # noqa: E402
    GAMMA_RANGE, GAMMA_STEP, apply_gamma_transformation,
    load_yolo_model, run_yolo_inference,
)
from analysis.coco_eval import evaluate_coco  # noqa: E402
from analysis.utils.format_detections import format_detections  # noqa: E402
from utils.imgDarkBright_tuning.brightness_methods import (  # noqa: E402
    BRIGHTNESS_METHODS, calculate_all_brightness_scores,
)
from utils.model_registry import get_model_names  # noqa: E402

HOLDOUT = ROOT / "data/base_images/comp_labelled"
DETECTIONS = ROOT / "results/competition_labelled"
OUTPUT = ROOT / "analysis/results/rq2/rq2b/validation"


def rq2b_default(model: str) -> Path:
    results_root = ROOT / "analysis/results/rq2/rq2b/brightness_scoring/brute_force"
    exact = results_root / model / "results.json"
    if exact.is_file():
        return exact

    for candidate in sorted(results_root.glob("*/results.json")):
        try:
            models = json.loads(candidate.read_text()).get("models", [])
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(models, str):
            models = [models]
        if model in models:
            return candidate
    return exact


def derive_ranges(path: Path, model: str) -> Dict[str, Dict]:
    """Use the unadjusted dark and bright RQ2b boundary means."""
    data = json.loads(path.read_text())
    models = data.get("models", [])
    models = [models] if isinstance(models, str) else models
    if model not in models:
        raise ValueError(f"{path} does not contain {model}")
    records = [r for r in data.get("per_image_results", []) if r.get("model", model) == model]
    result = {}
    for method in BRIGHTNESS_METHODS:
        bright = [float(r["scores_left"][method]) for r in records if r.get("scores_left")]
        dark = [float(r["scores_right"][method]) for r in records if r.get("scores_right")]
        if len(bright) < 2 or len(dark) < 2:
            raise ValueError(f"Too few boundary scores for {method}")
        bm, bs = float(np.mean(bright)), float(np.std(bright, ddof=1))
        dm, ds = float(np.mean(dark)), float(np.std(dark, ddof=1))
        lower, upper = dm, bm
        result[method] = {
            "lower": lower, "upper": upper, "sigma_multiplier": 0.0,
            "valid_interval": lower < upper,
            "invalid_reason": (None if lower < upper else
                               "dark boundary mean is not below the bright boundary mean"),
            "dark_boundary": {"rq2b_side": "right", "n": len(dark), "mean": dm,
                              "sample_standard_deviation": ds},
            "bright_boundary": {"rq2b_side": "left", "n": len(bright), "mean": bm,
                                "sample_standard_deviation": bs},
        }
    return result


def images_in_coco_order(root: Path) -> List[Path]:
    data = json.loads((root / "annotations_coco.json").read_text())
    paths = [root / "images" / item["file_name"] for item in data.get("images", [])]
    if not paths:
        raise ValueError("No holdout images declared")
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing holdout image: {missing[0]}")
    return paths


def load_per_image_ground_truth(annotations_path: Path) -> Dict[str, Dict]:
    """Create one-image COCO subsets without changing category IDs."""
    data = json.loads(annotations_path.read_text())
    by_image_id: Dict[int, List[Dict]] = {}
    for annotation in data.get("annotations", []):
        by_image_id.setdefault(annotation["image_id"], []).append(annotation)
    subsets = {}
    for image in data.get("images", []):
        subset = {
            "images": [image],
            "annotations": by_image_id.get(image["id"], []),
            "categories": data.get("categories", []),
        }
        subsets[image["file_name"]] = subset
        subsets[Path(image["file_name"]).stem] = subset
    return subsets


def quiet_ap(detections: Path, ground_truth: Dict) -> float:
    """Evaluate one image using its canonical COCO IDs and categories."""
    with tempfile.TemporaryDirectory(prefix="rq2b_coco_") as temporary:
        temporary_root = Path(temporary)
        gt_path = temporary_root / "ground_truth.json"
        detection_path = temporary_root / "detections.json"
        gt_path.write_text(json.dumps(ground_truth))
        with contextlib.redirect_stdout(io.StringIO()):
            format_detections(
                str(detections),
                str(detection_path),
                gt_coco_file=str(gt_path),
                as_ground_truth=False,
            )
            stats = evaluate_coco(str(gt_path), str(detection_path), iou_type="bbox")
        return float(stats[0]) if stats is not None else 0.0


def score_image(image: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in calculate_all_brightness_scores(image).items()}


def baseline(path: Path, ground_truth: Dict, detections: Path, threshold: float) -> Dict:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read {path}")
    detection = detections / f"{path.stem}.txt"
    if not detection.is_file():
        raise FileNotFoundError(f"Missing original detection: {detection}")
    ap = quiet_ap(detection, ground_truth)
    return {"image_name": path.name, "kind": "original", "side": "original",
            "gamma": 1.0, "ap": ap, "actual_failure": 0 <= ap < threshold,
            "brightness_scores": None}


def add_baseline_scores(path: Path, observation: Dict) -> None:
    """Brightness scoring is expensive; do it only for eligible originals."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read {path}")
    observation["brightness_scores"] = score_image(image)


def side_grid(side: str, lo: float, hi: float, step: float) -> List[float]:
    if side == "bright":
        values, extreme = np.arange(1-step, lo-step/2, -step), lo
    else:
        values, extreme = np.arange(1+step, hi+step/2, step), hi
    values = [float(round(np.clip(x, lo, hi), 10)) for x in values]
    if not values or not np.isclose(values[-1], extreme):
        values.append(float(extreme))
    return list(dict.fromkeys(values))


def failure_ap_targets(threshold: float, points: int) -> List[float]:
    """Return the centres of equal-width bins over ``[0, threshold)``.

    For example, five points and a 0.5 threshold produce targets of 0.45,
    0.35, 0.25, 0.15, and 0.05. Avoiding the interval endpoints prevents the
    selected set from being dominated by just-below-threshold and zero-AP
    observations.
    """
    width = threshold / points
    return [float(round(threshold - (index + 0.5) * width, 10))
            for index in range(points)]


def refine_ap_targets(
    evaluations: List[Dict], targets: Sequence[float],
    ap_at: Callable[[float], float], max_iterations: int = 4,
    tolerance: float = 0.02,
) -> None:
    """Add gamma evaluations close to target APs using bracketed interpolation.

    The coarse gamma grid establishes a local bracket. Interpolation improves
    AP spacing without evaluating the entire gamma range at a very fine step.
    Results are appended in place and later participate in target assignment.
    """
    coarse = list(evaluations)
    known_gammas = {item["gamma"] for item in evaluations}
    for target in targets:
        bracket = next((
            (left, right) for left, right in zip(coarse, coarse[1:])
            if (left["ap"] - target) * (right["ap"] - target) <= 0
        ), None)
        if bracket is None:
            continue
        left, right = ({**bracket[0]}, {**bracket[1]})
        if min(abs(left["ap"] - target), abs(right["ap"] - target)) <= tolerance:
            continue
        for _ in range(max_iterations):
            denominator = right["ap"] - left["ap"]
            if abs(denominator) < 1e-12:
                gamma = (left["gamma"] + right["gamma"]) / 2
            else:
                gamma = left["gamma"] + (
                    (right["gamma"] - left["gamma"])
                    * (target - left["ap"]) / denominator
                )
            low, high = sorted((left["gamma"], right["gamma"]))
            gamma = float(round(np.clip(gamma, low, high), 10))
            if gamma in (left["gamma"], right["gamma"]) or gamma in known_gammas:
                gamma = float(round((left["gamma"] + right["gamma"]) / 2, 10))
            if gamma in known_gammas:
                break
            item = {"gamma": gamma, "ap": ap_at(gamma),
                    "refined_for_target_ap": float(target)}
            evaluations.append(item)
            known_gammas.add(gamma)
            if abs(item["ap"] - target) <= tolerance:
                break
            if (left["ap"] - target) * (item["ap"] - target) <= 0:
                right = item
            else:
                left = item


def select_ap_spaced_failures(
    evaluations: Sequence[Dict], targets: Sequence[float], threshold: float,
) -> List[Dict]:
    """Assign a distinct evaluated gamma to each target AP where possible."""
    available = [item for item in evaluations if 0 <= item["ap"] < threshold]
    selected = []
    for target in targets:
        if not available:
            break
        chosen = min(
            available,
            key=lambda item: (abs(item["ap"] - target), abs(item["gamma"] - 1.0)),
        )
        selected.append({**chosen, "target_ap": float(target)})
        available.remove(chosen)
    return selected


def validate_image(path: Path, ground_truth: Dict, model, device: str, threshold: float,
                   lo: float, hi: float, step: float, points: int) -> Tuple[List[Dict], Dict]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read {path}")
    cache: Dict[float, float] = {}
    with tempfile.TemporaryDirectory(prefix="rq2b_holdout_") as tmp:
        tmp_root = Path(tmp)

        def ap_at(gamma: float) -> float:
            gamma = float(round(gamma, 10))
            if gamma in cache:
                return cache[gamma]
            work = tmp_root / f"gamma_{gamma:.10f}"
            transformed = apply_gamma_transformation(image, gamma, path.name, str(work))
            if not transformed:
                raise RuntimeError(f"Gamma transform failed: {path.name}, {gamma}")
            det_dir = work / "detections"
            run_yolo_inference(model, transformed, str(det_dir), device, original_image_name=path.name)
            cache[gamma] = quiet_ap(det_dir, ground_truth)
            return cache[gamma]

        observations, summaries = [], {}
        targets = failure_ap_targets(threshold, points)
        for side in ("bright", "dark"):
            searched, first = [], None
            grid = side_grid(side, lo, hi, step)
            for gamma in grid:
                ap = ap_at(gamma)
                searched.append({"gamma": gamma, "ap": ap})
                if first is None and 0 <= ap < threshold:
                    first = gamma
                failures = [item for item in searched if 0 <= item["ap"] < threshold]
                if (first is not None and len(failures) >= points
                        and ap <= min(targets)):
                    break
            refine_ap_targets(searched, targets, ap_at)
            selected = select_ap_spaced_failures(searched, targets, threshold)
            for item in selected:
                gamma, ap = item["gamma"], item["ap"]
                work = tmp_root / f"score_{side}_{gamma:.10f}"
                transformed_path = apply_gamma_transformation(image, gamma, path.name, str(work))
                transformed = cv2.imread(str(transformed_path)) if transformed_path else None
                if transformed is None:
                    raise RuntimeError(f"Could not score gamma image at {gamma}")
                observations.append({
                    "image_name": path.name, "kind": "transformed_failure",
                    "side": side, "gamma": gamma, "ap": ap,
                    "target_ap": item["target_ap"], "actual_failure": True,
                    "brightness_scores": score_image(transformed),
                })
            summaries[side] = {"first_failing_grid_gamma": first,
                               "target_ap_values": targets,
                               "search_evaluations": searched,
                               "selected_failure_count": len(selected),
                               "requested_failure_count": points}
    return observations, {"image_name": path.name, "sides": summaries,
                          "unique_gamma_inferences": len(cache)}


def div(a: int, b: int) -> Optional[float]:
    return a / b if b else None


def classify(observations: Sequence[Dict], ranges: Dict[str, Dict]) -> Dict[str, Dict]:
    output = {}
    for method, bounds in ranges.items():
        if not bounds["valid_interval"]:
            output[method] = {
                "n": len(observations), "tp": None, "fp": None, "tn": None, "fn": None,
                "failure_recall": None, "original_acceptance_rate": None,
                "balanced_accuracy": None, "matthews_correlation_coefficient": None,
                "bright_failure_recall": None, "dark_failure_recall": None,
                "unavailable_reason": bounds["invalid_reason"],
            }
            continue
        tp = fp = tn = fn = 0
        sides = {"bright": [0, 0], "dark": [0, 0]}
        for obs in observations:
            value = obs["brightness_scores"][method]
            outside = value < bounds["lower"] or value > bounds["upper"]
            obs.setdefault("method_predictions", {})[method] = {
                "value": value, "in_operating_range": not outside,
                "predicted_failure": outside}
            failure = obs["actual_failure"]
            if outside and failure: tp += 1
            elif outside: fp += 1
            elif failure: fn += 1
            else: tn += 1
            if failure and obs["side"] in sides:
                sides[obs["side"]][0] += int(outside)
                sides[obs["side"]][1] += 1
        recall, specificity = div(tp, tp+fn), div(tn, tn+fp)
        denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        output[method] = {
            "n": tp+fp+tn+fn, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "failure_recall": recall, "original_acceptance_rate": specificity,
            "balanced_accuracy": ((recall+specificity)/2 if recall is not None and specificity is not None else None),
            "matthews_correlation_coefficient": ((tp*tn-fp*fn)/denominator if denominator else None),
            "bright_failure_recall": div(*sides["bright"]),
            "dark_failure_recall": div(*sides["dark"]),
        }
    return output


def _representative_samples(items: Sequence[Dict], method: str, count: int) -> List[Dict]:
    """Choose deterministic samples spanning a category's score range."""
    ordered = sorted(items, key=lambda item: (
        item["brightness_scores"][method], item["image_name"], item["gamma"],
    ))
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[index] for index in dict.fromkeys(indices)]


def export_sample_images(
    output_root: Path, observations: Sequence[Dict], ranges: Dict[str, Dict],
    image_paths: Dict[str, Path], samples_per_category: int,
) -> Dict[str, Dict]:
    """Save representative too-dark, in-range, and out-of-range examples.

    ``out_of_range`` contains failures on either side of the interval, so it
    intentionally overlaps ``too_dark``. The manifest records whether each
    out-of-range example is below or above the fixed interval.
    """
    summary = {}
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rq2b_samples_") as temporary:
        transform_root = Path(temporary)
        for method, bounds in ranges.items():
            method_dir = output_root / method
            method_dir.mkdir(parents=True, exist_ok=True)
            for stale in method_dir.glob("*.jpg"):
                stale.unlink()

            if not bounds["valid_interval"]:
                manifest = {
                    "method": method, "available": False,
                    "reason": bounds["invalid_reason"], "samples": [],
                }
                (method_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
                summary[method] = {"available": False, "sample_count": 0}
                continue

            lower, upper = bounds["lower"], bounds["upper"]
            pools = {
                "too_dark": [o for o in observations
                             if o["brightness_scores"][method] < lower],
                "in_range": [o for o in observations
                             if lower <= o["brightness_scores"][method] <= upper],
                "out_of_range": [o for o in observations
                                 if not lower <= o["brightness_scores"][method] <= upper],
            }
            entries = []
            for category, pool in pools.items():
                for observation in _representative_samples(
                        pool, method, samples_per_category):
                    source_path = image_paths[observation["image_name"]]
                    source_image = cv2.imread(str(source_path))
                    if source_image is None:
                        raise ValueError(f"Cannot read {source_path}")
                    if observation["kind"] == "original":
                        sample_image = source_image
                    else:
                        transformed_path = apply_gamma_transformation(
                            source_image, observation["gamma"], source_path.name,
                            str(transform_root / method / category),
                        )
                        sample_image = (cv2.imread(str(transformed_path))
                                        if transformed_path else None)
                        if sample_image is None:
                            raise RuntimeError(
                                f"Could not recreate sample {source_path.name} "
                                f"at gamma {observation['gamma']}"
                            )
                    stem = source_path.stem.replace(" ", "_")
                    filename = (
                        f"{category}__{stem}__{observation['side']}__"
                        f"gamma-{observation['gamma']:.4f}__ap-{observation['ap']:.3f}.jpg"
                    )
                    if not cv2.imwrite(str(method_dir / filename), sample_image):
                        raise RuntimeError(f"Could not write {method_dir / filename}")
                    score = observation["brightness_scores"][method]
                    entries.append({
                        "file": filename, "category": category,
                        "range_side": ("below" if score < lower else
                                       "above" if score > upper else "inside"),
                        "source_image": observation["image_name"],
                        "kind": observation["kind"], "side": observation["side"],
                        "gamma": observation["gamma"], "ap": observation["ap"],
                        "brightness_score": score,
                    })
            manifest = {
                "method": method, "available": True, "lower": lower,
                "upper": upper, "samples_per_category": samples_per_category,
                "category_pool_sizes": {key: len(value) for key, value in pools.items()},
                "samples": entries,
            }
            (method_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            summary[method] = {"available": True, "sample_count": len(entries)}
    return summary


def rank(metrics: Dict[str, Dict]) -> List[str]:
    def key(method: str):
        m = metrics[method]
        return tuple(-math.inf if m[x] is None else m[x] for x in
                     ("matthews_correlation_coefficient", "balanced_accuracy", "failure_recall"))
    valid = [m for m in metrics if metrics[m]["tp"] is not None]
    invalid = [m for m in metrics if metrics[m]["tp"] is None]
    return sorted(valid, key=key, reverse=True) + invalid


def rank_labels(metrics: Dict[str, Dict]) -> Dict[str, str]:
    """Dense ranks for methods with a defined MCC; exact ties share a rank."""
    labels = {method: "N/A" for method in metrics}
    valid = [method for method in rank(metrics)
             if metrics[method]["matthews_correlation_coefficient"] is not None]
    previous, current = None, 0
    for position, method in enumerate(valid, 1):
        key = (metrics[method]["matthews_correlation_coefficient"],
               metrics[method]["balanced_accuracy"], metrics[method]["failure_recall"])
        if key != previous:
            current, previous = position, key
        labels[method] = str(current)
    return labels


def fmt(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def report(data: Dict) -> str:
    s, metrics = data["holdout_summary"], data["metrics"]
    lines = [f"# RQ2b Holdout Brightness Validation: {data['model']}", "",
        "## Design", "",
        f"Only original holdout images with per-image COCO AP@[0.50:0.95] ≥ {data['ap_threshold']:.2f} were eligible. "
        "For each eligible image, gamma was searched in the bright and dark directions. Coarse gamma brackets were refined and distinct failures were selected near evenly spaced target AP values.", "",
        "The operating ranges were fixed before validation using the unadjusted RQ2b boundary means: lower = dark/right mean and upper = bright/left mean. Eligible originals are successful controls.", "",
        f"Images checked: **{s['images_checked']}**; eligible originals: **{s['eligible_originals']}**; selected transformed failures: **{s['selected_transformed_failures']}**.", "",
        "## Fixed mean-boundary ranges", "",
        "| Method | Dark mean | Dark SD | Lower | Bright mean | Bright SD | Upper |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, b in data["operating_ranges"].items():
        d, a = b["dark_boundary"], b["bright_boundary"]
        validity = "" if b["valid_interval"] else " ⚠ invalid"
        lines.append(f"| {method}{validity} | {d['mean']:.5g} | {d['sample_standard_deviation']:.5g} | {b['lower']:.5g} | {a['mean']:.5g} | {a['sample_standard_deviation']:.5g} | {b['upper']:.5g} |")
    lines += ["", "## Validation results", "",
        "| Rank | Method | Failure recall | Bright recall | Dark recall | Original acceptance | Balanced accuracy | MCC | TP | FP | TN | FN |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    labels = rank_labels(metrics)
    for method in rank(metrics):
        m = metrics[method]
        if m["tp"] is None:
            lines.append(f"| N/A | {method} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(f"| {labels[method]} | {method} | {fmt(m['failure_recall'])} | {fmt(m['bright_failure_recall'])} | {fmt(m['dark_failure_recall'])} | {fmt(m['original_acceptance_rate'])} | {fmt(m['balanced_accuracy'])} | {fmt(m['matthews_correlation_coefficient'])} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |")
    r = metrics["rms_contrast"]
    lines += ["", "## RMS contrast", "",
        f"RMS rejected **{r['tp']} of {r['tp']+r['fn']}** transformed failures and accepted **{r['tn']} of {r['tn']+r['fp']}** useful originals. Failure recall: **{fmt(r['failure_recall'])}**; original acceptance: **{fmt(r['original_acceptance_rate'])}**; MCC: **{fmt(r['matthews_correlation_coefficient'])}**.", "",
        "## Sample images", "",
        "Representative too-dark, in-range, and out-of-range images for each valid method are in `sample_images/<method>`. Each method directory contains a JSON manifest with the fixed bounds and image metadata. The out-of-range set includes both dark and bright violations and therefore intentionally overlaps the too-dark set.", "",
        "![Validation plot](brightness_validation.png)", ""]
    return "\n".join(lines)


def plot(path: Path, observations: Sequence[Dict], ranges: Dict[str, Dict], threshold: float) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(15, 18), constrained_layout=True)
    colors = {"original": "#2e7d32", "bright": "#f9a825", "dark": "#1565c0"}
    markers = {"original": "o", "bright": "^", "dark": "v"}
    for ax, method in zip(axes.flat, BRIGHTNESS_METHODS):
        b = ranges[method]
        if b["valid_interval"]:
            ax.axvspan(b["lower"], b["upper"], color="#66bb6a", alpha=.15)
            ax.axvline(b["lower"], color="#2e7d32", ls="--"); ax.axvline(b["upper"], color="#2e7d32", ls="--")
        else:
            ax.text(.5, .95, "No valid mean-boundary interval", transform=ax.transAxes,
                    ha="center", va="top", color="#c62828")
        ax.axhline(threshold, color="#c62828", ls=":")
        for side in colors:
            selected = [o for o in observations if o["side"] == side]
            if selected:
                ax.scatter([o["brightness_scores"][method] for o in selected], [o["ap"] for o in selected], s=22, alpha=.7, c=colors[side], marker=markers[side], label=side)
        ax.set(title=method, xlabel="Brightness score", ylabel="Per-image AP")
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .982), ncol=3)
    fig.suptitle("Holdout AP vs fixed mean-boundary operating ranges", fontsize=16, y=.998)
    fig.savefig(path, dpi=180); plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="yolov8", choices=get_model_names())
    p.add_argument("--rq2b-results", type=Path)
    p.add_argument("--holdout-root", type=Path, default=HOLDOUT)
    p.add_argument("--detections-root", type=Path, default=DETECTIONS)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ap-threshold", type=float, default=.5)
    p.add_argument("--gamma-min", type=float, default=GAMMA_RANGE[0])
    p.add_argument("--gamma-max", type=float, default=GAMMA_RANGE[1])
    p.add_argument("--gamma-step", type=float, default=GAMMA_STEP)
    p.add_argument("--failure-points-per-side", type=int, default=5)
    p.add_argument("--samples-per-category", type=int, default=5,
                   help="Representative images saved per method/category; 0 disables export")
    p.add_argument("--num-images", type=int, help="Limit the holdout prefix for a smoke run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.gamma_min < 1 < args.gamma_max or args.gamma_step <= 0:
        raise ValueError("Require 0 < gamma-min < 1 < gamma-max and a positive step")
    if (not 0 < args.ap_threshold <= 1 or args.failure_points_per_side < 2
            or args.samples_per_category < 0):
        raise ValueError("Invalid AP threshold or failure point count")
    source = args.rq2b_results or rq2b_default(args.model)
    ranges = derive_ranges(source, args.model)
    images = images_in_coco_order(args.holdout_root)
    if args.num_images is not None:
        images = images[:args.num_images]
    annotations_path = args.holdout_root / "annotations_coco.json"
    ground_truth = load_per_image_ground_truth(annotations_path)
    detections = args.detections_root / args.model / "original"
    originals = [baseline(p, ground_truth[p.name], detections, args.ap_threshold) for p in images]
    eligible = [(p, o) for p, o in zip(images, originals) if o["ap"] >= args.ap_threshold]
    print(f"Eligible {args.model} originals: {len(eligible)}/{len(images)}")
    if not eligible:
        raise RuntimeError("No original holdout image has AP at or above the threshold")
    for image_path, observation in eligible:
        add_baseline_scores(image_path, observation)
    model, device = load_yolo_model(args.model, args.device)
    observations, searches = [o for _, o in eligible], []
    for i, (image_path, _) in enumerate(eligible, 1):
        print(f"[{i}/{len(eligible)}] {image_path.name}")
        transformed, search = validate_image(image_path, ground_truth[image_path.name], model, str(device), args.ap_threshold, args.gamma_min, args.gamma_max, args.gamma_step, args.failure_points_per_side)
        observations += transformed; searches.append(search)
    metrics = classify(observations, ranges)
    sample_summary = {}
    if args.samples_per_category:
        sample_summary = export_sample_images(
            args.output_dir / "sample_images", observations, ranges,
            {path.name: path for path in images}, args.samples_per_category,
        )
    data = {"schema_version": 2, "model": args.model,
        "ap_definition": "per-image COCO AP@[0.50:0.95] (COCOeval stats[0])",
        "ap_threshold": args.ap_threshold, "rq2b_source": str(source.resolve()),
        "holdout_root": str(args.holdout_root.resolve()), "original_detections": str(detections.resolve()),
        "range_rule": {"lower": "dark/right boundary mean",
                       "upper": "bright/left boundary mean",
                       "standard_deviation_adjustment": "none"},
        "gamma_search": {"minimum": args.gamma_min, "maximum": args.gamma_max,
                         "step": args.gamma_step,
                         "failure_points_requested_per_side": args.failure_points_per_side,
                         "selection": "distinct evaluated gamma closest to each target AP",
                         "target_refinement_max_iterations": 4,
                         "target_ap_tolerance": 0.02,
                         "target_ap_values": failure_ap_targets(
                             args.ap_threshold, args.failure_points_per_side)},
        "holdout_summary": {"images_checked": len(images), "eligible_originals": len(eligible), "ineligible_originals": len(images)-len(eligible), "selected_transformed_failures": sum(o["actual_failure"] for o in observations)},
        "operating_ranges": ranges, "metrics": metrics, "original_screening": originals,
        "gamma_searches": searches, "validation_observations": observations,
        "sample_images": {"root": str((args.output_dir / "sample_images").resolve()),
                          "samples_per_category": args.samples_per_category,
                          "methods": sample_summary}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_results.json").write_text(json.dumps(data, indent=2))
    plot(args.output_dir / "brightness_validation.png", observations, ranges, args.ap_threshold)
    (args.output_dir / "report.md").write_text(report(data))
    print(f"Wrote validation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
