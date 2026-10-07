#!/usr/bin/env python3
"""Validate fixed RQ2b brightness ranges on gamma-transformed dataset images.

Only images whose existing original detections have per-image AP >= 0.5 are
eligible. Every point in the RQ2a brute-force gamma grid is evaluated and
retained, including successful transformations.
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
from typing import Dict, List, Optional, Sequence, Tuple

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
        raise ValueError("No validation images declared")
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing validation image: {missing[0]}")
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


def gamma_grid(lo: float, hi: float, step: float) -> List[float]:
    """Use the same grid and endpoint rule as the RQ2a brute-force search."""
    values = np.arange(lo, hi, step)
    if not np.isclose(values[-1], hi):
        values = np.append(values, hi)
    return [float(value) for value in values]


def validate_image(path: Path, ground_truth: Dict, model, device: str, threshold: float,
                   lo: float, hi: float, step: float) -> Tuple[List[Dict], Dict]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read {path}")
    observations, evaluations = [], []
    for gamma in gamma_grid(lo, hi, step):
        # Release intermediate image/detection files after each evaluation.
        with tempfile.TemporaryDirectory(prefix="rq2b_holdout_") as tmp:
            work = Path(tmp)
            transformed_path = apply_gamma_transformation(image, gamma, path.name, str(work))
            if not transformed_path:
                raise RuntimeError(f"Gamma transform failed: {path.name}, {gamma}")
            det_dir = work / "detections"
            run_yolo_inference(model, transformed_path, str(det_dir), device,
                               original_image_name=path.name)
            ap = quiet_ap(det_dir, ground_truth)
            if not math.isfinite(ap) or not 0 <= ap <= 1:
                raise ValueError(f"Invalid AP for {path.name} at gamma={gamma}: {ap}")
            transformed = cv2.imread(str(transformed_path))
            if transformed is None:
                raise RuntimeError(f"Could not score gamma image at {gamma}")
            side = "bright" if gamma < 1 else "dark" if gamma > 1 else "neutral"
            observations.append({
                "image_name": path.name, "kind": "transformed",
                "side": side, "gamma": gamma, "ap": ap,
                "actual_failure": ap < threshold,
                "brightness_scores": score_image(transformed),
            })
            evaluations.append({"gamma": gamma, "ap": ap})
    return observations, {"image_name": path.name, "search_evaluations": evaluations,
                          "unique_gamma_inferences": len(evaluations)}


def div(a: int, b: int) -> Optional[float]:
    return a / b if b else None


def classify(observations: Sequence[Dict], ranges: Dict[str, Dict]) -> Dict[str, Dict]:
    output = {}
    for method, bounds in ranges.items():
        if not bounds["valid_interval"]:
            output[method] = {
                "n": len(observations), "tp": None, "fp": None, "tn": None, "fn": None,
                "failure_recall": None, "original_acceptance_rate": None,
                "success_acceptance_rate": None, "transformed_success_acceptance_rate": None,
                "balanced_accuracy": None, "matthews_correlation_coefficient": None,
                "bright_failure_recall": None, "dark_failure_recall": None,
                "unavailable_reason": bounds["invalid_reason"],
            }
            continue
        tp = fp = tn = fn = 0
        successes = {"original": [0, 0], "transformed": [0, 0]}
        sides = {"bright": [0, 0], "dark": [0, 0]}
        for obs in observations:
            value = obs["brightness_scores"][method]
            outside = value < bounds["lower"] or value > bounds["upper"]
            obs.setdefault("method_predictions", {})[method] = {
                "value": value, "in_operating_range": not outside,
                "predicted_failure": outside}
            failure = obs["actual_failure"]
            if not failure:
                group = "original" if obs["kind"] == "original" else "transformed"
                successes[group][0] += int(not outside)
                successes[group][1] += 1
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
            "failure_recall": recall, "success_acceptance_rate": specificity,
            "original_acceptance_rate": div(*successes["original"]),
            "transformed_success_acceptance_rate": div(*successes["transformed"]),
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

    ``out_of_range`` contains brightness scores on either side of the interval, so it
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


def directional_report(data: Dict) -> str:
    """Report each transformation direction independently, excluding originals."""
    lines = ["## Classification by transformation direction", "",
             "Accuracy is the fraction correctly classified: AP below the threshold "
             "and outside the operating range, or AP at/above the threshold and inside it. "
             "Each table uses only its stated direction; original images are excluded. "
             "Balanced accuracy averages successful-image acceptance and failure rejection.", ""]

    def percent(value: Optional[float]) -> str:
        return "N/A" if value is None else f"{100 * value:.2f}%"

    for side, title in (("bright", "Brightened images (gamma < 1)"),
                        ("dark", "Darkened images (gamma > 1)")):
        observations = [
            {"kind": o["kind"], "side": side,
             "actual_failure": o["ap"] < data["ap_threshold"],
             "brightness_scores": o["brightness_scores"]}
            for o in data["validation_observations"]
            if o["side"] == side and o["kind"] != "original"
        ]
        failures = sum(o["actual_failure"] for o in observations)
        lines += [f"### {title}", "",
                  f"Images: **{len(observations)}**; successes: **{len(observations)-failures}**; "
                  f"failures: **{failures}** (AP threshold: {data['ap_threshold']:.2f}).", "",
                  "| Method | Correct / total | Accuracy | Balanced accuracy | Successful images accepted | Failures rejected |",
                  "|---|---:|---:|---:|---:|---:|"]
        for method, m in classify(observations, data["operating_ranges"]).items():
            correct = None if m["tp"] is None else m["tp"] + m["tn"]
            count = "N/A" if correct is None else f"{correct} / {m['n']}"
            accuracy = None if correct is None else div(correct, m["n"])
            lines.append(f"| {method} | {count} | {percent(accuracy)} | "
                         f"{percent(m['balanced_accuracy'])} | "
                         f"{percent(m['success_acceptance_rate'])} | {percent(m['failure_recall'])} |")
        lines.append("")
    return "\n".join(lines)


def report(data: Dict) -> str:
    s, metrics = data["holdout_summary"], data["metrics"]
    lines = [f"# RQ2b Brightness Validation: {data.get('dataset', 'comp_labelled')} / {data['model']}", "",
        "## Design", "",
        f"Only original validation images with per-image COCO AP@[0.50:0.95] ≥ {data['ap_threshold']:.2f} were eligible. "
        "For each eligible image, the full RQ2a brute-force gamma grid was evaluated. All transformations were retained, including AP at or above the threshold; no interpolation or AP-based selection was used.", "",
        "The operating ranges were fixed before validation using the unadjusted RQ2b boundary means: lower = dark/right mean and upper = bright/left mean. Eligible originals are successful controls.", "",
        f"Images checked: **{s['images_checked']}**; eligible originals: **{s['eligible_originals']}**; transformations: **{s['transformed_images']}**; transformed successes: **{s['transformed_successes']}**; transformed failures: **{s['transformed_failures']}**.", "",
        "## Fixed mean-boundary ranges", "",
        "| Method | Dark mean | Dark SD | Lower | Bright mean | Bright SD | Upper |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, b in data["operating_ranges"].items():
        d, a = b["dark_boundary"], b["bright_boundary"]
        validity = "" if b["valid_interval"] else " ⚠ invalid"
        lines.append(f"| {method}{validity} | {d['mean']:.5g} | {d['sample_standard_deviation']:.5g} | {b['lower']:.5g} | {a['mean']:.5g} | {a['sample_standard_deviation']:.5g} | {b['upper']:.5g} |")
    lines += ["", "## Validation results", "",
        "| Rank | Method | Failure recall | Bright recall | Dark recall | Original acceptance | Transformed success acceptance | All success acceptance | Balanced accuracy | MCC | TP | FP | TN | FN |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    labels = rank_labels(metrics)
    for method in rank(metrics):
        m = metrics[method]
        if m["tp"] is None:
            lines.append(f"| N/A | {method} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(f"| {labels[method]} | {method} | {fmt(m['failure_recall'])} | {fmt(m['bright_failure_recall'])} | {fmt(m['dark_failure_recall'])} | {fmt(m['original_acceptance_rate'])} | {fmt(m['transformed_success_acceptance_rate'])} | {fmt(m['success_acceptance_rate'])} | {fmt(m['balanced_accuracy'])} | {fmt(m['matthews_correlation_coefficient'])} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |")
    lines += ["", directional_report(data)]
    r = metrics["rms_contrast"]
    lines += ["", "## RMS contrast", "",
        f"RMS rejected **{r['tp']} of {r['tp']+r['fn']}** transformed failures and accepted **{r['tn']} of {r['tn']+r['fp']}** successful images (original and transformed). Failure recall: **{fmt(r['failure_recall'])}**; transformed success acceptance: **{fmt(r['transformed_success_acceptance_rate'])}**; MCC: **{fmt(r['matthews_correlation_coefficient'])}**.", "",
        "Success acceptance is the fraction of images with AP at or above the threshold whose brightness score is inside the operating range. Balanced accuracy uses failure recall and acceptance of all successful images. TP/FP/TN/FN include originals and every transformation.", "",
        "## Sample images", "",
        "Representative too-dark, in-range, and out-of-range images for each valid method are in `sample_images/<method>`. Each method directory contains a JSON manifest with the fixed bounds and image metadata. The out-of-range set includes both dark and bright violations and therefore intentionally overlaps the too-dark set.", "",
        "![Validation plot](brightness_validation.png)", "",
        "## Confusion matrices", "",
        "Failure (AP below the threshold) is the positive class. Rows show actual "
        "failure/success; columns show predicted failure (outside range)/success "
        "(inside range). Cells show counts and percentages within each actual class; "
        "colour uses the same 0–100% scale for every method. Overall includes eligible "
        "originals and all transformations; directional plots exclude originals.", "",
        "![Overall confusion matrices](confusion_matrices_overall.png)", "",
        "![Brightened-only confusion matrices](confusion_matrices_brightened.png)", "",
        "![Darkened-only confusion matrices](confusion_matrices_darkened.png)", ""]
    return "\n".join(lines)


def confusion_metrics(data: Dict) -> Dict[str, Dict]:
    """Recompute independent subset counts without mutating saved observations."""
    observations = [{"kind": o["kind"], "side": o["side"],
                     "actual_failure": o["ap"] < data["ap_threshold"],
                     "brightness_scores": o["brightness_scores"]}
                    for o in data["validation_observations"]]
    subsets = {"overall": observations,
               "brightened": [o for o in observations if o["side"] == "bright" and o["kind"] != "original"],
               "darkened": [o for o in observations if o["side"] == "dark" and o["kind"] != "original"]}
    return {name: classify(rows, data["operating_ranges"]) for name, rows in subsets.items()}


def plot_confusion_matrices(output_dir: Path, data: Dict) -> None:
    """One multi-method PNG per subset, with raw counts and row-normalised colour."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for subset, results in confusion_metrics(data).items():
        ncols = 4
        nrows = math.ceil(len(results) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(19, 5 * nrows),
                                 squeeze=False, constrained_layout=True)
        for ax in axes.flat:
            ax.set_visible(False)
        heatmap = None
        for ax, (method, m) in zip(axes.flat, results.items()):
            ax.set_visible(True)
            title = method.replace("_", " ")
            if m["tp"] is None:
                ax.set_title(title, wrap=True)
                ax.text(.5, .5, "Unavailable\nInvalid operating interval",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            counts = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
            totals = counts.sum(axis=1, keepdims=True)
            proportions = np.divide(counts, totals, out=np.zeros((2, 2)), where=totals != 0)
            heatmap = ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
            for row, labels in enumerate((("TP", "FN"), ("FP", "TN"))):
                for col, label in enumerate(labels):
                    percent = f"{100 * proportions[row, col]:.1f}%" if totals[row, 0] else "N/A"
                    ax.text(col, row, f"{label}\n{counts[row, col]:,}\n{percent}",
                            ha="center", va="center", fontsize=12,
                            color="white" if proportions[row, col] > .55 else "black")
            accuracy = div(m["tp"] + m["tn"], m["n"])
            acc = "N/A" if accuracy is None else f"{100 * accuracy:.2f}%"
            ba = "N/A" if m["balanced_accuracy"] is None else f"{100 * m['balanced_accuracy']:.2f}%"
            ax.set_title(f"{title}\nAccuracy {acc} | Balanced {ba}", fontsize=11)
            ax.set_xticks([0, 1], ["Failure\n(outside range)", "Success\n(inside range)"])
            ax.set_yticks([0, 1], ["Failure", "Success"])
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Actual class (AP)")
        if heatmap is not None:
            fig.colorbar(heatmap, ax=list(axes.flat), shrink=.8,
                         label="Fraction within actual class (row)")
        n = next(iter(results.values()))["n"]
        scope = "originals + all transformations" if subset == "overall" else "transformations only"
        fig.suptitle(f"{data['model']} — {subset} ({scope}; n={n:,})\n"
                     f"Positive = failure: AP < {data['ap_threshold']:.2f} | Cells: count and row percentage",
                     fontsize=16)
        fig.savefig(output_dir / f"confusion_matrices_{subset}.png", dpi=160)
        plt.close(fig)


def plot(path: Path, observations: Sequence[Dict], ranges: Dict[str, Dict], threshold: float) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(15, 18), constrained_layout=True)
    colors = {"original": "#2e7d32", "bright": "#f9a825", "dark": "#1565c0", "neutral": "#757575"}
    markers = {"original": "o", "bright": "^", "dark": "v", "neutral": "s"}
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
                ax.scatter([o["brightness_scores"][method] for o in selected], [o["ap"] for o in selected], s=10, alpha=.35, c=colors[side], marker=markers[side], label=side)
        ax.set(title=method, xlabel="Brightness score", ylabel="Per-image AP", ylim=(-.02, 1.02))
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .982), ncol=3)
    fig.suptitle("Validation AP vs fixed mean-boundary operating ranges", fontsize=16, y=.998)
    fig.savefig(path, dpi=180); plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="yolov8", choices=get_model_names())
    p.add_argument("--rq2b-results", type=Path)
    p.add_argument("--dataset", default="comp_labelled",
                   help="Dataset under data/base_images (default: comp_labelled)")
    p.add_argument("--holdout-root", type=Path,
                   help="Override the selected dataset's image/annotation root")
    p.add_argument("--detections-root", type=Path,
                   help="Override results/<dataset>, containing <model>/original")
    p.add_argument("--output-dir", type=Path, default=OUTPUT,
                   help="Output base directory; outputs go under <dataset>/<model>/")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ap-threshold", type=float, default=.5)
    p.add_argument("--gamma-min", type=float, default=GAMMA_RANGE[0])
    p.add_argument("--gamma-max", type=float, default=GAMMA_RANGE[1])
    p.add_argument("--gamma-step", type=float, default=GAMMA_STEP)
    p.add_argument("--samples-per-category", type=int, default=5,
                   help="Representative images saved per method/category; 0 disables export")
    p.add_argument("--num-images", type=int, help="Limit the dataset prefix for a smoke run")
    args = p.parse_args(argv)
    if args.dataset in {".", ".."} or not args.dataset or "/" in args.dataset or "\\" in args.dataset:
        p.error("--dataset must be a single directory name")
    if args.holdout_root is None:
        args.holdout_root = ROOT / "data/base_images" / args.dataset
    if args.detections_root is None:
        # Preserve the historical detection directory for the competition holdout.
        args.detections_root = (DETECTIONS if args.dataset == "comp_labelled"
                                else ROOT / "results" / args.dataset)
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / args.dataset / args.model
    if not 0 < args.gamma_min < 1 < args.gamma_max or args.gamma_step <= 0:
        raise ValueError("Require 0 < gamma-min < 1 < gamma-max and a positive step")
    if not 0 < args.ap_threshold <= 1 or args.samples_per_category < 0:
        raise ValueError("Invalid AP threshold or sample count")
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
        raise RuntimeError("No original validation image has AP at or above the threshold")
    for image_path, observation in eligible:
        add_baseline_scores(image_path, observation)
    model, device = load_yolo_model(args.model, args.device)
    observations, searches = [o for _, o in eligible], []
    for i, (image_path, _) in enumerate(eligible, 1):
        print(f"[{i}/{len(eligible)}] {image_path.name}")
        transformed, search = validate_image(image_path, ground_truth[image_path.name], model, str(device), args.ap_threshold, args.gamma_min, args.gamma_max, args.gamma_step)
        observations += transformed; searches.append(search)
    metrics = classify(observations, ranges)
    sample_summary = {}
    if args.samples_per_category:
        sample_summary = export_sample_images(
            output_dir / "sample_images", observations, ranges,
            {path.name: path for path in images}, args.samples_per_category,
        )
    data = {"schema_version": 3, "model": args.model, "dataset": args.dataset,
        "ap_definition": "per-image COCO AP@[0.50:0.95] (COCOeval stats[0])",
        "ap_threshold": args.ap_threshold, "rq2b_source": str(source.resolve()),
        "holdout_root": str(args.holdout_root.resolve()), "original_detections": str(detections.resolve()),
        "range_rule": {"lower": "dark/right boundary mean",
                       "upper": "bright/left boundary mean",
                       "standard_deviation_adjustment": "none"},
        "gamma_search": {"minimum": args.gamma_min, "maximum": args.gamma_max,
                         "step": args.gamma_step,
                         "selection": "all grid points; no interpolation or AP filtering",
                         "values": gamma_grid(args.gamma_min, args.gamma_max, args.gamma_step)},
        "holdout_summary": {
            "images_checked": len(images), "eligible_originals": len(eligible),
            "ineligible_originals": len(images)-len(eligible),
            "transformed_images": len(observations)-len(eligible),
            "transformed_failures": sum(o["actual_failure"] for o in observations),
            "transformed_successes": sum(not o["actual_failure"] and o["kind"] != "original"
                                         for o in observations)},
        "operating_ranges": ranges, "metrics": metrics, "original_screening": originals,
        "gamma_searches": searches, "validation_observations": observations,
        "sample_images": {"root": str((output_dir / "sample_images").resolve()),
                          "samples_per_category": args.samples_per_category,
                          "methods": sample_summary}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_results.json").write_text(json.dumps(data, indent=2))
    plot(output_dir / "brightness_validation.png", observations, ranges, args.ap_threshold)
    plot_confusion_matrices(output_dir, data)
    (output_dir / "report.md").write_text(report(data))
    print(f"Wrote validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
