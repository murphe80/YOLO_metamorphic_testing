#!/usr/bin/env python3
"""Assess saved validation observations with three-way RQ2b brightness predictions.

Use existing RQ2b dark/bright boundary scores to form mean +/- 0.204 sample SD
uncertainty bands. Evaluation defaults to saved AP >= 0.1 filtered observations.
No inference, brightness scoring, or boundary search is run.
"""
from __future__ import annotations

import argparse
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

from utils.imgDarkBright_tuning.brightness_methods import BRIGHTNESS_METHODS  # noqa: E402
from utils.model_registry import get_model_display_name, get_model_names  # noqa: E402

OUTPUT = ROOT / "analysis/results/rq2/rq2b/validation"


def load_boundary_records(path: Path, model: str) -> List[Dict]:
    data = json.loads(path.read_text())
    models = data.get("models", [])
    models = [models] if isinstance(models, str) else models
    if model not in models:
        raise ValueError(f"{path} does not contain model {model}")
    records = [r for r in data.get("per_image_results", [])
               if r.get("model", models[0] if len(models) == 1 else None) == model]
    if not records:
        raise ValueError(f"{path} contains no boundary records for {model}")
    for record in records:
        for side in ("left", "right"):
            scores = record.get(f"scores_{side}")
            if scores is not None and (set(scores) != set(BRIGHTNESS_METHODS) or not all(
                    isinstance(value, (int, float)) and math.isfinite(value) for value in scores.values())):
                raise ValueError(f"Invalid {side} brightness scores for {record['image_name']}")
    return records


def derive_ranges(records: Sequence[Dict], sigma_multiplier: float = 0.204) -> Dict[str, Dict]:
    """Use existing RQ2b brightness boundary scores as saved; no new AP filtering."""
    result = {}
    for method in BRIGHTNESS_METHODS:
        bounds = {"sigma_multiplier": sigma_multiplier}
        for side, source_side in (("dark", "right"), ("bright", "left")):
            values = [float(r[f"scores_{source_side}"][method])
                      for r in records if r.get(f"scores_{source_side}")]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"Non-finite boundary score for {method}")
            mean = float(np.mean(values)) if values else None
            sd = float(np.std(values, ddof=1)) if len(values) >= 2 else None
            bounds[f"{side}_boundary"] = {
                "n": len(values), "mean": mean, "sample_standard_deviation": sd,
                "lower": mean - sigma_multiplier * sd if sd is not None else None,
                "upper": mean + sigma_multiplier * sd if sd is not None else None,
                "scores": values,
            }
        dark, bright = bounds["dark_boundary"], bounds["bright_boundary"]
        reason = None
        if min(dark["n"], bright["n"]) < 2:
            reason = "Need at least two saved boundary scores on each side"
        elif dark["mean"] >= bright["mean"]:
            reason = "Dark boundary mean is not below bright boundary mean"
        available = reason is None
        separated = available and dark["upper"] < bright["lower"]
        bounds.update({
            "available": available, "unavailable_reason": reason,
            "lower": dark["upper"], "upper": bright["lower"],
            "valid_interval": separated,
            "bands_overlap": available and not separated,
            "invalid_reason": reason or (None if separated else "Uncertainty bands overlap or touch"),
        })
        result[method] = bounds
    return result


def load_validation_results(path: Path, dataset: str, model: str) -> Dict:
    """Read saved observations without inference or rescoring; ignore old predictions."""
    if not path.is_file():
        raise FileNotFoundError(f"No saved validation observations at {path}; pass --validation-results PATH")
    data = json.loads(path.read_text())
    if data.get("model") != model:
        raise ValueError(f"{path} does not contain model {model}")
    if data.get("dataset") is not None and data["dataset"] != dataset:
        raise ValueError(f"{path} contains dataset {data['dataset']}, not {dataset}")
    threshold = data.get("ap_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError(f"{path}: invalid AP threshold")
    rows = data.get("validation_observations", [])
    if not rows:
        raise ValueError(f"{path}: no saved validation observations")
    for row in rows:
        scores, ap = row.get("brightness_scores") or {}, row.get("ap")
        if set(scores) != set(BRIGHTNESS_METHODS) or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in scores.values()):
            raise ValueError(f"{path}: missing or invalid brightness scores for {row.get('image_name')}")
        if not isinstance(ap, (int, float)) or not math.isfinite(ap) or not 0 <= ap <= 1:
            raise ValueError(f"{path}: invalid AP for {row.get('image_name')}")
        row["actual_failure"] = ap < threshold
        row.pop("method_predictions", None)
    return data


def div(a: int, b: int) -> Optional[float]:
    return a / b if b else None


def classify_binary(observations: Sequence[Dict], ranges: Dict[str, Dict]) -> Dict[str, Dict]:
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


def predict_brightness(value: float, bounds: Dict) -> Tuple[str, Optional[str]]:
    """Inclusive uncertainty bands take precedence; overlapping bands have no safe gap."""
    if not math.isfinite(value):
        return "uncertain", "non_finite_score"
    if not bounds["available"]:
        return "uncertain", bounds["unavailable_reason"]
    dark, bright = bounds["dark_boundary"], bounds["bright_boundary"]
    if any(b["lower"] <= value <= b["upper"] for b in (dark, bright)):
        return "uncertain", "boundary_band"
    if dark["upper"] < value < bright["lower"]:
        return "in_range", None
    return "out_of_range", None


def classify(observations: Sequence[Dict], ranges: Dict[str, Dict]) -> Dict[str, Dict]:
    # Retain support for historical mean-only JSON consumed by analysis utilities.
    if ranges and "available" not in next(iter(ranges.values())):
        return classify_binary(observations, ranges)
    output = {}
    labels = ("in_range", "uncertain", "out_of_range")
    for method, bounds in ranges.items():
        counts = [[0, 0, 0], [0, 0, 0]]  # actual success, failure
        uncertain_ap = []
        reasons = {}
        for obs in observations:
            value = obs["brightness_scores"][method]
            prediction, reason = predict_brightness(value, bounds)
            correct = (None if prediction == "uncertain" else
                       (not obs["actual_failure"] if prediction == "in_range" else obs["actual_failure"]))
            obs.setdefault("method_predictions", {})[method] = {
                "value": value, "prediction": prediction, "reason": reason,
                "correct": correct,
                "ap_residual": obs["ap"] - 0.5 if prediction == "uncertain" else None,
                "in_operating_range": None if prediction == "uncertain" else prediction == "in_range",
                "predicted_failure": None if prediction == "uncertain" else prediction == "out_of_range",
            }
            counts[int(obs["actual_failure"])][labels.index(prediction)] += 1
            if prediction == "uncertain":
                uncertain_ap.append(obs["ap"])
                reasons[reason] = reasons.get(reason, 0) + 1
        tn, uncertain_success, fp = counts[0]
        fn, uncertain_failure, tp = counts[1]
        n = len(observations)
        uncertain = uncertain_success + uncertain_failure
        definite = n - uncertain
        residuals = np.asarray(uncertain_ap, dtype=float) - 0.5
        recall, specificity = div(tp, tp + fn), div(tn, tn + fp)
        output[method] = {
            "n": n, "prediction_counts": dict(zip(labels, (tn + fn, uncertain, tp + fp))),
            "confusion_matrix": {"actual_labels": ["in_range", "out_of_range"],
                                 "predicted_labels": ["in_range", "out_of_range"],
                                 "counts": [[tn, fp], [fn, tp]]},
            "definite_count": definite, "uncertain_count": uncertain,
            "coverage": div(definite, n), "uncertain_rate": div(uncertain, n),
            "selective_accuracy": div(tp + tn, definite),
            "selective_balanced_accuracy": ((recall + specificity) / 2
                if recall is not None and specificity is not None else None),
            "correct_definite_fraction": div(tp + tn, n),
            "failure_recall": div(tp, sum(counts[1])),
            "success_acceptance_rate": div(tn, sum(counts[0])),
            "false_in_range_rate": div(fn, sum(counts[1])),
            "false_out_of_range_rate": div(fp, sum(counts[0])),
            "failure_rate_among_in_range": div(fn, tn + fn),
            "uncertain_reasons": reasons,
            "uncertain_residual_summary": ({"n": uncertain,
                "mean": float(np.mean(residuals)),
                "mean_absolute": float(np.mean(np.abs(residuals))),
                "rmse": float(np.sqrt(np.mean(residuals ** 2))),
                "min": float(np.min(residuals)), "q25": float(np.quantile(residuals, .25)),
                "median": float(np.median(residuals)), "q75": float(np.quantile(residuals, .75)),
                "max": float(np.max(residuals))} if uncertain else {"n": 0}),
            "uncertain_ap_summary": ({"n": uncertain, "mean": float(np.mean(uncertain_ap)),
                "min": min(uncertain_ap), "q25": float(np.quantile(uncertain_ap, .25)),
                "median": float(np.median(uncertain_ap)),
                "q75": float(np.quantile(uncertain_ap, .75)), "max": max(uncertain_ap)}
                if uncertain_ap else {"n": 0}),
            "unavailable_reason": bounds["unavailable_reason"],
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
    """Save disjoint in-range, uncertain, and out-of-range example sets."""
    summary = {}
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rq2b_samples_") as temporary:
        transform_root = Path(temporary)
        for method, bounds in ranges.items():
            method_dir = output_root / method
            method_dir.mkdir(parents=True, exist_ok=True)
            for stale in method_dir.glob("*.jpg"):
                stale.unlink()

            pools = {label: [o for o in observations
                             if o["method_predictions"][method]["prediction"] == label]
                     for label in ("in_range", "uncertain", "out_of_range")}
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
                        from utils.transform_base_metamorphic_rules import gen_metamorphic
                        success, transformed_path = gen_metamorphic(
                            source_image, "imgDarkBright",
                            {"brightness_constant": 0, "brightness_multiplier": 1.0,
                             "gamma": float(observation["gamma"]), "method": "gamma"},
                            os.path.splitext(source_path.name), str(transform_root / method / category),
                        )
                        if not success:
                            transformed_path = None
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
                        "prediction_reason": observation["method_predictions"][method]["reason"],
                        "source_image": observation["image_name"],
                        "kind": observation["kind"], "side": observation["side"],
                        "gamma": observation["gamma"], "ap": observation["ap"],
                        "brightness_score": score,
                    })
            manifest = {
                "method": method, "available": True, "operating_range": bounds,
                "samples_per_category": samples_per_category,
                "category_pool_sizes": {key: len(value) for key, value in pools.items()},
                "samples": entries,
            }
            (method_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            summary[method] = {"available": True, "sample_count": len(entries)}
    return summary


def fmt(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def directional_report_binary(data: Dict) -> str:
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


def metrics_table(metrics: Dict) -> List[str]:
    lines = [
        "| Method | In-range | Uncertain | Out-of-range | Coverage | Accuracy on definite predictions | Failures called in-range | Successes called out-of-range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, m in metrics.items():
        c = m["prediction_counts"]
        lines.append(f"| {method} | {c['in_range']} | {c['uncertain']} | {c['out_of_range']} | "
                     f"{fmt(m['coverage'])} | {fmt(m['selective_accuracy'])} | "
                     f"{fmt(m['false_in_range_rate'])} | {fmt(m['false_out_of_range_rate'])} |")
    return lines


def directional_report(data: Dict) -> str:
    if "available" not in next(iter(data["operating_ranges"].values())):
        return directional_report_binary(data)
    subsets = confusion_metrics(data)
    lines = ["## Classification by transformation direction", "",
             "Directional subsets exclude originals; they retain the saved transformation observations.", ""]
    for subset in ("brightened", "darkened"):
        lines += [f"### {subset.title()}", "", *metrics_table(subsets[subset]), ""]
    return "\n".join(lines)


def report(data: Dict) -> str:
    calibration = data["calibration"]
    lines = [f"# RQ2b Three-way Assessment: {data['dataset']} / {data['model']}", "",
        "## Design", "",
        f"Boundary calibration source: `{data['rq2b_source']}`.", "",
        f"Evaluation observations: `{data['validation_source']}`.", "",
        "The existing RQ2b scores_right (dark) and scores_left (bright) values form "
        "separate brightness distributions. All available saved boundary scores are used "
        "without additional AP filtering. No inference, brightness scoring, or boundary "
        "refinement is performed. Previously saved predictions are replaced; AP, brightness "
        "measurements, and the observation population are preserved.", "",
        f"Uncertainty bands use brightness mean +/- {calibration['sigma_multiplier']:g} sample SD "
        "(ddof=1). Limits are not clipped. Bands include their endpoints. In-range is "
        "strictly between the bands; out-of-range is beyond their outer limits. Overlapping "
        "bands leave no in-range interval. Insufficient samples or reversed boundary means "
        "produce only uncertain predictions with an explicit reason. Ordered means alone "
        "do not establish score monotonicity, especially for RMS contrast and the mean-shift ratio.", "",
        f"Evaluation observations: **{len(data['validation_observations'])}**. "
        f"Image names shared with calibration records: **{calibration['matching_image_names']}** "
        "(name matching alone does not establish dataset independence).", "",
        "## Brightness boundary distributions", "",
        "| Method | Side | N | Mean | Sample SD | Band lower | Band upper |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for method, bounds in data["operating_ranges"].items():
        for side in ("dark", "bright"):
            b = bounds[f"{side}_boundary"]
            lines.append(f"| {method} | {side} | {b['n']} | {fmt(b['mean'])} | "
                         f"{fmt(b['sample_standard_deviation'])} | {fmt(b['lower'])} | {fmt(b['upper'])} |")
    for method, bounds in data["operating_ranges"].items():
        if bounds["invalid_reason"]:
            lines += ["", f"- {method}: {bounds['invalid_reason']}."]
    lines += ["", "[Boundary distributions](boundary_distributions.pdf)", "",
        "Bounded methods share 0–1 x axes, 0–100% y axes, and identical bin edges. "
        "CIELAB L* is divided by 100 for display; RMS is not rescaled. "
        "The mean-shift separation-to-spread ratio is omitted from plots.", "",
        "## Three-way predictions", "",
        f"An in-range prediction is correct when AP >= {data['ap_threshold']:g}; "
        f"an out-of-range prediction is correct when AP < {data['ap_threshold']:g}. "
        "Uncertain predictions have no correct/incorrect label. They are assessed separately "
        "using the signed residual **AP - 0.5** and absolute distance **|AP - 0.5|**. "
        "Zero means AP is exactly 0.5; negative/positive residuals mean AP below/above 0.5.", "",
        "Coverage is the fraction receiving a definite prediction. Selective accuracy is "
        "accuracy on that fraction only. Confusion matrices exclude uncertain predictions. "
        "Error rates use all actual AP failures/successes as "
        "their respective denominators. Rates are fractions from 0 to 1; N/A means an empty denominator.", "",
        *metrics_table(data["metrics"]), "", directional_report(data), "",
        "## Uncertain AP residuals", "",
        "Boxes show the median and interquartile range; whiskers extend to the most extreme "
        "values within 1.5 IQR, and circles show outliers. The line inside each box is the median. "
        "All methods share the same signed residual axis. N/A means no uncertain observations.", "",
        "[Uncertain AP residuals](uncertain_residuals.pdf)", "",
        "| Method | N | Mean residual | Mean absolute residual | RMSE | Min | Q25 | Median | Q75 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, m in data["metrics"].items():
        a = m["uncertain_residual_summary"]
        values = " | ".join(fmt(a.get(k)) for k in
                            ("mean", "mean_absolute", "rmse", "min", "q25", "median", "q75", "max"))
        lines.append(f"| {method} | {a['n']} | {values} |")
    lines += ["", "## Sample images", "",
        "Disjoint in_range, uncertain, and out_of_range samples and manifests are under "
        "`sample_images/<method>` when sample export is enabled.", "",
        "[AP versus brightness](brightness_validation.pdf)", "",
        "## Confusion matrices for definite predictions", "",
        "Rows: actual in-range/out-of-range based on AP. Columns: predicted in-range/out-of-range. "
        "Only definite predictions contribute. Cells contain counts and percentages within each "
        "actual row; all panels share the 0–100% colour scale. Each panel reports overall "
        "accuracy across the predictions included in its matrix.", "",
        "[Overall](confusion_matrices_overall.pdf)", "",
        "[Brightened](confusion_matrices_brightened.pdf)", "",
        "[Darkened](confusion_matrices_darkened.pdf)", ""]
    return "\n".join(lines)


def confusion_metrics(data: Dict) -> Dict[str, Dict]:
    """Recompute independent subset counts without mutating saved observations."""
    observations = [{"kind": o["kind"], "side": o["side"], "ap": o["ap"],
                     "actual_failure": o["ap"] < data["ap_threshold"],
                     "brightness_scores": o["brightness_scores"]}
                    for o in data["validation_observations"]]
    subsets = {"overall": observations,
               "brightened": [o for o in observations if o["side"] == "bright" and o["kind"] != "original"],
               "darkened": [o for o in observations if o["side"] == "dark" and o["kind"] != "original"]}
    return {name: classify(rows, data["operating_ranges"]) for name, rows in subsets.items()}


def plot_confusion_matrices_binary(output_dir: Path, data: Dict) -> None:
    """One multi-method PDF per subset, with raw counts and row-normalised colour."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for subset, results in confusion_metrics(data).items():
        results = {m: values for m, values in results.items()
                   if m != "mean_shift_separation_spread_ratio"}
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
            heatmap = ax.pcolormesh(
                np.arange(3) - .5, np.arange(3) - .5, proportions,
                cmap="Blues", vmin=0, vmax=1, rasterized=False)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            for row, labels in enumerate((("TP", "FN"), ("FP", "TN"))):
                for col, label in enumerate(labels):
                    percent = f"{100 * proportions[row, col]:.1f}%" if totals[row, 0] else "N/A"
                    ax.text(col, row, f"{label}\n{counts[row, col]:,}\n{percent}",
                            ha="center", va="center", fontsize=12,
                            color="white" if proportions[row, col] > .55 else "black")
            accuracy = div(m["tp"] + m["tn"], m["n"])
            acc = "N/A" if accuracy is None else f"{100 * accuracy:.2f}%"
            ax.set_title(f"{title}\nOverall accuracy {acc}", fontsize=11)
            ax.set_xticks([0, 1], ["Failure\n(outside range)", "Success\n(inside range)"])
            ax.set_yticks([0, 1], ["Failure", "Success"])
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Actual class (AP)")
        if heatmap is not None:
            colorbar = fig.colorbar(heatmap, ax=list(axes.flat), shrink=.8,
                         label="Fraction within actual class (row)")
            colorbar.solids.set_rasterized(False)
        n = next(iter(results.values()))["n"]
        scope = "originals + all transformations" if subset == "overall" else "transformations only"
        fig.suptitle(f"{get_model_display_name(data['model'])} : {subset} ({scope}; n={n:,})\n"
                     f"Positive = failure: AP < {data['ap_threshold']:.2f} | Cells: count and row percentage",
                     fontsize=16)
        fig.savefig(output_dir / f"confusion_matrices_{subset}.pdf", dpi=160)
        plt.close(fig)


def plot_confusion_matrices(output_dir: Path, data: Dict) -> None:
    if "available" not in next(iter(data["operating_ranges"].values())):
        return plot_confusion_matrices_binary(output_dir, data)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subset, results in confusion_metrics(data).items():
        results = {m: values for m, values in results.items()
                   if m != "mean_shift_separation_spread_ratio"}
        fig, axes = plt.subplots(math.ceil(len(results) / 2), 2,
                                 figsize=(14, 4 * math.ceil(len(results) / 2)),
                                 squeeze=False, constrained_layout=True)
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, (method, m) in zip(axes.flat, results.items()):
            ax.set_visible(True)
            counts = np.array(m["confusion_matrix"]["counts"])
            totals = counts.sum(axis=1, keepdims=True)
            proportions = np.divide(counts, totals, out=np.zeros((2, 2)), where=totals != 0)
            heatmap = ax.pcolormesh(
                np.arange(3) - .5, np.arange(3) - .5, proportions,
                cmap="Blues", vmin=0, vmax=1, rasterized=False)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            for row in range(2):
                for col in range(2):
                    percent = f"{100 * proportions[row, col]:.1f}%" if totals[row, 0] else "N/A"
                    ax.text(col, row, f"{counts[row, col]:,}\n{percent}", ha="center", va="center",
                            color="white" if proportions[row, col] > .55 else "black")
            ax.set_xticks([0, 1], ["In-range", "Out-of-range"])
            ax.set_yticks([0, 1], ["In-range", "Out-of-range"])
            ax.set_xlabel("Brightness prediction")
            ax.set_ylabel("Actual AP class")
            accuracy = m['selective_accuracy']
            acc = "N/A" if accuracy is None else f"{100 * accuracy:.2f}%"
            ax.set_title(f"{method}\nOverall accuracy {acc}")
        colorbar = fig.colorbar(heatmap, ax=list(axes.flat), shrink=.7, label="Fraction within actual class (definite predictions)")
        colorbar.solids.set_rasterized(False)
        fig.suptitle(f"{get_model_display_name(data['model'])} : {subset}")
        fig.savefig(output_dir / f"confusion_matrices_{subset}.pdf", dpi=160)
        plt.close(fig)


def plot_uncertain_residuals(path: Path, data: Dict) -> None:
    """Compare signed AP residuals for uncertain predictions, with median lines."""
    methods = [m for m in data["operating_ranges"]
               if m != "mean_shift_separation_spread_ratio"]
    residuals = []
    for method in methods:
        residuals.append(np.asarray([
            o["ap"] - 0.5 for o in data["validation_observations"]
            if o["method_predictions"][method]["prediction"] == "uncertain"
        ], dtype=float))
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    positions = [i for i, values in enumerate(residuals) if values.size]
    values = [residuals[i] for i in positions]
    if values:
        ax.boxplot(values, positions=positions, vert=False, widths=.55, showmeans=False,
                   patch_artist=True, manage_ticks=False,
                   boxprops={"facecolor": "lightblue"},
                   flierprops={"marker": "o", "markersize": 3, "alpha": .35})
    for i, sample in enumerate(residuals):
        if not sample.size:
            ax.text(0, i, "No uncertain observations", ha="center", va="center")
    ax.set_xlim(-.5, .5)
    ax.set_xlabel("Signed residual AP - 0.5")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.grid(axis="x", alpha=.25)
    ax.set_yticks(range(len(methods)),
                  [f"{m} (n={len(r):,})" for m, r in zip(methods, residuals)])
    ax.set_ylim(len(methods) - .5, -.5)
    fig.suptitle(f"{data['dataset']} / {get_model_display_name(data['model'])} : uncertain predictions\n"
                 "Boxes: middle 50%; line: median; circles: outliers")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_boundary_distributions(path: Path, ranges: Dict[str, Dict]) -> None:
    """Compare plotted brightness scores with identical bins and axes.

    L* / 100 changes units only. RMS remains in its native 0–1 luminance units;
    it is not stretched to fill the axis. Percentages normalize sample counts.
    """
    ratio_method = "mean_shift_separation_spread_ratio"
    bounded = [(m, b) for m, b in ranges.items() if m != ratio_method]
    bins = np.linspace(0, 1, 26)
    fig, axes = plt.subplots(math.ceil(len(bounded) / 2), 2,
                             figsize=(14, 3.5 * math.ceil(len(bounded) / 2)), squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (method, bounds) in zip(axes.flat, bounded):
        ax.set_visible(True)
        scale = 100.0 if method == "cielab_lightness" else 1.0
        for side, color in (("dark", "#1565c0"), ("bright", "#ef6c00")):
            boundary = bounds[f"{side}_boundary"]
            values = np.asarray(boundary["scores"]) / scale
            if len(values):
                ax.hist(values, bins=bins, weights=np.full(len(values), 100 / len(values)),
                        alpha=.45, color=color, label=f"{side} (n={len(values)})")
                ax.axvline(boundary["mean"] / scale, color=color)
            if boundary["lower"] is not None:
                ax.axvspan(boundary["lower"] / scale, boundary["upper"] / scale,
                           color=color, alpha=.12)
        ax.set(title=method + (" (L* / 100)" if scale == 100 else ""),
               xlabel="Brightness score (0–1 units)", ylabel="Boundary images (%)",
               xlim=(0, 1), ylim=(0, 100))
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.tick_params(labelbottom=True, labelleft=True)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    multiplier = next(iter(ranges.values()))["sigma_multiplier"]
    fig.suptitle(f"Boundary distributions: mean ± {multiplier:g} sample SD\n"
                 "Shared axes and 0.04-wide bins")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_binary(path: Path, observations: Sequence[Dict], ranges: Dict[str, Dict], threshold: float) -> None:
    ranges = {m: b for m, b in ranges.items() if m != "mean_shift_separation_spread_ratio"}
    fig, axes = plt.subplots(math.ceil(len(ranges) / 2), 2,
                             figsize=(15, 4.5 * math.ceil(len(ranges) / 2)),
                             squeeze=False, constrained_layout=True)
    for ax in axes.flat:
        ax.set_visible(False)
    colors = {"original": "#2e7d32", "bright": "#f9a825", "dark": "#1565c0", "neutral": "#757575"}
    markers = {"original": "o", "bright": "^", "dark": "v", "neutral": "s"}
    for ax, method in zip(axes.flat, ranges):
        ax.set_visible(True)
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


def plot(path: Path, observations: Sequence[Dict], ranges: Dict[str, Dict], threshold: float) -> None:
    ranges = {m: b for m, b in ranges.items() if m != "mean_shift_separation_spread_ratio"}
    if "available" not in next(iter(ranges.values())):
        return plot_binary(path, observations, ranges, threshold)
    fig, axes = plt.subplots(math.ceil(len(ranges) / 2), 2,
                             figsize=(15, 4 * math.ceil(len(ranges) / 2)), squeeze=False,
                             constrained_layout=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (method, bounds) in zip(axes.flat, ranges.items()):
        ax.set_visible(True)
        if bounds["available"]:
            for side in ("dark", "bright"):
                b = bounds[f"{side}_boundary"]
                ax.axvspan(b["lower"], b["upper"], color="#f9a825", alpha=.12)
            if bounds["valid_interval"]:
                ax.axvspan(bounds["lower"], bounds["upper"], color="#66bb6a", alpha=.12)
        for label, color in (("in_range", "#2e7d32"), ("uncertain", "#f9a825"),
                             ("out_of_range", "#c62828")):
            selected = [o for o in observations if predict_brightness(o["brightness_scores"][method], bounds)[0] == label]
            ax.scatter([o["brightness_scores"][method] for o in selected],
                       [o["ap"] for o in selected], s=8, alpha=.35, c=color, label=label)
        ax.axhline(threshold, color="black", ls=":")
        ax.set(title=method, xlabel="Brightness score", ylabel="Per-image AP", ylim=(-.02, 1.02))
        ax.legend()
    fig.suptitle("Saved AP and three-way brightness predictions")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="yolov8", choices=get_model_names())
    p.add_argument("--dataset", default="val", help="Saved evaluation dataset (default: val)")
    p.add_argument("--rq2b-results", type=Path,
                   help="Existing brightness_scoring/brute_force/<model>/results.json")
    p.add_argument("--validation-results", type=Path,
                   help="Saved AP and brightness observations; defaults to validation/<dataset>/<model>/filtered_ap_0.1/validation_results.json")
    p.add_argument("--holdout-root", type=Path, help="Optional image root for sample export")
    p.add_argument("--output-dir", type=Path, default=OUTPUT,
                   help="Output base; reports go under <dataset>/<model>/three_class/")
    p.add_argument("--sigma-multiplier", type=float, default=0.204,
                   help="Uncertainty band half-width in sample SD (default: 0.204)")
    p.add_argument("--samples-per-category", type=int, default=0,
                   help="Optional sample export (default: 0, no source images needed)")
    args = p.parse_args(argv)
    if not math.isfinite(args.sigma_multiplier) or args.sigma_multiplier <= 0:
        p.error("Require a finite positive sigma multiplier")
    if args.samples_per_category < 0:
        p.error("Sample count must be nonnegative")
    if args.dataset in {".", ".."} or not args.dataset or "/" in args.dataset or "\\" in args.dataset:
        p.error("--dataset must be a single directory name")
    args.rq2b_results = args.rq2b_results or (
        ROOT / "analysis/results/rq2/rq2b/brightness_scoring/brute_force" / args.model / "results.json")
    args.validation_results = args.validation_results or (
        OUTPUT / args.dataset / args.model / "filtered_ap_0.1" / "validation_results.json")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / args.dataset / args.model / "three_class"
    records = load_boundary_records(args.rq2b_results, args.model)
    cached = load_validation_results(args.validation_results, args.dataset, args.model)
    print(f"Loaded RQ2b boundary scores: {args.rq2b_results}")
    print(f"Loaded evaluation observations: {args.validation_results}; no inference or scoring")
    observations = cached["validation_observations"]
    ranges = derive_ranges(records, args.sigma_multiplier)
    for method, bounds in ranges.items():
        if bounds["invalid_reason"]:
            print(f"{method}: {bounds['invalid_reason']}")
    metrics = classify(observations, ranges)
    sample_summary = {}
    if args.samples_per_category:
        image_root = args.holdout_root or Path(cached["holdout_root"])
        sample_summary = export_sample_images(
            output_dir / "sample_images", observations, ranges,
            {o["image_name"]: image_root / "images" / o["image_name"] for o in observations},
            args.samples_per_category)
    # Preserve the source population, AP threshold, summaries, and any AP-filter metadata.
    data = {**cached, "schema_version": 6, "dataset": args.dataset,
        "rq2b_source": str(args.rq2b_results.resolve()),
        "validation_source": str(args.validation_results.resolve()),
        "inference_rerun": False, "brightness_scoring_rerun": False,
        "calibration": {
            "source": str(args.rq2b_results.resolve()),
            "sigma_multiplier": args.sigma_multiplier, "sd_ddof": 1,
            "boundary_selection": "all available saved RQ2b boundary scores; no new AP filtering",
            "matching_image_names": len({r["image_name"] for r in records} &
                                        {o["image_name"] for o in observations}),
        },
        "range_rule": {"uncertain": "inside either inclusive brightness mean +/- k sample SD band",
                       "in_range": "strictly between dark upper and bright lower band limits",
                       "out_of_range": "outside both bands and the in-range interval",
                       "unavailable": "uncertain with reason", "limits_clipped": False},
        "evaluation_rule": {
            "in_range": {"ap_min": cached["ap_threshold"], "inclusive": True},
            "out_of_range": {"ap_max": cached["ap_threshold"], "inclusive": False},
            "uncertain": {"residual": "AP - 0.5", "reference_ap": 0.5, "correctness_assessed": False},
            "confusion_matrix_rows": "actual AP class; definite predictions only",
        },
        "operating_ranges": ranges, "metrics": metrics,
        "sample_images": {"root": str((output_dir / "sample_images").resolve()),
                          "samples_per_category": args.samples_per_category, "methods": sample_summary}}
    data.pop("scores_source", None)
    data["confusion_metrics"] = confusion_metrics(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_results.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    plot(output_dir / "brightness_validation.pdf", observations, ranges, cached["ap_threshold"])
    plot_boundary_distributions(output_dir / "boundary_distributions.pdf", ranges)
    plot_confusion_matrices(output_dir, data)
    plot_uncertain_residuals(output_dir / "uncertain_residuals.pdf", data)
    (output_dir / "report.md").write_text(report(data))
    print(f"Wrote three-way assessment to {output_dir}")


if __name__ == "__main__":
    main()
