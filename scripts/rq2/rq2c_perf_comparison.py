#!/usr/bin/env python3
"""
RQ2.c: Brightness-Threshold Performance Distribution

The final part of RQ2: for the operational brightness thresholds identified
in RQ2.a (the gamma values at which each image's AP crosses the target),
compare per-image AP at the original image against AP at the lower and
upper gamma thresholds, across the whole dataset.

For each YOLO model, this produces 3 separate PDFs with one violin plot each
(AP at Original / Lower threshold / Upper threshold), showing the spread of
per-image AP values across the dataset.

Additional overlay PDFs can be generated from the two saved RQ2.c summary
JSON files with --overlay-from-summaries. These also use separate files and
overlap the search methods only at the lower and upper thresholds.

"Lower" and "upper" refer to the two operational gamma crossings found by
RQ2.a (``gamma_left``/``gamma_right``)

Detections for the original image and both thresholds were already computed
by the RQ2.a search scripts and saved to disk, so this script only needs to
locate them and re-run COCO evaluation to obtain per-image AP.

Usage:
    # All models with available RQ2.a results, using golden-section thresholds
    python scripts/rq2/rq2c_perf_comparison.py

    # Specific models / search method
    python scripts/rq2/rq2c_perf_comparison.py --models yolov8 yolov5 --search-method brute_force

    # Quick test on a handful of images
    python scripts/rq2/rq2c_perf_comparison.py --num-images 10
"""

import os
import re
import sys
import json
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.model_registry import get_model_names
from analysis.utils.format_detections import format_detections
from analysis.coco_eval import evaluate_coco
from analysis.visualisation.plot_confidence_change import get_model_color, get_model_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "analysis" / "results" / "rq2"
SEARCH_METHODS = ["golden_section", "brute_force"]
THRESHOLDS = ("original", "lower", "upper")
AP_LABEL = "AP (IoU 0.50:0.95)"
METHOD_STYLES = {
    "brute_force": ("Brute force", "#0072B2"),
    "golden_section": ("Golden search", "#D55E00"),
}
THRESHOLD_LABELS = {
    "original": "Original",
    "lower": "Lower \u03b3 Threshold (brighter)",
    "upper": "Upper \u03b3 Threshold (darker)",
}

GAMMA_DIR_PATTERN = re.compile(r"^gamma_([0-9]*\.?[0-9]+)(?:_.*)?$")


def find_gamma_detections_dir(
    image_result_dir: Path,
    target_gamma: float,
) -> Tuple[Optional[Path], Optional[float]]:
    """
    Locate the ``detections/`` subdirectory of the RQ2.a gamma_* directory
    whose gamma value is numerically closest to ``target_gamma``.

    RQ2.a evaluates many gamma values per image and names each subdirectory
    ``gamma_<value>`` or ``gamma_<value>_<label>``; the exact directory used
    for the converged crossing gamma differs between the brute-force and
    golden-section scripts, so we match on the leading numeric value rather
    than a specific label (mirrors ``rq2b_brightness_calculation.py``).

    Returns:
        (detections_dir, gamma_diff) or (None, None) if not found.
    """
    if not image_result_dir.is_dir():
        return None, None

    best_dir = None
    best_diff = float("inf")

    for entry in image_result_dir.iterdir():
        if not entry.is_dir():
            continue
        match = GAMMA_DIR_PATTERN.match(entry.name)
        if not match:
            continue

        gamma_value = float(match.group(1))
        diff = abs(gamma_value - target_gamma)
        if diff >= best_diff:
            continue

        detections_dir = entry / "detections"
        if not detections_dir.is_dir():
            continue

        best_diff = diff
        best_dir = detections_dir

    return best_dir, (best_diff if best_dir is not None else None)


def calculate_ap(
    detections_dir: Path,
    gt_labels_dir: Path,
    image_name: str,
    img_width: int,
    img_height: int,
) -> float:
    """
    Compute per-image AP@[.50:.95] via COCO evaluation, restricted to this
    image's ground truth (matches the per-image evaluation approach used by
    the RQ2.a search scripts).

    Returns:
        ap
    """
    gt_label_file = Path(gt_labels_dir) / f"{Path(image_name).stem}.txt"
    if not gt_label_file.is_file():
        raise FileNotFoundError(
            f"Ground-truth label not found for {image_name}: {gt_label_file}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        det_json = os.path.join(tmpdir, "detections.json")
        gt_json = os.path.join(tmpdir, "ground_truth.json")

        format_detections(
            gt_label_file,
            gt_json,
            as_ground_truth=True,
            is_yolo_format=True,
            img_width=img_width,
            img_height=img_height,
        )
        format_detections(detections_dir, det_json, gt_coco_file=gt_json, as_ground_truth=False)

        try:
            stats = evaluate_coco(gt_json, det_json, iou_type="bbox")
            if stats is None:
                return 0.0
            # stats[0] = AP @ IoU=0.50:0.95
            return float(stats[0])
        except Exception as e:
            print(f"    Warning: COCO evaluation failed for {image_name}: {e}")
            return 0.0


def collect_model_performance(
    results_json_path: Path,
    gt_labels_dir: Path,
    base_images_dir: Path,
    num_images: Optional[int],
) -> List[Dict]:
    """
    For every image with both a lower and upper gamma crossing, compute
    AP at the original image, the lower threshold, and the upper threshold.
    """
    with open(results_json_path) as f:
        data = json.load(f)

    image_results = data.get("results", [])
    if num_images:
        image_results = image_results[:num_images]

    records = []
    for image_result in image_results:
        gamma_left = image_result.get("gamma_left")
        gamma_right = image_result.get("gamma_right")
        if gamma_left is None or gamma_right is None:
            continue

        image_name = image_result["image_name"]
        image_stem = Path(image_name).stem
        image_result_dir = results_json_path.parent / image_stem

        image_path = base_images_dir / image_name
        try:
            with Image.open(image_path) as im:
                img_width, img_height = im.size
        except Exception as e:
            print(f"  Warning: could not read image size for {image_name}: {e}")
            continue

        gamma_lower = min(gamma_left, gamma_right)
        gamma_upper = max(gamma_left, gamma_right)

        original_detections_dir = image_result_dir / "original_detections"
        if not original_detections_dir.is_dir():
            print(f"  Warning: no original detections for {image_name}, skipping")
            continue

        lower_dir, lower_diff = find_gamma_detections_dir(image_result_dir, gamma_lower)
        if lower_dir is None:
            print(f"  Warning: no detections found for lower gamma={gamma_lower:.3f} ({image_name}), skipping")
            continue

        upper_dir, upper_diff = find_gamma_detections_dir(image_result_dir, gamma_upper)
        if upper_dir is None:
            print(f"  Warning: no detections found for upper gamma={gamma_upper:.3f} ({image_name}), skipping")
            continue

        ap_original = calculate_ap(
            original_detections_dir, gt_labels_dir, image_name, img_width, img_height
        )
        ap_lower = calculate_ap(
            lower_dir, gt_labels_dir, image_name, img_width, img_height
        )
        ap_upper = calculate_ap(
            upper_dir, gt_labels_dir, image_name, img_width, img_height
        )

        records.append({
            "image_name": image_name,
            "gamma_lower": gamma_lower,
            "gamma_upper": gamma_upper,
            "lower_gamma_match_diff": lower_diff,
            "upper_gamma_match_diff": upper_diff,
            "ap_original": ap_original,
            "ap_lower": ap_lower,
            "ap_upper": ap_upper,
        })

    return records


def summarize_records(records: List[Dict]) -> Dict:
    """Compute mean/median/std/min/max of AP for each threshold."""
    summary = {}
    for threshold in THRESHOLDS:
        key = f"ap_{threshold}"
        values = [r[key] for r in records if r.get(key) is not None]
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        summary[key] = {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return summary


def plot_performance_distribution(
    model: str,
    records: List[Dict],
    output_path: Path,
    records_by_method: Optional[Dict[str, List[Dict]]] = None,
) -> List[Path]:
    """
    Save separate PDFs with one violin plot each (AP at
    Original/Lower/Upper), matching the violin styling used by
    ``plot_confidence_change.plot_lost_tp_rule``.

    Append _original, _lower, and _upper to output_path's stem and return
    the saved paths.

    When records_by_method is supplied, overlay its threshold distributions
    at the same position; the original-image panel still uses records.
    """
    if not records:
        return []

    color = get_model_color(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for threshold in THRESHOLDS:
        fig, axis = plt.subplots(figsize=(5, 5.5))
        key = f"ap_{threshold}"
        values = [r[key] for r in records if r.get(key) is not None]
        overlay = records_by_method is not None and threshold != "original"

        if not values and not overlay:
            axis.text(
                0.5, 0.5, "No observations",
                transform=axis.transAxes, ha="center", va="center", color="#666666",
            )

        distributions = [("", color, values)]
        if overlay:
            distributions = [
                (label, method_color, [r[key] for r in records_by_method[method]
                                       if r.get(key) is not None])
                for method, (label, method_color) in METHOD_STYLES.items()
            ]

        for label, distribution_color, distribution in distributions:
            if not distribution:
                continue
            violin_parts = axis.violinplot(
                distribution,
                positions=[0],
                widths=0.6,
                showmeans=False,
                showmedians=True,
            )
            for body in violin_parts["bodies"]:
                body.set_facecolor(distribution_color)
                body.set_edgecolor(distribution_color)
                body.set_alpha(0.35 if overlay else 0.55)
            for part_key in ("cmedians", "cbars", "cmins", "cmaxes"):
                violin_parts[part_key].set_color(distribution_color)
                if overlay and label == METHOD_STYLES["golden_section"][0]:
                    violin_parts[part_key].set_linestyle("--")

        axis.set_title(THRESHOLD_LABELS[threshold], fontsize=10)
        axis.set_xticks([])
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, alpha=0.3, linestyle=":", linewidth=0.8, axis="y")

        axis.set_ylabel(AP_LABEL, fontsize=11, fontweight="bold")

        fig.suptitle(
            f"{get_model_label(model)}: AP Distribution",
            fontsize=14,
            fontweight="bold",
        )
        if overlay:
            fig.legend(
                handles=[Patch(facecolor=c, edgecolor=c, alpha=0.35, label=label)
                         for label, c in METHOD_STYLES.values()],
                loc="lower center", ncol=2, frameon=False,
            )
        fig.tight_layout(rect=(0, 0.06 if overlay else 0, 1, 0.90))

        threshold_path = output_path.with_name(f"{output_path.stem}_{threshold}.pdf")
        fig.savefig(threshold_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved_paths.append(threshold_path)
    return saved_paths


def plot_overlays_from_summaries(
    summary_paths: List[Path], models: Optional[List[str]],
    preferred_method: str, output_dir: Path,
) -> int:
    """Overlay thresholds and use each unique image from either method for Original."""
    model_data = {}
    for path in summary_paths:
        with open(path) as f:
            summary = json.load(f)
        for model, data in summary.items():
            method = data.get("search_method")
            if method in SEARCH_METHODS:
                model_data.setdefault(model, {})[method] = data

    generated = 0
    for model in models if models is not None else model_data:
        methods = model_data.get(model, {})
        if any(not methods.get(method, {}).get("records") for method in SEARCH_METHODS):
            print(f"Skip {model}: overlay requires saved AP records from both search methods")
            continue
        if len({methods[method].get("dataset", "test") for method in SEARCH_METHODS}) != 1:
            print(f"Skip {model}: search methods used different datasets")
            continue
        # Original AP is independent of threshold-search success. Include the
        # union of images, counting shared images once and preferring the chosen
        # method's original AP when both summaries contain it.
        original_records = {}
        method_order = [m for m in SEARCH_METHODS if m != preferred_method] + [preferred_method]
        for method in method_order:
            for record in methods[method]["records"]:
                if record.get("ap_original") is not None:
                    original_records[record["image_name"]] = record
        output_path = output_dir / f"{model}_overlapping_performance_distribution.pdf"
        saved_paths = plot_performance_distribution(
            model, list(original_records.values()), output_path,
            records_by_method={method: methods[method]["records"] for method in SEARCH_METHODS},
        )
        for saved_path in saved_paths:
            print(f"Saved: {saved_path}")
            generated += 1
    return generated


def resolve_results_json(model: str, preferred_method: str) -> Tuple[Optional[Path], Optional[str]]:
    """Find an RQ2.a results.json for this model, preferring ``preferred_method``."""
    candidates = [preferred_method] + [m for m in SEARCH_METHODS if m != preferred_method]
    for method in candidates:
        candidate_path = RESULTS_ROOT / "rq2a" / method / model / "results.json"
        if candidate_path.is_file():
            return candidate_path, method
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="RQ2.c: Brightness-Threshold Performance Distribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All models with available RQ2.a results
  python scripts/rq2/rq2c_perf_comparison.py

  # Specific models, using brute-force thresholds
  python scripts/rq2/rq2c_perf_comparison.py --models yolov8 --search-method brute_force

  # Quick test
  python scripts/rq2/rq2c_perf_comparison.py --num-images 10
        """
    )
    parser.add_argument(
        "--overlay-from-summaries",
        type=Path,
        nargs=2,
        metavar=("SUMMARY_A", "SUMMARY_B"),
        help="Generate additional overlapping violins from saved RQ2.c summary JSONs "
             "for both methods, without recomputing AP. Original includes all unique "
             "images from either summary (shared AP values prefer --search-method); "
             "each threshold retains all observations from each method.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to include (default: all enabled models in config/models.json "
             "that have RQ2.a results available)",
    )
    parser.add_argument(
        "--search-method",
        type=str,
        choices=SEARCH_METHODS,
        default="golden_section",
        help="Preferred RQ2.a search method to source thresholds from "
             "(falls back to the other method if unavailable for a model). "
             "Default: golden_section",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Limit number of images processed per model (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: analysis/results/rq2/rq2c/performance_distributions)",
    )

    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else RESULTS_ROOT / "rq2c/performance_distributions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.overlay_from_summaries:
        if args.num_images is not None:
            parser.error("--num-images cannot be used with --overlay-from-summaries")
        generated = plot_overlays_from_summaries(
            args.overlay_from_summaries, args.models, args.search_method, output_dir
        )
        print(f"RQ2.c Complete: {generated} overlay plot(s) generated")
        if generated == 0:
            sys.exit(1)
        return

    models = args.models if args.models else get_model_names()

    print("="*80)
    print("RQ2.c: Brightness-Threshold Performance Distribution Configuration")
    print("="*80)
    print(f"Models requested: {models}")
    print(f"Preferred search method: {args.search_method}")
    print(f"Output: {output_dir}")
    print("="*80)

    all_model_data = {}
    generated = 0

    for model in models:
        results_json_path, chosen_method = resolve_results_json(model, args.search_method)
        if results_json_path is None:
            print(f"\nSkip {model}: no RQ2.a results found under {RESULTS_ROOT}")
            continue

        with open(results_json_path) as f:
            meta = json.load(f)
        dataset = meta.get("dataset", "test")
        gt_labels_dir = PROJECT_ROOT / "data" / "base_images" / dataset / "labels"
        base_images_dir = PROJECT_ROOT / "data" / "base_images" / dataset / "images"

        print(f"\n{'='*80}")
        print(f"Model: {model} (search method: {chosen_method})")
        print(f"{'='*80}")
        print(f"Results: {results_json_path}")
        print(f"Dataset: {dataset}")

        if not gt_labels_dir.is_dir() or not base_images_dir.is_dir():
            print(f"  Skip {model}: dataset paths not found ({gt_labels_dir}, {base_images_dir})")
            continue

        records = collect_model_performance(
            results_json_path, gt_labels_dir, base_images_dir, args.num_images
        )
        print(f"  Computed AP for {len(records)} image(s)")

        if not records:
            print(f"  Skip {model}: no complete image records")
            continue

        output_path = output_dir / f"{model}_{chosen_method}_performance_distribution.pdf"
        for saved_path in plot_performance_distribution(model, records, output_path):
            print(f"  Saved: {saved_path}")
            generated += 1

        all_model_data[model] = {
            "search_method": chosen_method,
            "results_json": str(results_json_path),
            "dataset": dataset,
            "num_images": len(records),
            "records": records,
            "summary": summarize_records(records),
        }

    summary_json = output_dir / f"results_{args.search_method}.json"
    with open(summary_json, "w") as f:
        json.dump(all_model_data, f, indent=2)
    print(f"\nSummary data saved to: {summary_json}")

    print(f"\n{'='*80}")
    print(f"RQ2.c Complete: {generated} plot(s) generated")
    print(f"{'='*80}")

    if generated == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
