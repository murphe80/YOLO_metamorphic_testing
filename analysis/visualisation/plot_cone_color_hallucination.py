#!/usr/bin/env python3
"""Plot four cone-colour rules from raw class/score/xyxy TXT detections.

Usage:
    python3 analysis/visualisation/plot_cone_color_hallucination.py
    python3 analysis/visualisation/plot_cone_color_hallucination.py --model yolov5

All percentages use the existing overlap metric's image weights: w_i = 1/S_i,
where S_i is the sum of the seven class-aware overlap regions for image i.
Each bar is 100 * sum(w_i * numerator_i) / sum(w_i * transformed_count_i).
Raw count labels are unweighted, so dividing them by the panel's detection
total will generally NOT reproduce the weighted percentage.

IoU accuracy means transformed-to-GT spatial precision, ignoring class, with
confidence-ordered one-to-one matching. It is not mean IoU or GT recall.
Class bars partition only new hallucinations (the existing trans_only region),
not all transformed FPs. Other/unknown classes are retained in the calculated results.
Each rule uses the same source-only cohort and prediction ownership policy as
imgColorWheel: other source objects are excluded; wrong-class and unowned
predictions remain error candidates.
Only available transformed TXT files form each rule's cohort; empty files are
valid, missing files are not silently interpreted as empty detections.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.model_registry import get_model_display_name

from analysis.utils.detection_matching import (
    match_detection_pairs,
    match_three_way_per_image,
)
from analysis.utils.format_detections import MODEL_TO_COCO_CATEGORY
from analysis.utils.color_wheel import cone_colour_mapping, overlap_cohort, grouped

# Canonical COCO IDs, not raw model IDs.
RULES = {
    "blue_to_orange": (2, 3),
    "yellow_to_blue": (1, 2),
    "yellow_to_orange": (1, 3),
    "blue_to_yellow": (2, 1),
}
BAR_CLASSES = (1, 2, 3, 4)
LABELS = (
    "New hallucination\nrate",
    "IoU accuracy\n(excluding class)",
    "New false\nyellow detections",
    "New false\nblue detections",
    "New false\norange detections",
    "New false\nbig orange\ndetections",
)


def load_raw(directory):
    """Read raw model output, preserving line order for matching tie breaks."""
    paths = sorted(directory.glob("*.txt"))
    if not paths:
        raise ValueError(f"No raw detection TXT files found in {directory}")
    result = {}
    for path in paths:
        detections = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                cls, score, x1, y1, x2, y2 = map(float, line.split())
                if not all(map(math.isfinite, (cls, score, x1, y1, x2, y2))):
                    raise ValueError("non-finite value")
                if cls != int(cls) or int(cls) not in MODEL_TO_COCO_CATEGORY:
                    raise ValueError(f"unsupported class {cls}")
                if not 0 <= score <= 1 or x2 <= x1 or y2 <= y1:
                    raise ValueError("invalid confidence or box")
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            detections.append({
                "category_id": MODEL_TO_COCO_CATEGORY[int(cls)],
                "score": score,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            })
        result[path.stem] = detections
    return result


def load_ground_truth(path):
    data = json.loads(path.read_text())
    names = {image["id"]: Path(image["file_name"]).stem for image in data["images"]}
    if len(set(names.values())) != len(names):
        raise ValueError(f"Duplicate image stems in {path}")
    result = {name: [] for name in names.values()}
    for annotation in data["annotations"]:
        result[names[annotation["image_id"]]].append(annotation)
    return result


def expected_classes(detections, source, target):
    return [dict(d, category_id=target if d["category_id"] == source
                 else d["category_id"]) for d in detections]


def new_detection_indices(original, transformed, gt, threshold):
    """Recover trans_only members with the repository's assignment precedence.

    Simply excluding all independent GT/original matches is not equivalent:
    a GT can already have been assigned to another overlap region.
    """
    og = dict(match_detection_pairs(original, gt, threshold, True))
    tg = dict(match_detection_pairs(transformed, gt, threshold, True))
    ot = dict(match_detection_pairs(original, transformed, threshold, True))
    assigned_gt = {og[o] for o, t in ot.items()
                   if o in og and t in tg and og[o] == tg[t]}
    # All original/transformed pairs are consumed by the first two regions.
    assigned_transformed = set(ot.values())
    for o, g in og.items():
        if o not in ot and g not in assigned_gt:
            assigned_gt.add(g)
    for t, g in tg.items():
        if t not in assigned_transformed and g not in assigned_gt:
            assigned_transformed.add(t)
            assigned_gt.add(g)
    return set(range(len(transformed))) - assigned_transformed


def calculate_rule(original, transformed, ground_truth, rule, threshold=0.5):
    mapping = cone_colour_mapping(rule)
    def flatten(by_name):
        return {"annotations": [dict(d, image_id=name)
                                for name, ds in by_name.items() for d in ds]}
    expected_original, selected, expected_gt = overlap_cohort(
        flatten(original), flatten(transformed), flatten(ground_truth), mapping
    )
    def with_empty(data, names):
        by_name = grouped(data)
        return {name: by_name.get(name, []) for name in names}
    result = calculate_cohort(
        with_empty(expected_original, original), with_empty(selected, transformed),
        with_empty(expected_gt, ground_truth), rule, threshold
    )
    result.update(cohort="source_only", mapping=mapping,
                  title=f"{mapping['mapping'].capitalize()} ({mapping['source']}-source cohort)")
    return result


def calculate_cohort(original, transformed, ground_truth, rule, threshold=0.5):
    """Calculate bars from an already selected cohort in expected-class space."""
    missing = set(transformed) - (set(original) & set(ground_truth))
    if missing:
        raise ValueError(f"{rule}: missing original detections or GT for {sorted(missing)}")
    weighted = [0.0] * len(LABELS)
    counts = [0] * len(LABELS)
    denominator = 0.0
    total_detections = 0
    other_count = 0
    other_weight = 0.0
    for name, trans in transformed.items():
        orig = original[name]
        gt = ground_truth[name]
        regions = match_three_way_per_image(orig, trans, gt, threshold, True)
        new = new_detection_indices(orig, trans, gt, threshold)
        if len(new) != regions["trans_only"]:
            raise AssertionError(f"{rule}/{name}: trans_only reconstruction disagrees")
        spatial = match_detection_pairs(trans, gt, threshold, False)
        classes = Counter(trans[index]["category_id"] for index in new)
        numerators = [len(new), len(spatial), *(classes[c] for c in BAR_CLASSES)]
        image_total = sum(regions.values())
        weight = 1 / image_total if image_total else 0.0
        for index, value in enumerate(numerators):
            counts[index] += value
            weighted[index] += value * weight
        other = sum(value for cls, value in classes.items() if cls not in BAR_CLASSES)
        other_count += other
        other_weight += other * weight
        denominator += len(trans) * weight
        total_detections += len(trans)
    if not total_detections:
        raise ValueError(f"{rule}: no saved detections; rates are undefined")
    percentages = [100 * value / denominator for value in weighted]
    other_percentage = 100 * other_weight / denominator
    if counts[0] != sum(counts[2:]) + other_count:
        raise AssertionError("Class counts do not partition new hallucinations")
    if not math.isclose(percentages[0], sum(percentages[2:]) + other_percentage,
                        abs_tol=1e-9):
        raise AssertionError("Class rates do not partition new hallucinations")
    return dict(rule=rule, percentages=percentages, counts=counts,
                images=len(transformed), detections=total_detections,
                other_count=other_count, other_percentage=other_percentage,
                absent_images=len(set(ground_truth) - set(transformed)))


def plot(results, model, threshold, output, rule_family="cone colour rules"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = ["#7950A0", "#278779", "#E4BD35", "#397AC3", "#EF963C", "#AD541A"]
    rows = max(1, math.ceil(len(results) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(18, 5.5 * rows), sharey=True, squeeze=False)
    for ax, result in zip(axes.flat, results):
        bars = ax.bar(range(len(LABELS)), result["percentages"], color=colors, width=.68)
        for bar, percentage, count in zip(bars, result["percentages"], result["counts"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.4,
                    f"{percentage:.2f}%\nn = {count:,}", ha="center", va="bottom",
                    fontsize=10)
        title = result.get("title", result["rule"].replace("_to_", " → ").capitalize())
        ax.set_title(f"{title}\n{result['images']:,} images · "
                     f"{result['detections']:,} transformed detections", fontsize=13, pad=12)
        ax.set_xticks(range(len(LABELS)), LABELS, fontsize=9)
        ax.set_ylim(0, 112)
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_ylabel("Image-normalized percentage")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(results):]:
        ax.set_visible(False)
    fig.suptitle(f"{get_model_display_name(model)} : {rule_family}: new false positives and spatial accuracy",
                 fontsize=19, y=.98)
    fig.subplots_adjust(left=.065, right=.985, bottom=.09, top=.88, hspace=.38, wspace=.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolov26_v2")
    parser.add_argument("--dataset", default="test")
    parser.add_argument("--results-dir", type=Path,
                        help="Dataset results directory containing model folders")
    parser.add_argument("--gt-path", type=Path)
    parser.add_argument("--output", type=Path, help="Output PDF path (overwrites if present)")
    parser.add_argument("--iou-threshold", type=float, default=.5)
    args = parser.parse_args()
    if not 0 <= args.iou_threshold < 1:
        parser.error("--iou-threshold must be in [0, 1)")
    results_dir = args.results_dir or PROJECT_ROOT / "results" / args.dataset
    gt_path = args.gt_path or PROJECT_ROOT / "data/base_images" / args.dataset / "annotations_coco.json"
    output = args.output or (PROJECT_ROOT / "analysis/results/rq1" / args.dataset /
                            "detection_overlap_metrics/plots/cone_color_hallucination.pdf")
    try:
        original = load_raw(results_dir / args.model / "original")
        gt = load_ground_truth(gt_path)
        results = [calculate_rule(original, load_raw(results_dir / args.model / rule),
                                  gt, rule, args.iou_threshold) for rule in RULES]
        plot(results, args.model, args.iou_threshold, output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
    for result in results:
        print(f"{result['rule']}: " + ", ".join(
            f"{label.replace(chr(10), ' ')}={rate:.2f}% (n={count})"
            for label, rate, count in zip(LABELS, result["percentages"], result["counts"])))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
