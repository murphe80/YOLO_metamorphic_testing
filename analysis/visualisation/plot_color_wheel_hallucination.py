#!/usr/bin/env python3
"""Plot source-specific colour-wheel hallucinations in the cone-colour bar style.

Uses the reviewed colour-wheel ownership/exclusion policy and the same
image-normalized percentages, raw counts and class-independent spatial
precision as plot_cone_color_hallucination.py. Unowned predictions remain FP
candidates in each cohort. Orange → blue at hue 80 and orange → yellow at
hue 20 are deliberately excluded.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils.color_wheel import metadata, select_predictions, grouped
from analysis.visualisation.plot_cone_color_hallucination import (
    LABELS, load_raw, load_ground_truth, calculate_cohort, plot,
)

MAPPINGS = (
    ("imgColorWheel_4", "yellow_to_blue"),
    ("imgColorWheel_8", "yellow_to_orange"),
    ("imgColorWheel_6", "blue_to_yellow"),
    ("imgColorWheel_4", "blue_to_orange"),
)


def flatten(by_name):
    return [dict(d, image_id=name) for name, ds in by_name.items() for d in ds]


def calculate_mapping(original, transformed, ground_truth, variant, key, threshold=.5):
    record = metadata(variant)
    mapping = next((m for m in record["included_mappings"] if m["key"] == key), None)
    if mapping is None:
        raise ValueError(f"{variant}/{key} is not an included colour-wheel mapping")
    missing = set(transformed) - (set(original) & set(ground_truth))
    if missing:
        raise ValueError(f"{variant}: missing original detections or GT for {sorted(missing)}")
    orig, gt, trans = map(flatten, (original, ground_truth, transformed))
    selected_orig = select_predictions(orig, gt, mapping, transformed=False)
    expected_orig = [dict(d, category_id=mapping["target_category_id"]) for d in selected_orig]
    expected_gt = [dict(d, category_id=mapping["target_category_id"])
                   for d in gt if d["category_id"] == mapping["source_category_id"]]
    selected_trans = select_predictions(trans, gt, mapping, fallback=orig,
                                        include_wrong_classes=True)
    # Retain empty images: missing files and empty source cohorts are different.
    def with_empty(ds, names):
        by_name = grouped(ds)
        return {name: by_name.get(name, []) for name in names}
    result = calculate_cohort(with_empty(expected_orig, original),
                              with_empty(selected_trans, transformed),
                              with_empty(expected_gt, ground_truth), key, threshold)
    result.update(variant_id=variant, hue_rotation=record["hue_rotation"], mapping=mapping,
                  title=f"{mapping['mapping'].capitalize()} : {variant} (hue {record['hue_rotation']})")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov26_v2")
    parser.add_argument("--dataset", default="test")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--gt-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=.5)
    args = parser.parse_args()
    if not 0 <= args.iou_threshold < 1:
        parser.error("--iou-threshold must be in [0, 1)")
    base = (args.results_dir or PROJECT_ROOT / "results" / args.dataset) / args.model
    gt_path = args.gt_path or PROJECT_ROOT / "data/base_images" / args.dataset / "annotations_coco.json"
    output = args.output or (PROJECT_ROOT / "analysis/results/rq1" / args.dataset /
                            "detection_overlap_metrics/plots/color_wheel_hallucination.pdf")
    try:
        original = load_raw(base / "original")
        gt = load_ground_truth(gt_path)
        results = []
        for variant, key in MAPPINGS:
            raw = load_raw(base / variant)
            transformed = {name.removesuffix("_imgColorWheel"): ds for name, ds in raw.items()}
            if len(transformed) != len(raw):
                raise ValueError(f"Duplicate normalized image stems in {variant}")
            results.append(calculate_mapping(original, transformed, gt, variant, key, args.iou_threshold))
        plot(results, args.model, args.iou_threshold, output, "image colour wheel rules")
        output.with_suffix(".json").write_text(json.dumps({
            "model": args.model, "dataset": args.dataset, "iou_threshold": args.iou_threshold,
            "labels": LABELS, "weighting": "Per-image 1 / sum(seven overlap regions); denominator is weighted transformed detections.",
            "spatial_accuracy": "Class-independent one-to-one transformed-to-GT precision, not recall or mean IoU.",
            "evaluation_policy": "Source cohorts use GT ownership then original-prediction fallback at IoU >= 0.5. Unowned predictions are included in each cohort.",
            "results": results,
        }, indent=2) + "\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
    for result in results:
        print(f"{result['variant_id']}/{result['rule']}: new hallucination "
              f"{result['percentages'][0]:.2f}%, spatial precision {result['percentages'][1]:.2f}%")
    print(f"Saved {output} and {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
