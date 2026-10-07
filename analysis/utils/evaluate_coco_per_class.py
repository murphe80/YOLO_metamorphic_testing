#!/usr/bin/env python3
"""Evaluate COCO detection metrics overall and for each object class."""

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco_per_class(gt_file, dt_file, iou_type="bbox"):
    """Return overall and per-class COCO AP/AR metrics."""
    print(f"Loading ground truth from: {gt_file}")
    coco_gt = COCO(gt_file)

    print(f"Loading detection results from: {dt_file}")
    coco_dt = coco_gt.loadRes(dt_file)

    print(f"\nInitializing COCO evaluation (IOU type: {iou_type})")
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    image_ids = sorted(coco_gt.getImgIds())
    print(f"Evaluating on all {len(image_ids)} images")

    coco_eval.evaluate()
    coco_eval.accumulate()

    print("\n=== Overall Evaluation Metrics ===")
    coco_eval.summarize()

    stats = coco_eval.stats
    keys = [
        "AP", "AP50", "AP75", "APs", "APm", "APl",
        "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
    ]
    overall_stats = dict(zip(keys, [float(value) for value in stats]))

    print(f"\nAP @ IoU=0.50:0.95: {stats[0]:.4f}")
    print(f"AP @ IoU=0.50:      {stats[1]:.4f}")
    print(f"AP @ IoU=0.75:      {stats[2]:.4f}")
    print(f"AR @ IoU=0.50:0.95: {stats[8]:.4f}")

    category_ids = sorted(coco_gt.getCatIds())
    category_info = {
        category["id"]: category
        for category in coco_gt.loadCats(category_ids)
    }
    precision = coco_eval.eval["precision"]
    recall = coco_eval.eval["recall"]
    per_class_stats = {}

    print("\n=== Per-Class Evaluation Metrics ===")
    print(
        f"{'Class':<30} {'CatID':<8} {'AP@.5:.95':<12} "
        f"{'AP@.50':<10} {'AP@.75':<10} {'AR@100':<10}"
    )
    print("-" * 80)

    for category_index, category_id in enumerate(category_ids):
        category_name = category_info.get(
            category_id, {"name": f"category_{category_id}"}
        )["name"]

        precision_all = precision[:, :, category_index, 0, 2]
        valid_precision = precision_all[precision_all > -1]
        ap_all = (
            float(np.mean(valid_precision))
            if valid_precision.size
            else float("nan")
        )

        precision_50 = precision[0, :, category_index, 0, 2]
        valid_precision_50 = precision_50[precision_50 > -1]
        ap_50 = (
            float(np.mean(valid_precision_50))
            if valid_precision_50.size
            else float("nan")
        )

        precision_75 = precision[5, :, category_index, 0, 2]
        valid_precision_75 = precision_75[precision_75 > -1]
        ap_75 = (
            float(np.mean(valid_precision_75))
            if valid_precision_75.size
            else float("nan")
        )

        recall_all = recall[:, category_index, 0, 2]
        valid_recall = recall_all[recall_all > -1]
        ar_all = (
            float(np.mean(valid_recall))
            if valid_recall.size
            else float("nan")
        )

        per_class_stats[category_id] = {
            "name": category_name,
            "AP": ap_all,
            "AP50": ap_50,
            "AP75": ap_75,
            "AR100": ar_all,
        }
        print(
            f"{category_name:<30} {category_id:<8} {ap_all:<12.4f} "
            f"{ap_50:<10.4f} {ap_75:<10.4f} {ar_all:<10.4f}"
        )

    return overall_stats, per_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate COCO detection results overall and per class."
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="Ground-truth annotations in COCO JSON format.",
    )
    parser.add_argument(
        "--dt",
        required=True,
        help="Detection results in COCO JSON format.",
    )
    parser.add_argument(
        "--iou-type",
        "--iou_type",
        dest="iou_type",
        default="bbox",
        choices=["bbox", "segm", "keypoints"],
        help="Evaluation type (default: bbox).",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        metavar="PATH",
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    overall_stats, per_class_stats = evaluate_coco_per_class(
        args.gt, args.dt, args.iou_type
    )

    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "overall": overall_stats,
            "per_class": per_class_stats,
        }
        with args.stats_out.open("w") as handle:
            json.dump(output, handle, indent=2)
            handle.write("\n")
        print(f"\nStats saved to: {args.stats_out}")


if __name__ == "__main__":
    main()
