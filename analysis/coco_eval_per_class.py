"""
Takes formatted ground truth and detection data and evaluates the performance of the detection model.
Output is AP (Average Precision) and AR (Average Recall) metrics using COCO evaluation,
broken down per class (category_id).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco_per_class(gt_file, dt_file, iou_type='bbox'):
    """
    Evaluate detections against ground truth using COCO metrics, broken down per class.

    Args:
        gt_file:  Path to ground truth annotations file (COCO format JSON)
        dt_file:  Path to detection results file (COCO format JSON)
        iou_type: Type of evaluation ('bbox', 'segm', or 'keypoints')

    Returns:
        overall_stats: dict of overall AP/AR metrics (same as original evaluate_coco)
        per_class_stats: dict mapping category_id -> dict of per-class metrics
    """
    print(f"Loading ground truth from: {gt_file}")
    cocoGt = COCO(gt_file)

    print(f"Loading detection results from: {dt_file}")
    cocoDt = cocoGt.loadRes(dt_file)

    # ------------------------------------------------------------------ #
    # Overall evaluation (unchanged from original)
    # ------------------------------------------------------------------ #
    print(f"\nInitializing COCO evaluation (IOU type: {iou_type})")
    cocoEval = COCOeval(cocoGt, cocoDt, iou_type)

    imgIds = sorted(cocoGt.getImgIds())
    print(f"Evaluating on all {len(imgIds)} images")

    cocoEval.evaluate()
    cocoEval.accumulate()

    print("\n=== Overall Evaluation Metrics ===")
    cocoEval.summarize()

    stats = cocoEval.stats
    keys = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
            'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
    overall_stats = dict(zip(keys, [float(x) for x in stats]))

    print(f"\nAP @ IoU=0.50:0.95: {stats[0]:.4f}")
    print(f"AP @ IoU=0.50:      {stats[1]:.4f}")
    print(f"AP @ IoU=0.75:      {stats[2]:.4f}")
    print(f"AR @ IoU=0.50:0.95: {stats[8]:.4f}")

    # ------------------------------------------------------------------ #
    # Per-class evaluation
    # COCOeval stores precision/recall in cocoEval.eval after accumulate():
    #   precision shape: [T, R, K, A, M]
    #     T = IoU thresholds (10: 0.5:0.05:0.95)
    #     R = recall thresholds (101: 0:0.01:1)
    #     K = categories
    #     A = area ranges (4: all, small, medium, large)
    #     M = max detections (3: 1, 10, 100)
    #   recall shape: [T, K, A, M]
    # ------------------------------------------------------------------ #
    cat_ids = sorted(cocoGt.getCatIds())
    cat_info = {cat['id']: cat for cat in cocoGt.loadCats(cat_ids)}

    precision = cocoEval.eval['precision']  # [T, R, K, A, M]
    recall    = cocoEval.eval['recall']     # [T, K, A, M]

    per_class_stats = {}

    print("\n=== Per-Class Evaluation Metrics ===")
    print(f"{'Class':<30} {'CatID':<8} {'AP@.5:.95':<12} {'AP@.50':<10} {'AP@.75':<10} {'AR@100':<10}")
    print("-" * 80)

    for k_idx, cat_id in enumerate(cat_ids):
        cat_name = cat_info[cat_id]['name'] if cat_id in cat_info else f"category_{cat_id}"

        # AP @ IoU 0.50:0.95  — average over all T thresholds, area=all (A=0), maxDet=100 (M=2)
        prec_all = precision[:, :, k_idx, 0, 2]        # [T, R]
        valid    = prec_all[prec_all > -1]
        ap_all   = float(np.mean(valid)) if valid.size else float('nan')

        # AP @ IoU 0.50       — T=0, area=all, maxDet=100
        prec_50  = precision[0, :, k_idx, 0, 2]
        valid_50 = prec_50[prec_50 > -1]
        ap_50    = float(np.mean(valid_50)) if valid_50.size else float('nan')

        # AP @ IoU 0.75       — T=5, area=all, maxDet=100
        prec_75  = precision[5, :, k_idx, 0, 2]
        valid_75 = prec_75[prec_75 > -1]
        ap_75    = float(np.mean(valid_75)) if valid_75.size else float('nan')

        # AR @ IoU 0.50:0.95  — average over all T thresholds, area=all, maxDet=100 (M=2)
        rec_all  = recall[:, k_idx, 0, 2]              # [T]
        valid_r  = rec_all[rec_all > -1]
        ar_all   = float(np.mean(valid_r)) if valid_r.size else float('nan')

        per_class_stats[cat_id] = {
            'name':    cat_name,
            'AP':      ap_all,
            'AP50':    ap_50,
            'AP75':    ap_75,
            'AR100':   ar_all,
        }

        print(f"{cat_name:<30} {cat_id:<8} {ap_all:<12.4f} {ap_50:<10.4f} {ap_75:<10.4f} {ar_all:<10.4f}")

    return overall_stats, per_class_stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate object detection results using COCO metrics (overall + per class)"
    )
    parser.add_argument('--gt',       type=str, required=True,
                        help='Path to ground truth annotations file (COCO format JSON)')
    parser.add_argument('--dt',       type=str, required=True,
                        help='Path to detection results file (COCO format JSON)')
    parser.add_argument('--iou_type', type=str, default='bbox',
                        choices=['bbox', 'segm', 'keypoints'],
                        help='Type of evaluation (default: bbox)')
    parser.add_argument('--stats-out', type=str, default=None, metavar='PATH',
                        help='Path to save evaluation stats as JSON.')

    args = parser.parse_args()

    overall_stats, per_class_stats = evaluate_coco_per_class(args.gt, args.dt, args.iou_type)

    if args.stats_out:
        stats_path = Path(args.stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            'overall':   overall_stats,
            'per_class': per_class_stats,
        }

        with open(stats_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nStats saved to: {stats_path}")


if __name__ == "__main__":
    main()