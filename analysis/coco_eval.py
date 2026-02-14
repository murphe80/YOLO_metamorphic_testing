"""
Takes formatted ground truth and detection data and evaluates the performance of the detection model.
Output is AP (Average Precision) and AR (Average Recall) metrics using COCO evaluation.
"""

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco(gt_file, dt_file, iou_type='bbox'):
    """
    Evaluate detections against ground truth using COCO metrics.
    
    Args:
        gt_file: Path to ground truth annotations file (COCO format JSON)
        dt_file: Path to detection results file (COCO format JSON)
        iou_type: Type of evaluation ('bbox', 'segm', or 'keypoints')
    
    Returns:
        stats: Array of evaluation metrics
    """
    print(f"Loading ground truth from: {gt_file}")
    cocoGt = COCO(gt_file)
    
    print(f"Loading detection results from: {dt_file}")
    cocoDt = cocoGt.loadRes(dt_file)
    
    # Initialize COCOeval
    print(f"\nInitializing COCO evaluation (IOU type: {iou_type})")
    cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
    

    imgIds = sorted(cocoGt.getImgIds())
    print(f"Evaluating on all {len(imgIds)} images")
    
    # Run evaluation
    print("\nRunning evaluation...")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    # Access evaluation results (COCO: AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl)
    stats = cocoEval.stats

    print("\n=== Evaluation Metrics ===")
    print(f"AP @ IoU=0.50:0.95: {stats[0]:.4f}")
    print(f"AP @ IoU=0.50:      {stats[1]:.4f}")
    print(f"AP @ IoU=0.75:      {stats[2]:.4f}")
    print(f"AR @ IoU=0.50:0.95: {stats[8]:.4f}")

    return stats


def main():
    """Main function to run COCO evaluation from command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate object detection results using COCO metrics"
    )
    parser.add_argument(
        '--gt',
        type=str,
        required=True,
        help='Path to ground truth annotations file (COCO format JSON)'
    )
    parser.add_argument(
        '--dt',
        type=str,
        required=True,
        help='Path to detection results file (COCO format JSON)'
    )
    parser.add_argument(
        '--iou_type',
        type=str,
        default='bbox',
        choices=['bbox', 'segm', 'keypoints'],
        help='Type of evaluation (default: bbox)'
    )
    parser.add_argument(
        '--stats-out',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to save evaluation stats (JSON). If not set, stats are only printed.'
    )

    args = parser.parse_args()

    # Run evaluation
    stats = evaluate_coco(args.gt, args.dt, args.iou_type)

    if args.stats_out:
        stats_path = Path(args.stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        # COCO stats: AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl
        keys = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
        stats_dict = dict(zip(keys, [float(x) for x in stats]))
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"\nStats saved to: {stats_path}")


if __name__ == "__main__":
    main()
