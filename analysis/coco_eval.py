"""
Takes formatted ground truth and detection data and evaluates the performance of the detection model.
Output is AP (Average Precision) and AR (Average Recall) metrics using COCO evaluation.
"""

import argparse
import json
import tempfile
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _allowed_filenames_from_dt(dt_data, gt_id_to_file_name):
    """
    Collect file names for images covered by the detection file.
    Uses `images[].file_name` when present, and maps `image_id` from detection
    annotations through gt_id_to_file_name for list-only result files.
    """
    allowed = set()
    if isinstance(dt_data, dict):
        for img in dt_data.get("images") or []:
            fn = img.get("file_name")
            if fn:
                allowed.add(fn)
        anns = dt_data.get("annotations") or []
    elif isinstance(dt_data, list):
        anns = dt_data
    else:
        anns = []

    for ann in anns:
        if not isinstance(ann, dict):
            continue
        iid = ann.get("image_id")
        if iid is not None and iid in gt_id_to_file_name:
            allowed.add(gt_id_to_file_name[iid])
    return allowed


def _filter_gt_by_filenames(gt_data, allowed_file_names):
    """Keep only images (and their annotations) whose file_name is in allowed_file_names."""
    kept_images = [im for im in gt_data["images"] if im["file_name"] in allowed_file_names]
    kept_ids = {im["id"] for im in kept_images}
    kept_ann = [a for a in gt_data["annotations"] if a["image_id"] in kept_ids]
    out = {
        k: v
        for k, v in gt_data.items()
        if k not in ("images", "annotations", "categories")
    }
    out["images"] = kept_images
    out["annotations"] = kept_ann
    out["categories"] = gt_data.get("categories", [])
    return out


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
    with open(gt_file) as f:
        gt_data = json.load(f)

    with open(dt_file) as f:
        dt_data = json.load(f)

    gt_id_to_fn = {img["id"]: img["file_name"] for img in gt_data["images"]}
    allowed = _allowed_filenames_from_dt(dt_data, gt_id_to_fn)

    if allowed:
        n_before = len(gt_data["images"])
        gt_data = _filter_gt_by_filenames(gt_data, allowed)
        print(
            f"Filtered ground truth to {len(gt_data['images'])}/{n_before} images "
            "present in detection file (by file name)."
        )
    else:
        print(
            "Warning: No image file names inferred from detection file; "
            "using full ground truth."
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(gt_data, tmp)
        tmp_gt_path = tmp.name

    try:
        cocoGt = COCO(tmp_gt_path)
    finally:
        Path(tmp_gt_path).unlink(missing_ok=True)

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
