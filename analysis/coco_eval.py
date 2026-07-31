"""
Takes formatted ground truth and detection data and evaluates the performance of the detection model.
Output is AP (Average Precision) and AR (Average Recall) metrics using COCO evaluation.
"""

import argparse
import json
import tempfile
import time
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


def _allowed_filenames_from_image_dir(image_dir):
    """Collect image file names from an explicitly supplied evaluation directory."""
    image_path = Path(image_dir)
    supported_extensions = {".jpg", ".jpeg", ".png"}
    return {
        path.name
        for path in image_path.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    }


def evaluate_coco(gt_file, dt_file, iou_type='bbox', image_dir=None):
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
    if image_dir:
        allowed = _allowed_filenames_from_image_dir(image_dir)
        filter_source = f"image directory {image_dir}"
    else:
        allowed = _allowed_filenames_from_dt(dt_data, gt_id_to_fn)
        filter_source = "detection file"

    # region agent log
    _dt_annotations_debug = dt_data.get("annotations", []) if isinstance(dt_data, dict) else (dt_data if isinstance(dt_data, list) else [])
    with open("/moredata/emma/metamorphic_v2/YOLO_metamorphic_testing/.cursor/debug-3f141a.log", "a") as _debug_file:
        _debug_file.write(json.dumps({"sessionId": "3f141a", "runId": "pre-fix", "hypothesisId": "H1,H3,H4", "location": "analysis/coco_eval.py:97", "message": "Loaded COCO evaluation inputs", "data": {"gtFile": str(gt_file), "dtFile": str(dt_file), "gtImages": len(gt_data.get("images", [])), "gtAnnotations": len(gt_data.get("annotations", [])), "dtAnnotations": len(_dt_annotations_debug), "dtUniqueImageIds": len({a.get("image_id") for a in _dt_annotations_debug if isinstance(a, dict)}), "allowedImages": len(allowed), "gtCategoryIds": sorted({c.get("id") for c in gt_data.get("categories", [])}), "dtCategoryIds": sorted({a.get("category_id") for a in _dt_annotations_debug if isinstance(a, dict)})}, "timestamp": int(time.time() * 1000)}) + "\n")
    # endregion

    if allowed:
        n_before = len(gt_data["images"])
        gt_data = _filter_gt_by_filenames(gt_data, allowed)
        print(
            f"Filtered ground truth to {len(gt_data['images'])}/{n_before} images "
            f"present in {filter_source} (by file name)."
        )
        if not gt_data["images"]:
            raise ValueError(
                f"No ground-truth image names match the images in {filter_source}."
            )
    else:
        print(
            f"Warning: No image file names inferred from {filter_source}; "
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
    if isinstance(dt_data, dict):
        dt_annotations = dt_data.get("annotations", [])
    else:
        dt_annotations = dt_data
    valid_image_ids = {image["id"] for image in gt_data["images"]}
    dt_annotations = [
        annotation
        for annotation in dt_annotations
        if annotation.get("image_id") in valid_image_ids
    ]

    # region agent log
    with open("/moredata/emma/metamorphic_v2/YOLO_metamorphic_testing/.cursor/debug-3f141a.log", "a") as _debug_file:
        _debug_file.write(json.dumps({"sessionId": "3f141a", "runId": "pre-fix", "hypothesisId": "H1,H3", "location": "analysis/coco_eval.py:145", "message": "Prepared filtered COCO evaluation set", "data": {"keptGtImages": len(gt_data.get("images", [])), "keptGtAnnotations": len(gt_data.get("annotations", [])), "keptDtAnnotations": len(dt_annotations), "keptDtUniqueImageIds": len({a.get("image_id") for a in dt_annotations}), "excludedGtImages": len(gt_id_to_fn) - len(gt_data.get("images", [])), "excludedDtAnnotations": len(_dt_annotations_debug) - len(dt_annotations)}, "timestamp": int(time.time() * 1000)}) + "\n")
    # endregion

    if dt_annotations:
        cocoDt = cocoGt.loadRes(dt_annotations)
    else:
        # pycocotools.loadRes assumes at least one annotation. Build an empty
        # result set explicitly so images with zero detections are still scored.
        cocoDt = COCO()
        cocoDt.dataset = {
            "images": gt_data["images"],
            "annotations": [],
            "categories": gt_data.get("categories", []),
        }
        cocoDt.createIndex()
    
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
    parser.add_argument(
        '--image-dir',
        type=str,
        default=None,
        help=(
            'Optional directory defining the exact image subset to evaluate. '
            'This includes images whose detection files are empty.'
        )
    )

    args = parser.parse_args()

    # Run evaluation
    stats = evaluate_coco(args.gt, args.dt, args.iou_type, args.image_dir)

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
