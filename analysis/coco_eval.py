"""
Takes formatted ground truth and detection data and evaluates the performance of the detection model.
Output is AP (Average Precision) and AR (Average Recall) metrics using COCO evaluation.
"""

import argparse
import json
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _image_key(file_name):
    """Return the dataset identity shared by image files and COCO entries."""
    return Path(file_name).stem


def _filter_gt_by_image_keys(gt_data, allowed_image_keys):
    """Keep only ground-truth images explicitly selected by dataset identity."""
    kept_images = [
        image
        for image in gt_data["images"]
        if _image_key(image["file_name"]) in allowed_image_keys
    ]
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


def _allowed_image_keys_from_image_dir(image_dir):
    """Collect dataset image identities from an explicit evaluation directory."""
    image_path = Path(image_dir)
    if not image_path.is_dir():
        raise ValueError(f"Evaluation image directory does not exist: {image_dir}")

    supported_extensions = {".jpg", ".jpeg", ".png"}
    image_keys = {
        _image_key(path.name)
        for path in image_path.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    }
    if not image_keys:
        raise ValueError(f"Evaluation image directory contains no images: {image_dir}")
    return image_keys


def evaluate_coco(gt_file, dt_file, iou_type='bbox', image_dir=None):
    """
    Evaluate detections against ground truth using COCO metrics.
    
    Args:
        gt_file: Path to ground truth annotations file (COCO format JSON)
        dt_file: Path to detection results file (COCO format JSON)
        iou_type: Type of evaluation ('bbox', 'segm', or 'keypoints')
        image_dir: Optional directory defining an explicit evaluation subset.
            Without it, every image declared by the ground-truth COCO file is
            evaluated, including images absent from detection annotations.
    
    Returns:
        stats: Array of evaluation metrics
    """
    print(f"Loading ground truth from: {gt_file}")
    with open(gt_file) as f:
        gt_data = json.load(f)

    with open(dt_file) as f:
        dt_data = json.load(f)

    if image_dir:
        allowed = _allowed_image_keys_from_image_dir(image_dir)
        declared = {_image_key(image["file_name"]) for image in gt_data["images"]}
        missing = allowed - declared
        if missing:
            examples = ", ".join(sorted(missing)[:5])
            raise ValueError(
                f"{len(missing)} image(s) from {image_dir} are not declared in "
                f"the ground-truth COCO file (for example: {examples})."
            )

        n_before = len(gt_data["images"])
        gt_data = _filter_gt_by_image_keys(gt_data, allowed)
        print(
            f"Selected {len(gt_data['images'])}/{n_before} ground-truth images "
            f"from image directory {image_dir}."
        )
    else:
        print(
            f"Using all {len(gt_data['images'])} images declared by the "
            "ground-truth COCO file."
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
    elif isinstance(dt_data, list):
        dt_annotations = dt_data
    else:
        raise ValueError("Detection JSON must be a COCO result list or object.")
    valid_image_ids = {image["id"] for image in gt_data["images"]}
    n_before = len(dt_annotations)
    dt_annotations = [
        annotation
        for annotation in dt_annotations
        if annotation.get("image_id") in valid_image_ids
    ]
    if len(dt_annotations) != n_before:
        print(
            f"Ignored {n_before - len(dt_annotations)} detection annotation(s) "
            "outside the evaluation image universe."
        )

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
    cocoEval.params.imgIds = imgIds
    
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

    parser.add_argument("--color-wheel-variant", help="Reviewed hue variant; evaluate each source class separately")
    args = parser.parse_args()

    if args.color_wheel_variant:
        from analysis.utils.color_wheel import evaluate_files
        result = evaluate_files(args.gt, args.dt, args.color_wheel_variant, args.image_dir)
        if args.stats_out:
            path = Path(args.stats_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
        else:
            print(json.dumps(result, indent=2))
        return

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
