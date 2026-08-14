#!/usr/bin/env python3
"""
RQ2.a: Brute Force Gamma Search

Linearly sweeps through gamma values to find operational brightness thresholds
where model AP crosses the target threshold.

Finds both left (darker) and right (brighter) crossing points.

Usage:
    python scripts/rq2a_brute_force_search.py --model yolov8 --target-ap 0.7
    python scripts/rq2a_brute_force_search.py --model yolov8 --num-images 10
"""

import os
import sys
import json
import time
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.matias_metamorphic import gen_metamorphic
from analysis.utils.format_detections import format_detections
from analysis.coco_eval import evaluate_coco

# Configuration
GAMMA_RANGE = (0.01, 20.0)
GAMMA_STEP = 0.1
AP_TOLERANCE = 0.05
MAX_ROOT_REFINEMENT_ITERATIONS = 35
BRIGHTNESS_CONSTANT = 0
BRIGHTNESS_MULTIPLIER = 1.0
METHOD = "gamma"

# Model weights paths
MODEL_WEIGHTS = {
    "yolov5": "models/yolov5/yolov5_param_kfold.pt",
    "yolov8": "models/yolov8/yolov8_param_kfold.pt",
    "yolov12": "models/yolov12/yolov12_param_kfold.pt",
}

# Class names and colors for visualization
CLASS_NAMES = {
    0: "yellow",
    1: "orange",
    2: "large_orange",
    3: "blue",
    4: "other",
    5: "unknown"
}

CLASS_COLORS = {
    0: (0, 255, 255),    # yellow (BGR)
    1: (0, 165, 255),    # orange (BGR)
    2: (0, 100, 255),    # dark orange (BGR)
    3: (255, 0, 0),      # blue (BGR)
    4: (128, 128, 128),  # gray (BGR)
    5: (255, 255, 255)   # white (BGR)
}


def load_yolo_model(model_name: str, device: str = "cuda:0"):
    """Load YOLO model."""
    weights_path = MODEL_WEIGHTS.get(model_name)
    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} from {weights_path} on {device}...")
    model = YOLO(weights_path)
    return model, device


def draw_detections_on_image(image_path: str, detections_txt_path: str, output_path: str):
    """
    Draw detection boxes on image and save.

    Args:
        image_path: Path to image file
        detections_txt_path: Path to detections .txt file (class conf x1 y1 x2 y2 format)
        output_path: Path to save annotated image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not load image for visualization: {image_path}")
        return

    # Parse detections
    if not os.path.exists(detections_txt_path):
        print(f"Warning: Detections file not found: {detections_txt_path}")
        return

    with open(detections_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                cls = int(parts[0])
                conf = float(parts[1])
                x1, y1, x2, y2 = map(float, parts[2:6])

                # Get color and label
                color = CLASS_COLORS.get(cls, (255, 255, 255))
                label = CLASS_NAMES.get(cls, "unknown")

                # Draw bounding box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Draw label background
                label_text = f"{label} {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    img,
                    (int(x1), int(y1) - text_height - baseline - 5),
                    (int(x1) + text_width, int(y1)),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    img,
                    label_text,
                    (int(x1), int(y1) - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )

    # Save annotated image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"  📊 Saved visualization: {output_path}")


def run_yolo_inference(model, image_path: str, output_dir: str, device: str = "cuda:0", original_image_name: str = None):
    """
    Run YOLO inference and save detections in YOLO format.

    Args:
        model: YOLO model
        image_path: Path to image (may be transformed)
        output_dir: Directory to save detections
        device: Torch device
        original_image_name: Original image name (without transformation suffix) for detection filename

    Returns:
        detections_filepath: Path to saved detections .txt file
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    results = model.predict(
        source=img,
        conf=0.25,
        iou=0.45,
        device=device,
        verbose=False
    )

    # Extract detections
    detections = []
    boxes = results[0].boxes
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf.squeeze().cpu().item())
        cls = int(box.cls.squeeze().cpu().item())
        x1, y1, x2, y2 = xyxy
        detections.append(f"{cls} {conf:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

    # Save detections using original image name (without transformation suffix)
    os.makedirs(output_dir, exist_ok=True)
    if original_image_name:
        # Use provided original name (for transformed images)
        img_name = Path(original_image_name).stem
    else:
        # Use actual image name (for original images)
        img_name = Path(image_path).stem
    detections_filepath = os.path.join(output_dir, f"{img_name}.txt")
    with open(detections_filepath, 'w') as f:
        f.writelines(detections)

    return detections_filepath


def calculate_map(
    detections_dir: str,
    gt_labels_dir: str,
    image_name: str,
    img_width: int = 1920,
    img_height: int = 1080,
) -> float:
    """
    Calculate mAP using COCO evaluation.

    Args:
        detections_dir: Directory with the image's detection .txt file
        gt_labels_dir: Directory with ground truth .txt files (in YOLO format)
        image_name: Filename of the image being evaluated. Only its matching
            ground-truth label is included in this per-image evaluation.
        img_width: Image width for denormalizing coordinates
        img_height: Image height for denormalizing coordinates

    Returns:
        mAP: Mean Average Precision @ IoU=0.50:0.95
    """
    gt_label_file = Path(gt_labels_dir) / f"{Path(image_name).stem}.txt"
    if not gt_label_file.is_file():
        raise FileNotFoundError(
            f"Ground-truth label not found for {image_name}: {gt_label_file}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Format detections to COCO JSON
        det_json = os.path.join(tmpdir, "detections.json")
        gt_json = os.path.join(tmpdir, "ground_truth.json")

        # This search scores one image at a time. Formatting the entire labels
        # directory here makes COCO treat the other dataset images as missed
        # detections, driving every per-image AP close to zero.
        format_detections(
            gt_label_file,
            gt_json,
            as_ground_truth=True,
            is_yolo_format=True,
            img_width=img_width,
            img_height=img_height,
        )

        # Then format detections using the ground truth file to match image IDs
        format_detections(detections_dir, det_json, gt_coco_file=gt_json, as_ground_truth=False)

        # Evaluate
        try:
            stats = evaluate_coco(gt_json, det_json, iou_type='bbox')
            # stats[0] is AP @ IoU=0.50:0.95 (mAP)
            return float(stats[0]) if stats is not None else 0.0
        except Exception as e:
            print(f"  Warning: COCO evaluation failed: {e}")
            return 0.0


def apply_gamma_transformation(
    image: np.ndarray,
    gamma: float,
    image_name: str,
    output_dir: str
) -> Optional[str]:
    """
    Apply gamma transformation to image.

    Args:
        image: Image in BGR format (as loaded by cv2.imread)
        gamma: Gamma correction value
        image_name: Original image filename
        output_dir: Directory to save transformed image

    Returns:
        output_filepath: Path to transformed image, or None on failure
    """
    config = {
        "brightness_constant": BRIGHTNESS_CONSTANT,
        "brightness_multiplier": BRIGHTNESS_MULTIPLIER,
        "gamma": float(gamma),
        "method": METHOD,
    }

    # Image is already in BGR format from cv2.imread, no conversion needed
    image_path_tuple = os.path.splitext(image_name)

    success, output_filepath = gen_metamorphic(
        image, "imgDarkBright", config, image_path_tuple, output_dir
    )

    return output_filepath if success else None


def evaluate_gamma(
    gamma: float,
    image: np.ndarray,
    image_name: str,
    gt_labels_dir: str,
    model,
    device: str,
    results_dir: str,
    label: str,
    all_results: List[Tuple[float, float]],
    img_width: int,
    img_height: int,
) -> float:
    """Evaluate AP for one gamma value and record the result."""
    transform_dir = os.path.join(results_dir, f"gamma_{gamma:.3f}_{label}")
    transformed_path = apply_gamma_transformation(image, gamma, image_name, transform_dir)

    if transformed_path is None:
        raise RuntimeError(f"Failed to apply gamma transformation at gamma={gamma:.3f}")

    detections_dir = os.path.join(transform_dir, "detections")
    run_yolo_inference(
        model,
        transformed_path,
        detections_dir,
        device,
        original_image_name=image_name,
    )
    ap = calculate_map(detections_dir, gt_labels_dir, image_name, img_width, img_height)
    all_results.append((float(gamma), float(ap)))
    return ap


def refine_crossing(
    lo_gamma: float,
    lo_ap: float,
    hi_gamma: float,
    hi_ap: float,
    target_ap: float,
    image: np.ndarray,
    image_name: str,
    gt_labels_dir: str,
    model,
    device: str,
    results_dir: str,
    side_label: str,
    all_results: List[Tuple[float, float]],
    img_width: int,
    img_height: int,
) -> Tuple[float, float]:
    """Refine a target-AP bracket until the AP tolerance is met."""
    best_gamma, best_ap = min(
        ((lo_gamma, lo_ap), (hi_gamma, hi_ap)),
        key=lambda result: abs(result[1] - target_ap),
    )

    if abs(best_ap - target_ap) < 1e-9:
        return best_gamma, best_ap

    for iteration in range(MAX_ROOT_REFINEMENT_ITERATIONS):
        denominator = hi_ap - lo_ap
        if abs(denominator) < 1e-9:
            gamma = (lo_gamma + hi_gamma) / 2.0
        else:
            gamma = lo_gamma + (hi_gamma - lo_gamma) * (target_ap - lo_ap) / denominator
            gamma = float(np.clip(gamma, lo_gamma, hi_gamma))

        if gamma == lo_gamma or gamma == hi_gamma:
            gamma = (lo_gamma + hi_gamma) / 2.0

        ap = evaluate_gamma(
            gamma,
            image,
            image_name,
            gt_labels_dir,
            model,
            device,
            results_dir,
            f"root_{side_label}_iter{iteration}",
            all_results,
            img_width,
            img_height,
        )
        print(
            f"  [{side_label} refinement {iteration + 1}] "
            f"gamma={gamma:.3f}, mAP={ap:.4f}, "
            f"bracket=[{lo_gamma:.3f}, {hi_gamma:.3f}]"
        )

        if abs(ap - target_ap) < abs(best_ap - target_ap):
            best_gamma, best_ap = gamma, ap

        if abs(ap - target_ap) <= AP_TOLERANCE:
            return gamma, ap

        if (ap - target_ap) * (lo_ap - target_ap) > 0:
            lo_gamma, lo_ap = gamma, ap
        else:
            hi_gamma, hi_ap = gamma, ap

    return best_gamma, best_ap


def brute_force_gamma_search(
    image_path: str,
    gt_labels_dir: str,
    model,
    device: str,
    target_ap: float,
    results_dir: str,
) -> Dict:
    """
    Perform brute force search through gamma values.

    Returns:
        {
            'image_name': str,
            'original_ap': float,
            'gamma_left': float or None,
            'ap_left': float or None,
            'gamma_right': float or None,
            'ap_right': float or None,
            'iterations': int,
            'inference_count': int,
            'time_seconds': float,
            'all_results': [(gamma, ap), ...],
        }
    """
    image_name = Path(image_path).name
    print(f"\n{'='*60}")
    print(f"Processing: {image_name}")
    print(f"{'='*60}")

    start_time = time.time()

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Get image dimensions for ground truth conversion
    img_height, img_width = image.shape[:2]

    # Calculate original AP
    print("Calculating original AP...")
    orig_det_dir = os.path.join(results_dir, "original_detections")
    run_yolo_inference(model, image_path, orig_det_dir, device)
    original_ap = calculate_map(
        orig_det_dir, gt_labels_dir, image_name, img_width, img_height
    )
    print(f"  Original mAP: {original_ap:.4f}")

    # Check if original AP meets target threshold
    if original_ap < target_ap:
        print(f"  ⚠ Skipping image: Original AP ({original_ap:.4f}) is below target AP ({target_ap:.2f})")
        elapsed_time = time.time() - start_time
        return {
            'image_name': image_name,
            'original_ap': float(original_ap),
            'peak_gamma': None,
            'peak_ap': float(original_ap),
            'gamma_left': None,
            'ap_left': None,
            'gamma_right': None,
            'ap_right': None,
            'iterations': 0,
            'inference_count': 1,
            'time_seconds': elapsed_time,
            'all_results': [],
            'skipped': True,
            'skip_reason': f'Original AP ({original_ap:.4f}) below target ({target_ap:.2f})'
        }

    gamma_values = np.arange(GAMMA_RANGE[0], GAMMA_RANGE[1], GAMMA_STEP)
    if not np.isclose(gamma_values[-1], GAMMA_RANGE[1]):
        gamma_values = np.append(gamma_values, GAMMA_RANGE[1])
    print(f"\nStarting brute force search:")
    print(f"  Gamma range: [{GAMMA_RANGE[0]}, {GAMMA_RANGE[1]}]")
    print(f"  Step size: {GAMMA_STEP}")
    print(f"  Total iterations: {len(gamma_values)}")
    print(f"  Target AP: {target_ap:.2f}")
    print(f"  Root refinement AP tolerance: {AP_TOLERANCE:.2f}")

    all_results = []
    gamma_left = None
    ap_left = None
    gamma_right = None
    ap_right = None
    inference_count = 0
    grid_results = []

    # Track peak for finding crossings
    peak_gamma = None
    peak_ap = -1.0

    for i, gamma in enumerate(gamma_values):
        print(f"\n[Iteration {i+1}/{len(gamma_values)}] gamma={gamma:.2f}")

        # Apply gamma transformation
        transform_dir = os.path.join(results_dir, f"gamma_{gamma:.2f}")
        transformed_path = apply_gamma_transformation(image, gamma, image_name, transform_dir)

        if transformed_path is None:
            print(f"  ⚠ Failed to apply transformation")
            continue

        # Run inference (pass original image name for proper ground truth matching)
        detections_dir = os.path.join(transform_dir, "detections")
        run_yolo_inference(model, transformed_path, detections_dir, device, original_image_name=image_name)
        inference_count += 1

        # Calculate AP
        ap = calculate_map(
            detections_dir, gt_labels_dir, image_name, img_width, img_height
        )
        grid_results.append((float(gamma), float(ap)))
        all_results.append((float(gamma), float(ap)))
        print(f"  mAP: {ap:.4f}")

        # Save visualization for whole number gamma values
        if gamma == int(gamma) and 1 <= gamma <= 5:
            img_name_stem = Path(image_name).stem
            detection_file = os.path.join(detections_dir, f"{img_name_stem}.txt")
            vis_output_path = os.path.join(transform_dir, f"{img_name_stem}_annotated.png")
            draw_detections_on_image(transformed_path, detection_file, vis_output_path)

        # Track peak
        if ap > peak_ap:
            peak_ap = ap
            peak_gamma = gamma

    if grid_results:
        peak_index = max(range(len(grid_results)), key=lambda index: grid_results[index][1])

        for index in range(1, peak_index + 1):
            lo_gamma, lo_ap = grid_results[index - 1]
            hi_gamma, hi_ap = grid_results[index]
            if lo_ap < target_ap <= hi_ap:
                gamma_left, ap_left = refine_crossing(
                    lo_gamma,
                    lo_ap,
                    hi_gamma,
                    hi_ap,
                    target_ap,
                    image,
                    image_name,
                    gt_labels_dir,
                    model,
                    device,
                    results_dir,
                    "left",
                    all_results,
                    img_width,
                    img_height,
                )
                print(f"  ✓ Left crossing found: gamma={gamma_left:.3f}, AP={ap_left:.4f}")
                break

        for index in range(peak_index + 1, len(grid_results)):
            lo_gamma, lo_ap = grid_results[index - 1]
            hi_gamma, hi_ap = grid_results[index]
            if lo_ap > target_ap >= hi_ap:
                gamma_right, ap_right = refine_crossing(
                    lo_gamma,
                    lo_ap,
                    hi_gamma,
                    hi_ap,
                    target_ap,
                    image,
                    image_name,
                    gt_labels_dir,
                    model,
                    device,
                    results_dir,
                    "right",
                    all_results,
                    img_width,
                    img_height,
                )
                print(f"  ✓ Right crossing found: gamma={gamma_right:.3f}, AP={ap_right:.4f}")
                break

    all_results.sort(key=lambda result: result[0])
    inference_count = len(all_results)

    elapsed_time = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print("Brute Force Search Complete")
    print(f"{'='*60}")
    print(f"Original AP: {original_ap:.4f}")
    print(f"Peak AP: {peak_ap:.4f} at gamma={peak_gamma:.3f}")
    print(f"Left crossing: {'gamma=' + f'{gamma_left:.3f}, AP={ap_left:.4f}' if gamma_left else 'Not found'}")
    print(f"Right crossing: {'gamma=' + f'{gamma_right:.3f}, AP={ap_right:.4f}' if gamma_right else 'Not found'}")
    print(f"Iterations: {len(all_results)}")
    print(f"Inference count: {inference_count}")
    print(f"Time: {elapsed_time:.1f}s")

    return {
        'image_name': image_name,
        'original_ap': float(original_ap),
        'peak_gamma': float(peak_gamma) if peak_gamma else None,
        'peak_ap': float(peak_ap),
        'gamma_left': float(gamma_left) if gamma_left else None,
        'ap_left': float(ap_left) if ap_left else None,
        'gamma_right': float(gamma_right) if gamma_right else None,
        'ap_right': float(ap_right) if ap_right else None,
        'iterations': len(all_results),
        'inference_count': inference_count,
        'time_seconds': elapsed_time,
        'all_results': [(float(g), float(a)) for g, a in all_results],
    }


def main():
    parser = argparse.ArgumentParser(
        description="RQ2.a: Brute Force Gamma Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: all images, YOLOv8, target AP=0.7
  python scripts/rq2a_brute_force_search.py --model yolov8

  # Process 10 images
  python scripts/rq2a_brute_force_search.py --model yolov8 --num-images 10

  # Different target AP
  python scripts/rq2a_brute_force_search.py --model yolov8 --target-ap 0.6
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8",
        choices=["yolov5", "yolov8", "yolov12"],
        help="YOLO model to use (default: yolov8)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="test",
        help="Dataset name under data/base_images/ (default: test)"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to process (default: all images)"
    )
    parser.add_argument(
        "--target-ap",
        type=float,
        default=0.7,
        help="Target AP threshold (default: 0.7)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device (default: cuda:0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: analysis/results/rq2a_brute_force_<model>)"
    )

    args = parser.parse_args()

    # Set up paths
    images_dir = f"data/base_images/{args.dataset}/images"
    gt_labels_dir = f"data/base_images/{args.dataset}/labels"

    if args.output_dir is None:
        args.output_dir = f"analysis/results/rq2a_brute_force_{args.model}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Validate paths
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    if not os.path.exists(gt_labels_dir):
        print(f"Error: Ground truth labels not found: {gt_labels_dir}")
        sys.exit(1)

    # Load model
    model, device = load_yolo_model(args.model, args.device)

    # Get image files
    image_files = sorted(Path(images_dir).glob("*.png")) + sorted(Path(images_dir).glob("*.jpg"))
    if args.num_images:
        image_files = image_files[:args.num_images]

    if not image_files:
        print(f"Error: No images found in {images_dir}")
        sys.exit(1)

    # Print configuration
    print("="*60)
    print("RQ2.a: Brute Force Gamma Search Configuration")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Images to process: {len(image_files)}")
    print(f"Target AP: {args.target_ap}")
    print(f"Gamma range: [{GAMMA_RANGE[0]}, {GAMMA_RANGE[1]}]")
    print(f"Gamma step: {GAMMA_STEP}")
    print(f"AP tolerance: {AP_TOLERANCE}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print("="*60)

    # Process images
    results = []
    failed_images = []

    for img_path in image_files:
        try:
            result = brute_force_gamma_search(
                str(img_path),
                gt_labels_dir,
                model,
                device,
                args.target_ap,
                os.path.join(args.output_dir, img_path.stem),
            )
            results.append(result)
        except Exception as e:
            error_msg = f"Failed to process {img_path.name}: {e}"
            print(f"\n❌ ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            failed_images.append({'image': img_path.name, 'error': str(e)})

    # Separate skipped and processed results
    skipped_images = [r for r in results if r.get('skipped', False)]
    processed_results = [r for r in results if not r.get('skipped', False)]

    # Save results
    output_json = os.path.join(args.output_dir, "results.json")
    final_results = {
        "method": "brute_force",
        "model": args.model,
        "dataset": args.dataset,
        "target_ap": args.target_ap,
        "gamma_range": list(GAMMA_RANGE),
        "gamma_step": GAMMA_STEP,
        "ap_tolerance": AP_TOLERANCE,
        "max_root_refinement_iterations": MAX_ROOT_REFINEMENT_ITERATIONS,
        "total_images": len(image_files),
        "processed_images": len(processed_results),
        "skipped_images": len(skipped_images),
        "failed_images": len(failed_images),
        "results": processed_results,
        "skipped": skipped_images,
        "failures": failed_images,
        "summary": {
            "avg_gamma_left": float(np.mean([r['gamma_left'] for r in processed_results if r['gamma_left']])) if any(r.get('gamma_left') for r in processed_results) else None,
            "avg_gamma_right": float(np.mean([r['gamma_right'] for r in processed_results if r['gamma_right']])) if any(r.get('gamma_right') for r in processed_results) else None,
            "avg_peak_gamma": float(np.mean([r['peak_gamma'] for r in processed_results if r['peak_gamma']])) if any(r.get('peak_gamma') for r in processed_results) else None,
            "avg_iterations": float(np.mean([r['iterations'] for r in processed_results])) if processed_results else 0,
            "avg_inference_count": float(np.mean([r['inference_count'] for r in processed_results])) if processed_results else 0,
            "total_time_seconds": float(sum(r['time_seconds'] for r in results)),
            "avg_time_per_image": float(np.mean([r['time_seconds'] for r in processed_results])) if processed_results else 0,
        }
    }

    with open(output_json, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("RQ2.a: Brute Force Search Complete")
    print(f"{'='*60}")
    print(f"Total images: {len(image_files)}")
    print(f"Processed: {len(processed_results)}")
    print(f"Skipped (AP < target): {len(skipped_images)}")
    print(f"Failed: {len(failed_images)}")
    if processed_results:
        if final_results['summary']['avg_gamma_left']:
            print(f"Average left crossing gamma: {final_results['summary']['avg_gamma_left']:.2f}")
        if final_results['summary']['avg_gamma_right']:
            print(f"Average right crossing gamma: {final_results['summary']['avg_gamma_right']:.2f}")
        if final_results['summary']['avg_peak_gamma']:
            print(f"Average peak gamma: {final_results['summary']['avg_peak_gamma']:.2f}")
        print(f"Average iterations: {final_results['summary']['avg_iterations']:.0f}")
        print(f"Average time per image: {final_results['summary']['avg_time_per_image']:.1f}s")
    print(f"Total time: {final_results['summary']['total_time_seconds']:.1f}s")
    print(f"\nResults saved to: {output_json}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
