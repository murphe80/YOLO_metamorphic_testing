#!/usr/bin/env python3
"""
RQ2.a: Golden-Section Gamma Search

Uses golden-section search to efficiently find operational brightness thresholds
where model AP crosses the target threshold.

Finds both left (darker) and right (brighter) crossing points using:
1. Golden-section search to find peak AP
2. Regula-falsi to find crossings on left and right slopes

Usage:
    python scripts/rq2a_golden_section_search.py --model yolov8 --target-ap 0.7
    python scripts/rq2a_golden_section_search.py --model yolov8 --num-images 10
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
from utils.model_registry import (
    get_default_model_name,
    get_model_by_name,
    get_model_names,
)
from analysis.utils.format_detections import format_detections
from analysis.coco_eval import evaluate_coco

# Configuration
GAMMA_LOW = 0.01
GAMMA_HIGH = 20.0
TOLERANCE_GAMMA = 0.1  # Convergence threshold for peak
AP_TOLERANCE = 0.05  # Tolerance for target AP matching
MAX_ITERATIONS = 35
BRIGHTNESS_CONSTANT = 0
BRIGHTNESS_MULTIPLIER = 1.0
METHOD = "gamma"

# Golden-section ratio
_GSR = (np.sqrt(5) - 1) / 2  # ≈ 0.618

PROJECT_ROOT = Path(__file__).parent.parent
AVAILABLE_MODELS = get_model_names()
DEFAULT_MODEL = get_default_model_name()

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


def atomic_json_dump(data: Dict, path: str) -> None:
    """Write JSON atomically so an interruption cannot corrupt a checkpoint."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temporary_path, output_path)


def load_json(path: str) -> Optional[Dict]:
    """Load a JSON object, returning None when it is missing or invalid."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_yolo_model(model_name: str, device: str = "cuda:0"):
    """Load YOLO model."""
    model_config = get_model_by_name(model_name)
    if model_config is None or model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not enabled in config/models.json"
        )

    weights_path = PROJECT_ROOT / model_config["weights"]
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} from {weights_path} on {device}...")
    model = YOLO(str(weights_path))
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
    print(f"    📊 Saved visualization: {output_path}")


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

        # The search scores one image at a time. Restrict the ground truth to
        # that image so unrelated dataset images are not counted as misses.
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
    iteration_counter: List[int],
    all_results: List[Tuple[float, float]],
    img_width: int = 1920,
    img_height: int = 1080,
) -> float:
    """
    Apply gamma, run inference, calculate AP.

    Args:
        iteration_counter: List with single element for tracking iteration count
        all_results: List to store (gamma, ap) tuples for visualization
        img_width: Image width for denormalizing ground truth
        img_height: Image height for denormalizing ground truth

    Returns:
        mAP value
    """
    iteration_counter[0] += 1
    print(f"  [Iteration {iteration_counter[0]}] Evaluating gamma={gamma:.3f} ({label})")

    transform_dir = os.path.join(results_dir, f"gamma_{gamma:.3f}_{label}")
    detections_dir = os.path.join(transform_dir, "detections")
    detection_file = os.path.join(detections_dir, f"{Path(image_name).stem}.txt")

    # Recover artifacts created before per-image checkpointing was added.
    if os.path.isfile(detection_file):
        ap = calculate_map(
            detections_dir, gt_labels_dir, image_name, img_width, img_height
        )
        all_results.append((float(gamma), float(ap)))
        print(f"    ↻ Recovered existing detection, mAP: {ap:.4f}")
        return ap

    # Apply transformation
    transformed_path = apply_gamma_transformation(image, gamma, image_name, transform_dir)

    if transformed_path is None:
        print(f"    ⚠ Transformation failed")
        return 0.0

    # Run inference (pass original image name for proper ground truth matching)
    run_yolo_inference(model, transformed_path, detections_dir, device, original_image_name=image_name)

    # Calculate AP
    ap = calculate_map(detections_dir, gt_labels_dir, image_name, img_width, img_height)
    print(f"    mAP: {ap:.4f}")

    # Store result for visualization
    all_results.append((float(gamma), float(ap)))

    # Save visualization for whole number gamma values
    if gamma == int(gamma) and 1 <= gamma <= 5:
        img_name_stem = Path(image_name).stem
        detection_file = os.path.join(detections_dir, f"{img_name_stem}.txt")
        vis_output_path = os.path.join(transform_dir, f"{img_name_stem}_annotated.png")
        draw_detections_on_image(transformed_path, detection_file, vis_output_path)

    return ap


def find_peak_gamma(
    image: np.ndarray,
    image_name: str,
    gt_labels_dir: str,
    model,
    device: str,
    results_dir: str,
    iteration_counter: List[int],
    all_results: List[Tuple[float, float]],
    img_width: int = 1920,
    img_height: int = 1080,
) -> Tuple[float, float]:
    """
    Use golden-section search to find gamma that maximizes AP.

    Returns:
        (peak_gamma, peak_ap)
    """
    print("\n[Phase 1] Golden-section search for peak AP")

    a = GAMMA_LOW
    b = GAMMA_HIGH

    # Initial interior probes
    m1 = b - _GSR * (b - a)
    m2 = a + _GSR * (b - a)

    f1 = evaluate_gamma(m1, image, image_name, gt_labels_dir, model, device,
                       results_dir, "peak_m1_init", iteration_counter, all_results,
                       img_width, img_height)
    f2 = evaluate_gamma(m2, image, image_name, gt_labels_dir, model, device,
                       results_dir, "peak_m2_init", iteration_counter, all_results,
                       img_width, img_height)

    iteration = 0
    while (b - a) > TOLERANCE_GAMMA and iteration < MAX_ITERATIONS:
        iteration += 1

        if f1 < f2:
            # Peak is in [m1, b]
            a, m1, f1 = m1, m2, f2
            m2 = a + _GSR * (b - a)
            f2 = evaluate_gamma(m2, image, image_name, gt_labels_dir, model, device,
                               results_dir, f"peak_iter{iteration}_m2", iteration_counter,
                               all_results, img_width, img_height)
        else:
            # Peak is in [a, m2]
            b, m2, f2 = m2, m1, f1
            m1 = b - _GSR * (b - a)
            f1 = evaluate_gamma(m1, image, image_name, gt_labels_dir, model, device,
                               results_dir, f"peak_iter{iteration}_m1", iteration_counter,
                               all_results, img_width, img_height)

    peak_gamma = (a + b) / 2.0
    peak_ap = (f1 + f2) / 2.0

    print(f"  ✓ Peak found: gamma={peak_gamma:.3f}, mAP={peak_ap:.4f}")
    return peak_gamma, peak_ap


def regula_falsi_search(
    lo_gamma: float,
    hi_gamma: float,
    lo_ap: float,
    hi_ap: float,
    target_ap: float,
    image: np.ndarray,
    image_name: str,
    gt_labels_dir: str,
    model,
    device: str,
    results_dir: str,
    side_label: str,
    iteration_counter: List[int],
    all_results: List[Tuple[float, float]],
    img_width: int = 1920,
    img_height: int = 1080,
) -> Tuple[float, float]:
    """
    Use regula-falsi to find gamma where AP crosses target.

    Returns:
        (best_gamma, best_ap)
    """
    best_gamma = lo_gamma
    best_ap = lo_ap

    for i in range(MAX_ITERATIONS):
        # Regula-falsi interpolation
        denom = hi_ap - lo_ap
        if abs(denom) < 1e-9:
            mid = (lo_gamma + hi_gamma) / 2.0
        else:
            mid = lo_gamma + (hi_gamma - lo_gamma) * (target_ap - lo_ap) / denom
            mid = float(np.clip(mid, min(lo_gamma, hi_gamma), max(lo_gamma, hi_gamma)))

        mid_ap = evaluate_gamma(mid, image, image_name, gt_labels_dir, model, device,
                               results_dir, f"rf_{side_label}_iter{i}", iteration_counter,
                               all_results, img_width, img_height)

        best_gamma = mid
        best_ap = mid_ap

        print(f"    [{side_label}] gamma={mid:.3f}, mAP={mid_ap:.4f}, bracket=[{lo_gamma:.3f}, {hi_gamma:.3f}]")

        # Check convergence
        if abs(mid_ap - target_ap) <= AP_TOLERANCE:
            print(f"    ✓ Converged: |{mid_ap:.4f} - {target_ap:.2f}| <= {AP_TOLERANCE:.2f}")
            break

        # Tighten bracket
        if (mid_ap - target_ap) * (lo_ap - target_ap) > 0:
            lo_gamma, lo_ap = mid, mid_ap
        else:
            hi_gamma, hi_ap = mid, mid_ap

    return best_gamma, best_ap


def golden_section_gamma_search(
    image_path: str,
    gt_labels_dir: str,
    model,
    device: str,
    target_ap: float,
    results_dir: str,
) -> Dict:
    """
    Perform golden-section search to find gamma crossings.

    Returns:
        {
            'image_name': str,
            'original_ap': float,
            'peak_gamma': float,
            'peak_ap': float,
            'gamma_left': float or None,
            'ap_left': float or None,
            'gamma_right': float or None,
            'ap_right': float or None,
            'iterations': int,
            'inference_count': int,
            'time_seconds': float,
        }
    """
    image_name = Path(image_path).name
    print(f"\n{'='*60}")
    print(f"Processing: {image_name}")
    print(f"{'='*60}")

    start_time = time.time()
    iteration_counter = [0]
    all_results = []  # Store (gamma, ap) tuples for visualization

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Get image dimensions for ground truth conversion
    img_height, img_width = image.shape[:2]

    # Calculate original AP
    print("Calculating original AP...")
    orig_det_dir = os.path.join(results_dir, "original_detections")
    original_detection = os.path.join(
        orig_det_dir, f"{Path(image_name).stem}.txt"
    )
    if not os.path.isfile(original_detection):
        run_yolo_inference(model, image_path, orig_det_dir, device)
    else:
        print("  ↻ Reusing existing original detection")
    original_ap = calculate_map(
        orig_det_dir, gt_labels_dir, image_name, img_width, img_height
    )
    print(f"  Original mAP: {original_ap:.4f}")

    print(f"\nTarget AP: {target_ap:.2f} ± {AP_TOLERANCE:.2f}")

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

    # Phase 1: Find peak
    peak_gamma, peak_ap = find_peak_gamma(
        image, image_name, gt_labels_dir, model, device, results_dir,
        iteration_counter, all_results, img_width, img_height
    )

    # Check if target is reachable
    if peak_ap < (target_ap - AP_TOLERANCE):
        print(f"\n⚠ Target AP {target_ap:.2f} is unreachable (peak={peak_ap:.4f})")
        elapsed_time = time.time() - start_time
        return {
            'image_name': image_name,
            'original_ap': float(original_ap),
            'peak_gamma': float(peak_gamma),
            'peak_ap': float(peak_ap),
            'gamma_left': None,
            'ap_left': None,
            'gamma_right': None,
            'ap_right': None,
            'iterations': iteration_counter[0],
            'inference_count': iteration_counter[0],
            'time_seconds': elapsed_time,
            'all_results': all_results,
            'note': 'Target AP unreachable',
        }

    # Phase 2: Find crossings using regula-falsi
    print("\n[Phase 2] Regula-falsi search for crossings")

    # Evaluate endpoints
    print("  Evaluating endpoints...")
    ap_low = evaluate_gamma(GAMMA_LOW, image, image_name, gt_labels_dir, model, device,
                           results_dir, "endpoint_low", iteration_counter, all_results,
                           img_width, img_height)
    ap_high = evaluate_gamma(GAMMA_HIGH, image, image_name, gt_labels_dir, model, device,
                            results_dir, "endpoint_high", iteration_counter, all_results,
                            img_width, img_height)

    gamma_left = None
    ap_left = None
    gamma_right = None
    ap_right = None

    # Left slope: [LOW → peak]
    # Check if target is bracketed: we need target_ap to lie between ap_low and peak_ap
    print("\n  Searching left slope...")
    if ap_low < target_ap < peak_ap:
        gamma_left, ap_left = regula_falsi_search(
            GAMMA_LOW, peak_gamma, ap_low, peak_ap, target_ap,
            image, image_name, gt_labels_dir, model, device, results_dir,
            "left", iteration_counter, all_results, img_width, img_height
        )
        print(f"  ✓ Left crossing: gamma={gamma_left:.3f}, mAP={ap_left:.4f}")
    else:
        print(f"  ⊗ Target not bracketed on left slope (AP_low={ap_low:.4f}, peak={peak_ap:.4f}, target={target_ap:.2f})")

    # Right slope: [peak → HIGH]
    # Check if target is bracketed: we need target_ap to lie between ap_high and peak_ap
    print("\n  Searching right slope...")
    if ap_high < target_ap < peak_ap:
        gamma_right, ap_right = regula_falsi_search(
            peak_gamma, GAMMA_HIGH, peak_ap, ap_high, target_ap,
            image, image_name, gt_labels_dir, model, device, results_dir,
            "right", iteration_counter, all_results, img_width, img_height
        )
        print(f"  ✓ Right crossing: gamma={gamma_right:.3f}, mAP={ap_right:.4f}")
    else:
        print(f"  ⊗ Target not bracketed on right slope (peak={peak_ap:.4f}, AP_high={ap_high:.4f}, target={target_ap:.2f})")

    elapsed_time = time.time() - start_time

    # Sort all_results by gamma for cleaner visualization
    all_results.sort(key=lambda x: x[0])

    # Summary
    print(f"\n{'='*60}")
    print("Golden-Section Search Complete")
    print(f"{'='*60}")
    print(f"Original AP: {original_ap:.4f}")
    print(f"Peak: gamma={peak_gamma:.3f}, mAP={peak_ap:.4f}")
    print(f"Left crossing: {'gamma=' + f'{gamma_left:.3f}, mAP={ap_left:.4f}' if gamma_left else 'Not found'}")
    print(f"Right crossing: {'gamma=' + f'{gamma_right:.3f}, mAP={ap_right:.4f}' if gamma_right else 'Not found'}")
    print(f"Iterations: {iteration_counter[0]}")
    print(f"Time: {elapsed_time:.1f}s")

    return {
        'image_name': image_name,
        'original_ap': float(original_ap),
        'peak_gamma': float(peak_gamma),
        'peak_ap': float(peak_ap),
        'gamma_left': float(gamma_left) if gamma_left else None,
        'ap_left': float(ap_left) if ap_left else None,
        'gamma_right': float(gamma_right) if gamma_right else None,
        'ap_right': float(ap_right) if ap_right else None,
        'iterations': iteration_counter[0],
        'inference_count': iteration_counter[0],
        'time_seconds': elapsed_time,
        'all_results': all_results,
    }


def build_final_results(
    model_name: str,
    dataset: str,
    target_ap: float,
    image_files: List[Path],
    results: List[Dict],
    failed_images: List[Dict],
) -> Dict:
    skipped_images = [r for r in results if r.get("skipped", False)]
    processed_results = [r for r in results if not r.get("skipped", False)]
    return {
        "method": "golden_section",
        "model": model_name,
        "dataset": dataset,
        "target_ap": target_ap,
        "gamma_range": [GAMMA_LOW, GAMMA_HIGH],
        "gamma_tolerance": TOLERANCE_GAMMA,
        "ap_tolerance": AP_TOLERANCE,
        "max_iterations": MAX_ITERATIONS,
        "total_images": len(image_files),
        "completed_images": len(results),
        "processed_images": len(processed_results),
        "skipped_images": len(skipped_images),
        "failed_images": len(failed_images),
        "results": processed_results,
        "skipped": skipped_images,
        "failures": failed_images,
        "summary": {
            "avg_gamma_left": float(np.mean([r["gamma_left"] for r in processed_results if r["gamma_left"]])) if any(r.get("gamma_left") for r in processed_results) else None,
            "avg_gamma_right": float(np.mean([r["gamma_right"] for r in processed_results if r["gamma_right"]])) if any(r.get("gamma_right") for r in processed_results) else None,
            "avg_peak_gamma": float(np.mean([r["peak_gamma"] for r in processed_results if r.get("peak_gamma")])) if any(r.get("peak_gamma") for r in processed_results) else None,
            "avg_iterations": float(np.mean([r["iterations"] for r in processed_results])) if processed_results else 0,
            "avg_inference_count": float(np.mean([r["inference_count"] for r in processed_results])) if processed_results else 0,
            "total_time_seconds": float(sum(r["time_seconds"] for r in results)),
            "avg_time_per_image": float(np.mean([r["time_seconds"] for r in processed_results])) if processed_results else 0,
        },
    }


def run_model(
    model_name: str,
    args: argparse.Namespace,
    image_files: List[Path],
    gt_labels_dir: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, "results.json")
    if os.path.isfile(output_json):
        print(
            f"↻ Skipping {model_name}: aggregate results already exist at "
            f"{output_json}"
        )
        return

    results: List[Dict] = []

    for img_path in image_files:
        image_checkpoint = load_json(
            os.path.join(output_dir, img_path.stem, "results.json")
        )
        if (
            image_checkpoint
            and image_checkpoint.get("method") == "golden_section"
            and image_checkpoint.get("model") == model_name
            and image_checkpoint.get("dataset") == args.dataset
            and image_checkpoint.get("target_ap") == args.target_ap
            and image_checkpoint.get("gamma_range") == [GAMMA_LOW, GAMMA_HIGH]
            and image_checkpoint.get("gamma_tolerance") == TOLERANCE_GAMMA
            and isinstance(image_checkpoint.get("result"), dict)
        ):
            results.append(image_checkpoint["result"])

    if results:
        print(f"↻ Resuming {model_name}: {len(results)} completed images found")

    completed_names = {result["image_name"] for result in results}
    failed_images: List[Dict] = []
    model, device = load_yolo_model(model_name, args.device)

    print("=" * 60)
    print("RQ2.a: Golden-Section Gamma Search Configuration")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Images to process: {len(image_files)}")
    print(f"Already completed: {len(completed_names)}")
    print(f"Target AP: {args.target_ap}")
    print(f"Gamma range: [{GAMMA_LOW}, {GAMMA_HIGH}]")
    print(f"Gamma tolerance: {TOLERANCE_GAMMA}")
    print(f"AP tolerance: {AP_TOLERANCE}")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    for img_path in image_files:
        if img_path.name in completed_names:
            print(f"↻ Skipping completed image: {img_path.name}")
            continue
        try:
            result = golden_section_gamma_search(
                str(img_path),
                gt_labels_dir,
                model,
                device,
                args.target_ap,
                os.path.join(output_dir, img_path.stem),
            )
            atomic_json_dump(
                {
                    "method": "golden_section",
                    "model": model_name,
                    "dataset": args.dataset,
                    "target_ap": args.target_ap,
                    "gamma_range": [GAMMA_LOW, GAMMA_HIGH],
                    "gamma_tolerance": TOLERANCE_GAMMA,
                    "result": result,
                },
                os.path.join(output_dir, img_path.stem, "results.json"),
            )
            results.append(result)
        except Exception as e:
            error_msg = f"Failed to process {img_path.name}: {e}"
            print(f"\n❌ ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            failed_images.append({"image": img_path.name, "error": str(e)})

    final_results = build_final_results(
        model_name,
        args.dataset,
        args.target_ap,
        image_files,
        results,
        failed_images,
    )
    atomic_json_dump(final_results, output_json)

    print(f"\n{'='*60}")
    print("RQ2.a: Golden-Section Search Complete")
    print(f"{'='*60}")
    print(f"Total images: {len(image_files)}")
    print(f"Processed: {final_results['processed_images']}")
    print(f"Skipped (AP < target): {final_results['skipped_images']}")
    print(f"Failed: {len(failed_images)}")
    print(f"Results saved to: {output_json}")
    print(f"{'='*60}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="RQ2.a: Golden-Section Gamma Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: all images, YOLOv8, target AP=0.7
  python scripts/rq2a_golden_section_search.py --model yolov8

  # Process 10 images
  python scripts/rq2a_golden_section_search.py --model yolov8 --num-images 10

  # Different target AP
  python scripts/rq2a_golden_section_search.py --model yolov8 --target-ap 0.6

  # Run every enabled model sequentially
  python scripts/rq2a_golden_section_search.py --model all
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=AVAILABLE_MODELS + ["all"],
        help=f"YOLO model to use, or 'all' (default: {DEFAULT_MODEL})"
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
        default=0.5,
        help="Target AP threshold (default: 0.5)"
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
        help=(
            "Output directory. With --model all, model-specific directories "
            "are created beneath it."
        )
    )

    args = parser.parse_args()

    # Set up paths
    images_dir = f"data/base_images/{args.dataset}/images"
    gt_labels_dir = f"data/base_images/{args.dataset}/labels"

    # Validate paths
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    if not os.path.exists(gt_labels_dir):
        print(f"Error: Ground truth labels not found: {gt_labels_dir}")
        sys.exit(1)

    # Get image files
    image_files = sorted(Path(images_dir).glob("*.png")) + sorted(Path(images_dir).glob("*.jpg"))
    if args.num_images:
        image_files = image_files[:args.num_images]

    if not image_files:
        print(f"Error: No images found in {images_dir}")
        sys.exit(1)

    model_names = AVAILABLE_MODELS if args.model == "all" else [args.model]
    for model_name in model_names:
        if args.output_dir is None:
            output_dir = (
                f"analysis/results/rq2a/rq2a_golden_section_{model_name}"
            )
        elif args.model == "all":
            output_dir = os.path.join(
                args.output_dir, f"rq2a_golden_section_{model_name}"
            )
        else:
            output_dir = args.output_dir
        run_model(model_name, args, image_files, gt_labels_dir, output_dir)


if __name__ == "__main__":
    main()
