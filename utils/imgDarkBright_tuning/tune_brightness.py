#!/usr/bin/env python3
"""
Brightness Tuning for YOLO Metamorphic Testing (RQ2)

Find the optimal brightness transformation for each image using a search-based algorithm.

For every image in the dataset:
1. Calculate brightness metrics (RMS contrast, mean shift variance, luminosity)
2. Search for brightness level where accuracy for blue or yellow cones reaches target threshold
3. Save optimal configuration and brightness scores
4. Save transformed image

Supports both sample and test datasets via --dataset parameter.

Usage:
    python tune_brightness.py --dataset fsoco_sample --model yolov5 --target_ap 0.8
    python tune_brightness.py --dataset test --model yolov8 --target_ap 0.7
"""

import os
import sys
import json
import cv2 as cv
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.imgDarkBright_tuning.search_brightness import search_brightness
from utils.imgDarkBright_tuning.mean_shift import mean_shift_clustering_variance


import numpy as np

def luminosity_bt_709(image: np.ndarray) -> float:
    """
    image: H x W x 3 float array (values 0.0–1.0 or 0–255)
    returns: mean BT.709 luminosity
    """
    # convert to rgb
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    weights = np.array([0.2126, 0.7152, 0.0722])  # BT.709
    return float(np.mean(image @ weights))

def extract_value_HSV(image_bgr: np.ndarray) -> np.ndarray:
    """
    Extract the value channel from the HSV image
    """
    hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0  # normalise to [0, 1]
    return v

def low_high_regions_ratio(image_bgr: np.ndarray) -> float:
    """
    Calculate the ratio of pixels that are 0 or 255 vs total pixels in the value channel of the HSV image
    """
    v = extract_value_HSV(image_bgr)
    low_region = v == 0
    high_region = v == 1
    return float((np.sum(low_region) + np.sum(high_region)) / v.size)

def calculate_rms(image_bgr: np.ndarray) -> float:
    """
    Compute RMS contrast of the image

    Args:
        image_bgr: Input BGR image as numpy array
    Returns:
        rms: RMS contrast of the image in [0, 1]
    """
    hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
    v   = hsv[:, :, 2].astype(np.float32) / 255.0  # normalise to [0, 1]

    rms    = float(np.std(v))

    return rms

def calculate_brightness_scores(image_bgr: np.ndarray) -> dict[str, float]:
    """
    Calculate the brightness scores of the image
    Returns json object with the brightness scores
    """
    rms = calculate_rms(image_bgr)
    lh_ratio = low_high_regions_ratio(image_bgr)
    mean_shift_variance = mean_shift_clustering_variance(image_bgr)
    luminosity_709 = luminosity_bt_709(image_bgr)
    return {
        "rms": rms,
        "low_high_regions_ratio": lh_ratio,
        "mean_shift_variance": mean_shift_variance,
        "luminosity_709": luminosity_709
    }

def get_confidence_level(image_filename, detections_path):
    """
    Get the average confidence level for an image's detections.

    ``detections_path`` may be either a directory containing one detection
    file per image or the path to a single detection file.

    Returns the average confidence level of the detections in the image
    """
    if os.path.isdir(detections_path):
        image_stem = Path(image_filename).stem
        detections_file = os.path.join(detections_path, f"{image_stem}.txt")
    else:
        detections_file = detections_path

    with open(detections_file, "r") as f:
        detections = f.readlines()

    # Get the confidence level of the detections (second number on each line)
    confidence_levels = [
        float(detection.split()[1]) for detection in detections if detection.strip()
    ]
    return np.mean(confidence_levels)

def tune_brightness(images_path, original_detections_path, target_ap, gt_detections_filepath, output_dir=None):
    """
    Tune the brightness of the image

    Args:
        images_path: Path to input images
        original_detections_path: Path to model detections on original images
        target_ap: Target AP threshold to search for
        gt_detections_filepath: Path to ground truth labels
        output_dir: Output directory for results (optional)
    """
    if output_dir is None:
        output_dir = "data/transformed_images/tuned_brightness"

    os.makedirs(output_dir, exist_ok=True)

    # iterate over all images in the images_path directory
    results = []
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]
    print(f"Found {len(image_files)} images in {images_path}")

    for image_path in image_files:
        print(f"Tuning brightness of {image_path}")
        image = cv.imread(os.path.join(images_path, image_path))

        if image is None:
            print(f"  Warning: Could not read {image_path}, skipping")
            continue

        brightness_scores = calculate_brightness_scores(image)
        original_confidence_level = get_confidence_level(image_path, original_detections_path)
        optimal_config, ap_b, ap_y, updated_image_filepath, updated_detections_path = search_brightness(
            image, image_path, target_ap, gt_detections_filepath
        )
        output_image = cv.imread(updated_image_filepath)
        updated_confidence_level = get_confidence_level(
            updated_image_filepath, updated_detections_path
        )
        tuned_brightness_scores = calculate_brightness_scores(output_image)

        results.append({
            "image_path": image_path,
            "brightness_scores": brightness_scores,
            "tuned_brightness_scores": tuned_brightness_scores,
            "optimal_config": optimal_config,
            "ap_blue": ap_b,
            "ap_yellow": ap_y,
            "original_confidence_level": original_confidence_level,
            "updated_confidence_level": updated_confidence_level
        })
        print(f"  ✓ AP blue: {ap_b:.3f}, AP yellow: {ap_y:.3f}")

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results_path

def main():
    """
    Main function to tune the brightness of the images
    """
    parser = argparse.ArgumentParser(
        description="Brightness tuning via search algorithm (RQ2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample dataset with YOLOv5
  python tune_brightness.py --dataset fsoco_sample --model yolov5

  # Test dataset with YOLOv8, target AP 0.7
  python tune_brightness.py --dataset test --model yolov8 --target_ap 0.7
        """
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="fsoco_sample",
        help="Dataset name: fsoco_sample, test, or custom (default: fsoco_sample)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolov5", "yolov8", "yolov12"],
        help="Model to use for detections"
    )

    # Paths (auto-configured from dataset if not specified)
    parser.add_argument(
        "--images_path",
        type=str,
        help="Path to images (default: data/base_images/<dataset>/images or segmented_images)"
    )
    parser.add_argument(
        "--original_detections_path",
        type=str,
        help="Path to original model detections (default: results/<model>/original)"
    )
    parser.add_argument(
        "--gt_detections_filepath",
        type=str,
        help="Path to ground truth labels (default: data/base_images/<dataset>/labels)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (default: data/transformed_images/<dataset>_tuned_brightness)"
    )

    # Algorithm parameters
    parser.add_argument(
        "--target_ap",
        type=float,
        default=0.8,
        help="Target AP threshold (default: 0.8)"
    )

    args = parser.parse_args()

    # Auto-configure paths based on dataset
    if args.images_path is None:
        if args.dataset == "fsoco_sample":
            args.images_path = "data/base_images/fsoco_sample/segmented_images"
        else:
            args.images_path = f"data/base_images/{args.dataset}/images"

    if args.original_detections_path is None:
        args.original_detections_path = f"results/{args.dataset}/{args.model}/original"

    if args.gt_detections_filepath is None:
        if args.dataset == "fsoco_sample":
            args.gt_detections_filepath = "data/base_images/fsoco_sample/bounding_boxes"
        else:
            args.gt_detections_filepath = f"data/base_images/{args.dataset}/labels"

    if args.output_dir is None:
        args.output_dir = f"data/transformed_images/{args.dataset}_tuned_brightness"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate paths
    if not os.path.exists(args.images_path):
        print(f"Error: Images directory not found: {args.images_path}")
        sys.exit(1)

    if not os.path.exists(args.original_detections_path):
        print(f"Error: Original detections not found: {args.original_detections_path}")
        print(f"Run inference first: python models/{args.model}/infer_*.py")
        sys.exit(1)

    if not os.path.exists(args.gt_detections_filepath):
        print(f"Error: Ground truth labels not found: {args.gt_detections_filepath}")
        sys.exit(1)

    print("=" * 60)
    print("Brightness Tuning Configuration")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Target AP: {args.target_ap}")
    print(f"Images: {args.images_path}")
    print(f"Detections: {args.original_detections_path}")
    print(f"Ground truth: {args.gt_detections_filepath}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    print()

    # Run brightness tuning
    results_path = tune_brightness(
        args.images_path,
        args.original_detections_path,
        args.target_ap,
        args.gt_detections_filepath,
        args.output_dir
    )

    print()
    print("=" * 60)
    print("Brightness tuning complete!")
    print(f"Results saved to: {results_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
