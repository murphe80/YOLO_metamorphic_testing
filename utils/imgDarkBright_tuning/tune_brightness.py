# script to find the optimal brightness transformation for the image

# For every image in the dataset, do the following:
# 1. Calculate the variance of the clusters of the image using mean shift segmentation 
# 2. Implement search algorithm to find brightness level for which accuracy for either blue or yellow cones is x%. 
# 3. Save original brightness, brightness level for which accuracy for either blue or yellow cones is x%. (json) 
# 4. Save the image with the brightness transformation applied in /data/transformed_images/tuned_brightness/

import os
import json
import cv2 as cv
import numpy as np
import argparse
from search_brightness import search_brightness 
from mean_shift import mean_shift_clustering_variance

# Define directories
DATASET_DIR = "../../data/base_images/fsoco_sample/images"
OUTPUT_DIR = "../../data/transformed_images/tuned_brightness"
GT_DETECTIONS_FILEPATH = "../../data/base_images/test/labels"

os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np

def luminosity_bt_709(image: np.ndarray) -> np.ndarray:
    """
    image: H x W x 3 float array (values 0.0–1.0 or 0–255)
    returns: H x W grayscale luminosity array
    """
    # convert to rgb
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    weights = np.array([0.2126, 0.7152, 0.0722])  # BT.709
    return image @ weights # matrix multiplication (dot product)

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
    high_region = v == 255
    return (np.sum(low_region) + np.sum(high_region)) / v.size

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

def calculate_brightness_scores(image_bgr: np.ndarray) -> tuple[float, float, float, float]:
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

def get_confidence_level(image, detections_path):
    """
    Get the confidence level of the detections in the image
    Returns the average confidence level of the detections in the image
    """
    # Find .txt detections file for the target input image
    detections_file = os.path.join(detections_path, image.replace(".jpg", ".txt"))
    with open(detections_file, "r") as f:
        detections = f.readlines()
    # Get the confidence level of the detections (second number on each line)
    confidence_levels = [float(detection.split(" ")[1]) for detection in detections]
    return np.mean(confidence_levels)

def tune_brightness(images_path, original_detections_path, target_ap, gt_detections_filepath):
    """
    Tune the brightness of the image
    """
    # iterate over all images in the images_path directory
    results = []
    print(f"Images found in {images_path}: {os.listdir(images_path)}")
    for image_path in os.listdir(images_path):
        print(f"Tuning brightness of {image_path}")
        image = cv.imread(os.path.join(images_path, image_path))
        brightness_scores = calculate_brightness_scores(image)
        original_confidence_level = get_confidence_level(image,original_detections_path)
        optimal_config, ap_b, ap_y, updated_image_filepath, updated_detections_path = search_brightness(image, image_path, target_ap, gt_detections_filepath)
        output_image = cv.imread(updated_image_filepath)
        updated_confidence_level = get_confidence_level(output_image, updated_detections_path)
        tuned_brightness_scores = calculate_brightness_scores(output_image) 
        results.append({"image_path": image_path, "brightness_scores": brightness_scores, "tuned_brightness_scores": tuned_brightness_scores, "optimal_config": optimal_config, "ap_blue": ap_b, "ap_yellow": ap_y, "original_confidence_level": original_confidence_level, "updated_confidence_level": updated_confidence_level})
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f)

def main():
    """
    Main function to tune the brightness of the images
    """
    # cli arguments
    parser = argparse.ArgumentParser(description="Tune the brightness of the images")
    parser.add_argument("--images_path", type=str, default=DATASET_DIR, help="Path to the images")
    parser.add_argument("--original_detections_path", type=str, required=True, help="Path to the original model detections")
    parser.add_argument("--target_ap", type=float, default=0.8, help="Target AP")
    parser.add_argument("--gt_detections_filepath", type=str, default=GT_DETECTIONS_FILEPATH, help="Path to the ground truth detections")
    args = parser.parse_args()

    tune_brightness(args.images_path, args.original_detections_path, args.target_ap, args.gt_detections_filepath)

if __name__ == "__main__":
    main()