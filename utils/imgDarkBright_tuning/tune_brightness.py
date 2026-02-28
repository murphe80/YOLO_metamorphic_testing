# script to find the optimal brightness transformation for the image

# For every image in the dataset, do the following:
# 1. Calculate the brightness of the image (overall brightness)
# 2. Implement search algorithm to find brightness level for which accuracy for either blue or yellow cones is x%. 
# 3. Save original brightness, brightness level for which accuracy for either blue or yellow cones is x%. (json) 
# 4. Save the image with the brightness transformation applied in /data/transformed_images/tuned_brightness/

import os
import json
import cv2 as cv
import numpy as np
import argparse
from search_brightness import search_brightness 
# Define directories
DATASET_DIR = "../../data/base_images/fsoco_test_segmented/all_images"
OUTPUT_DIR = "../../data/transformed_images/tuned_brightness"


import numpy as np

def luminosity(image: np.ndarray) -> np.ndarray:
    """
    image: H x W x 3 float array (values 0.0–1.0 or 0–255)
    returns: H x W grayscale luminosity array
    """
    # convert to rgb
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    weights = np.array([0.2126, 0.7152, 0.0722])  # BT.709
    return image @ weights # matrix multiplication (dot product)


def tune_brightness(images_path, target_ap, gt_detections_filepath):
    """
    Tune the brightness of the image
    """
    # iterate over all images in the images_path
    results = []
    for image_path in os.listdir(images_path):
        image = cv.imread(os.path.join(images_path, image_path))
        image_original_brightness = luminosity(image)
        optimal_config, final_ap, next_image_filepath = search_brightness(image, image_path, target_ap, gt_detections_filepath)
        output_image = cv.imread(next_image_filepath)
        image_tuned_brightness = luminosity(output_image)
        results.append({"image_path": image_path, "image_original_brightness": image_original_brightness, "optimal_config": optimal_config, "final_ap": final_ap, "image_tuned_brightness": image_tuned_brightness})
   
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f)

def main():
    """
    Main function to tune the brightness of the images
    """
    # cli arguments
    parser = argparse.ArgumentParser(description="Tune the brightness of the images")
    parser.add_argument("--images_path", type=str, required=True, help="Path to the images")
    parser.add_argument("--target_ap", type=float, required=True, help="Target AP")
    parser.add_argument("--gt_detections_filepath", type=str, required=True, help="Path to the ground truth detections")
    args = parser.parse_args()
    
    tune_brightness(args.images_path, args.target_ap, args.gt_detections_filepath)