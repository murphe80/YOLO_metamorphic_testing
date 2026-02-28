# Search algorithm to find the optimal brightness transformation for the image
# use regula falsi method to search for the optimal brightness transformation

# 1. Get brightness level for which accuracy is above and below x% (save results to json in /utils/imgDarkBright_tuning/results/)
# 2. Iteratively narrow down the range until the range is less than 0.1%
# 3. Return the brightness level for which accuracy is x%

import os
import json
import cv2 as cv
import numpy as np
from utils.matias_metamorphic import gen_metamorphic
from analysis.utils.format_detections import format_detections
from analysis.coco_eval_per_class import evaluate_coco_per_class

BRIGHTNESS_MULTIPLIER_LOW = 0.1
BRIGHTNESS_MULTIPLIER_HIGH = 0.9
BRIGHTNESS_CONSTANT = 0
GAMMA = 1
METHOD = "gamma"
TARGET_AP = 0.6
THRESHOLD_AP = 0.01
MAX_ITERATIONS = 20
BLUE_CLASS_ID = 3
YELLOW_CLASS_ID = 0

RESULTS_DIR = "/utils/imgDarkBright_tuning/search_results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_results(results):
    """
    Save results to json in RESULTS_DIR
    """
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f)

def find_starting_range(image, image_path, target_accuracy, gt_detections_filepath):
    """
    Find starting brightness levels for which accuracy is above and below target accuracy.
    low_config  → accuracy BELOW target  (the bracket floor)
    high_config → accuracy ABOVE target  (the bracket ceiling)
    """
    low_brightness_config = {
        "brightness_constant": BRIGHTNESS_CONSTANT,
        "brightness_multiplier": BRIGHTNESS_MULTIPLIER_LOW,
        "gamma": GAMMA,
        "method": METHOD
    }
    high_brightness_config = {
        "brightness_constant": BRIGHTNESS_CONSTANT,
        "brightness_multiplier": BRIGHTNESS_MULTIPLIER_HIGH,
        "gamma": GAMMA,
        "method": METHOD
    }

    # find setting below threshold accuracy
    output_dir = os.path.join(RESULTS_DIR, "low_brightness")
    os.makedirs(output_dir, exist_ok=True)
    found_low_brightness = False
    low_brightness_accuracy = 0
    while not found_low_brightness:
        success, output_filepath = gen_metamorphic(image, "imgDarkBright", low_brightness_config, image_path, output_dir)
        if success:
            low_brightness_accuracy = get_accuracy(output_filepath, gt_detections_filepath)
            if low_brightness_accuracy < target_accuracy:
                found_low_brightness = True
            else:
                low_brightness_config["brightness_multiplier"] -= 0.1
                if low_brightness_config["brightness_multiplier"] <= 0:
                    raise Exception("Failed to find a brightness level with accuracy below target")
        else:
            raise Exception("Failed to run inference for low brightness level")

    # find setting above threshold accuracy
    output_dir = os.path.join(RESULTS_DIR, "high_brightness")
    os.makedirs(output_dir, exist_ok=True)
    found_high_brightness = False
    high_brightness_accuracy = 0
    while not found_high_brightness:
        success, output_filepath = gen_metamorphic(image, "imgDarkBright", high_brightness_config, image_path, output_dir)
        if success:
            high_brightness_accuracy = get_accuracy(output_filepath, gt_detections_filepath)
            if high_brightness_accuracy >= target_accuracy:  # this was already correct
                found_high_brightness = True
            else:
                if high_brightness_config["brightness_multiplier"] >= 1.0:
                    raise Exception("Failed to find high brightness level")
                high_brightness_config["brightness_multiplier"] += 0.1
        else:
            raise Exception("Failed to run inference for high brightness level")

    return low_brightness_config, high_brightness_config, low_brightness_accuracy, high_brightness_accuracy


def get_accuracy(transformed_detections_filepath, gt_detections_filepath): 
    """
    Load the image from the output filepath
    Get accuracy for the image
    Return the accuracy
    """
    # format detections to coco format
    transformed_det_path = os.path.join(RESULTS_DIR, "coco_transf_detections.json")
    gt_det_path = os.path.join(RESULTS_DIR, "coco_gt_detections.json")
    format_detections(transformed_detections_filepath, transformed_det_path)
    format_detections(gt_detections_filepath, gt_det_path)

    # get accuracy from coco detections
    _, per_class_stats = evaluate_coco_per_class(gt_det_path, transformed_det_path)
    accuracy_blue = per_class_stats[BLUE_CLASS_ID]["AP"]    # BLUE_CLASS_ID must be int
    accuracy_yellow = per_class_stats[YELLOW_CLASS_ID]["AP"]
    return accuracy_blue, accuracy_yellow


def get_next_brightness_config(low_config, high_config, low_ap, high_ap, target_ap=TARGET_AP):
    """Next brightness config via regula falsi: interpolate multiplier so AP → target_ap."""
    lo, hi = low_config["brightness_multiplier"], high_config["brightness_multiplier"]
    denom = high_ap - low_ap
    if abs(denom) < 1e-9:
        mult = (lo + hi) / 2
    else:
        mult = lo + (hi - lo) * (target_ap - low_ap) / denom
        mult = np.clip(mult, min(lo, hi), max(lo, hi))
    return {**low_config, "brightness_multiplier": float(mult)}


def run_iteration(brightness_config, image, image_path, gt_detections_filepath, iteration_number): 
    """
    Run inference for the brightness configuration
    """
    output_dir = os.path.join(RESULTS_DIR, f"iteration_{iteration_number}")
    os.makedirs(output_dir, exist_ok=True)
    success, output_filepath = gen_metamorphic(image, "imgDarkBright", brightness_config, image_path, output_dir)
    if success:
        accuracy_blue, accuracy_yellow = get_accuracy(output_filepath, gt_detections_filepath)  # FIX: pass gt path
        return output_filepath, accuracy_blue, accuracy_yellow
    else:
        raise Exception("Failed to run inference for brightness configuration")


def search_brightness(image, image_path, target_accuracy, gt_detections_filepath):
    """
    Search for the optimal brightness transformation for the image
    """
    # convert to bgr
    image_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    low_param, high_param, low_ap, high_ap = find_starting_range(
        image_bgr, image_path, target_accuracy, gt_detections_filepath
    )

    curr_ap_diff = min(abs(low_ap - TARGET_AP), abs(high_ap - TARGET_AP))
    iterations = 0
    next_config = None
    next_ap = None

    while curr_ap_diff > THRESHOLD_AP and iterations < MAX_ITERATIONS:
        next_config = get_next_brightness_config(low_param, high_param, low_ap, high_ap)
        next_image_filepath, next_ap_blue, next_ap_yellow = run_iteration(next_config, image, image_path, gt_detections_filepath, iterations)  # FIX: pass gt path
        # take the max of the two accuracies
        next_ap = max(next_ap_blue, next_ap_yellow)

        # update bracket
        if next_ap < TARGET_AP:
            low_param, low_ap = next_config, next_ap   # new floor
        else:
            high_param, high_ap = next_config, next_ap  # new ceiling

        curr_ap_diff = abs(next_ap - TARGET_AP)
        iterations += 1

    if next_config is None or next_ap is None:
        if abs(low_ap - TARGET_AP) < abs(high_ap - TARGET_AP):
            return low_param, low_ap
        else:
            return high_param, high_ap

    return next_config, next_ap, next_image_filepath