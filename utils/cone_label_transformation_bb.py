#!/usr/bin/env python3
"""
GT-based cone label removal with geometric matching constrained to bounding boxes.

Stacked image format:
- Top half    : original RGB image
- Bottom half : backprojection segmentation mask
"""

import os
import cv2 as cv
import json
import numpy as np
import argparse

# ------------------------------------------------------------
# Ground truth reader
# ------------------------------------------------------------

def read_ground_truth_labels(label_file):
    detections = []
    with open(label_file, "r") as f:
        data = json.load(f)

    class_map = {
        'blue_cone': 0,
        'yellow_cone': 1,
        'orange_cone': 2,
        'big_orange_cone': 2,
        'unknown_cone': 3
    }

    for obj in data.get("objects", []):
        ext = obj.get("points", {}).get("exterior", [])
        if len(ext) < 2:
            continue
        (x1, y1), (x2, y2) = ext
        detections.append({
            "class": class_map.get(obj["classTitle"], 3),
            "class_name": obj["classTitle"],
            "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        })
    return detections

# ------------------------------------------------------------
# Locate cones and match upper/lower components inside a bbox
# ------------------------------------------------------------

def locate_and_match_cones_in_bbox(segmented, bbox, min_area=30, x_cluster_thres=80):
    """Group all components in bbox by horizontal proximity, return top/bottom of each group."""
    x1, y1, x2, y2 = map(int, bbox)
    h_total, w_total = segmented.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_total, x2), min(h_total, y2)

    # Crop segmentation mask to bbox
    roi = segmented[y1:y2, x1:x2]
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)
    
    # Apply morphological closing to fill gaps and connect nearby components
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return {}

    # Collect valid components (filter by area)
    components = []
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        if area >= min_area:
            cx = centroids[i][0]
            cy = centroids[i][1]
            components.append({
                'idx': i,
                'cx': cx,
                'cy': cy,
                'stats': (stats[i][0]+x1, stats[i][1]+y1, stats[i][2], stats[i][3], area)
            })
    
    if len(components) < 2:
        return {}
    
    # Sort by x-coordinate for clustering
    components.sort(key=lambda c: c['cx'])
    
    # Group components by horizontal proximity
    groups = []
    current_group = [components[0]]
    for i in range(1, len(components)):
        if components[i]['cx'] - current_group[-1]['cx'] < x_cluster_thres:
            current_group.append(components[i])
        else:
            groups.append(current_group)
            current_group = [components[i]]
    groups.append(current_group)
    
    # For each group, find topmost and bottommost components
    matched = {}
    for group in groups:
        if len(group) < 2:
            continue
        group.sort(key=lambda c: c['cy'])
        upper = group[0]['stats']
        lower = group[-1]['stats']
        matched[upper] = lower
    
    return matched

# ------------------------------------------------------------
# Fill cone labels
# ------------------------------------------------------------

def fill_cone_labels(result_image, segmented, matched_cones, mode="avg", lower_threshold=80, upper_threshold=20):
    """
    Fill cone labels with colors from segmented image.
    
    Args:
        lower_threshold: Minimum brightness to accept a pixel (discards dark colors)
        upper_threshold: Minimum value to consider a pixel as non-zero
    """
    h_img, w_img = segmented.shape[:2]
    
    for upper, lower in matched_cones.items():
        ux, uy, uw, uh, _ = upper
        lx, ly, lw, lh, _ = lower

        # Define trapezoid points
        pts = np.array([
            [ux, uy + int(uh*0.9)],
            [lx + int(lw*0.3), ly + int(lh*0.05)],
            [lx + int(lw*0.7), ly],
            [ux + int(uw*1.1), uy + int(uh*0.9)]
        ], np.int32).reshape((-1,1,2))

        if mode == "white":
            color = (255,255,255)
            cv.fillPoly(result_image, [pts], color)
        elif mode == "black":
            color = (0,0,0)
            cv.fillPoly(result_image, [pts], color)
        else:  # mode == "avg" - use Nth closest y-direction cone pixel with smoothing
            # Define bounding box limits for search (based on cone components)
            bbox_y_min = min(uy, ly)
            bbox_y_max = max(uy + uh, ly + lh)
            
            # Create a mask for the trapezoid region
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv.fillPoly(mask, [pts], 255)
            
            # Find all pixels in the trapezoid
            y_coords, x_coords = np.where(mask == 255)
            
            # For each pixel in the trapezoid, find the Nth closest non-zero pixel in y-direction
            for py, px in zip(y_coords, x_coords):
                found_pixels = []
                target_count = 10
                
                # Calculate max search distance within bounding box
                max_offset_up = py - bbox_y_min
                max_offset_down = bbox_y_max - py
                max_offset = max(max_offset_up, max_offset_down)
                
                # Search outward in both directions along y-axis within bbox
                for offset in range(max_offset + 1):
                    # Alternate between downward and upward search
                    search_directions = []
                    if offset == 0:
                        search_directions = [0]
                    else:
                        if py + offset <= bbox_y_max:
                            search_directions.append(offset)
                        if py - offset >= bbox_y_min:
                            search_directions.append(-offset)
                    
                    for dy in search_directions:
                        search_y = py + dy
                        seg_pixel = segmented[search_y, px]
                        
                        # Check if pixel is non-zero and not too dark
                        if np.any(seg_pixel > upper_threshold):
                            # Check if pixel is bright enough (not too dark)
                            brightness = np.max(seg_pixel)
                            if brightness >= lower_threshold:
                                found_pixels.append(seg_pixel.copy())
                                
                                # Stop once we have the target count
                                if len(found_pixels) >= target_count:
                                    break
                    
                    if len(found_pixels) >= target_count:
                        break
                
                # Use the Nth pixel if found, otherwise use the last one found, or black
                if len(found_pixels) >= target_count:
                    result_image[py, px] = found_pixels[target_count - 1]
                elif len(found_pixels) > 0:
                    result_image[py, px] = found_pixels[-1]
                else:
                    result_image[py, px] = (0, 0, 0)
            
            # Post-processing: Apply median smoothing to the trapezoid region and surrounding area
            if len(y_coords) > 0:
                # Get the bounding rectangle of the trapezoid with padding
                padding = 10  # pixels to extend beyond trapezoid
                trap_y_min = max(0, int(np.min(y_coords)) - padding)
                trap_y_max = min(h_img - 1, int(np.max(y_coords)) + padding)
                trap_x_min = max(0, int(np.min(x_coords)) - padding)
                trap_x_max = min(w_img - 1, int(np.max(x_coords)) + padding)
                
                # Extract and smooth the expanded trapezoid region
                trap_region = result_image[trap_y_min:trap_y_max+1, trap_x_min:trap_x_max+1].copy()
                if trap_region.shape[0] > 0 and trap_region.shape[1] > 0:
                    smoothed_region = cv.medianBlur(trap_region, 5)  # 5x5 kernel
                    
                    # Apply smoothed values to the entire expanded region
                    result_image[trap_y_min:trap_y_max+1, trap_x_min:trap_x_max+1] = smoothed_region
    
    return result_image

# ------------------------------------------------------------
# Process stacked image
# ------------------------------------------------------------

def process_stacked_image(stacked_path, label_file, output_dir, transform_mode):
    stacked = cv.imread(stacked_path)
    if stacked is None:
        print(f"Could not read {stacked_path}")
        return

    h_total, w, _ = stacked.shape
    h = h_total // 2
    original = stacked[:h, :]
    segmented = stacked[h:, :]

    detections = read_ground_truth_labels(label_file)
    if not detections:
        print(f"No GT labels for {stacked_path}")
        return

    result = original.copy()

    # Process each GT bounding box separately
    for det in detections:
        bbox = det["bbox"]
        matched_cones = locate_and_match_cones_in_bbox(segmented, bbox)
        result = fill_cone_labels(result, segmented, matched_cones, transform_mode)

    out_path = os.path.join(output_dir, os.path.basename(stacked_path))
    cv.imwrite(out_path, result)
    print(f"Saved: {out_path}")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stacked_images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--transform_mode",
        choices=["avg", "white", "black"],
        default="avg"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for label_file in os.listdir(args.labels_dir):
        if not label_file.endswith(".json"):
            continue
        image_name = label_file.replace(".json","")
        stacked_path = os.path.join(args.stacked_images_dir, image_name)
        label_path = os.path.join(args.labels_dir, label_file)
        if os.path.exists(stacked_path):
            process_stacked_image(stacked_path, label_path, args.output_dir, args.transform_mode)

if __name__ == "__main__":
    main()
