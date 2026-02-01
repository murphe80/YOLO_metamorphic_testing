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

def locate_and_match_cones_in_bbox(segmented, bbox, min_area=250, DIST_THRES=100):
    """Locate upper/lower cone parts only inside the given bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    h_total, w_total = segmented.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_total, x2), min(h_total, y2)

    # Crop segmentation mask to bbox
    roi = segmented[y1:y2, x1:x2]
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return {}

    # Calculate slope/intercept for LHS of each component
    slopes = [0] * num_labels
    intercepts = [0] * num_labels
    used = set()
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        x, y, w, h = stats[i, cv.CC_STAT_LEFT], stats[i, cv.CC_STAT_TOP], stats[i, cv.CC_STAT_WIDTH], stats[i, cv.CC_STAT_HEIGHT]
        if area < min_area:
            slopes[i] = np.inf
            intercepts[i] = np.inf
            used.add(i)
            continue
        # LHS line
        x1_line, y1_line = x, y
        x2_line, y2_line = x, y+h
        slopes[i] = (y2_line - y1_line) / (x2_line - x1_line + 1e-6)
        intercepts[i] = y1_line - slopes[i] * x1_line

    # Match components
    matched = {}
    for i in range(1, num_labels):
        if i in used:
            continue
        best_match = None
        best_idx = -1
        for j in range(1, num_labels):
            if i == j or j in used:
                continue
            diff = abs(slopes[i] - slopes[j]) + abs(intercepts[i] - intercepts[j])/100
            c1 = np.array(centroids[i])
            c2 = np.array(centroids[j])
            horizontal_dist = abs(c1[0] - c2[0])
            if np.linalg.norm(c1 - c2) < DIST_THRES and horizontal_dist < stats[i, cv.CC_STAT_WIDTH]*2:
                if best_match is None or diff < best_match:
                    best_match = diff
                    best_idx = j
        if best_idx != -1:
            used.add(i)
            used.add(best_idx)
            # Convert stats to proper tuple
            upper_full = tuple(list(stats[i][:4]) + [stats[i][4]])
            lower_full = tuple(list(stats[best_idx][:4]) + [stats[best_idx][4]])
            # Translate coords to full image
            upper_full = (upper_full[0]+x1, upper_full[1]+y1, upper_full[2], upper_full[3], upper_full[4])
            lower_full = (lower_full[0]+x1, lower_full[1]+y1, lower_full[2], lower_full[3], lower_full[4])
            matched[upper_full] = lower_full
            slopes[i] = slopes[best_idx] = np.inf

    return matched

# ------------------------------------------------------------
# Fill cone labels
# ------------------------------------------------------------

def fill_cone_labels(result_image, segmented, matched_cones, mode="avg"):
    for upper, lower in matched_cones.items():
        ux, uy, uw, uh, _ = upper
        lx, ly, lw, lh, _ = lower

        pts = np.array([
            [ux, uy + int(uh*0.9)],
            [lx + int(lw*0.2), ly + int(lh*0.1)],
            [lx + int(lw*0.7), ly],
            [ux + int(uw*1.1), uy + int(uh*0.9)]
        ], np.int32).reshape((-1,1,2))

        if mode == "white":
            color = (255,255,255)
        elif mode == "black":
            color = (0,0,0)
        else:
            roi_upper = result_image[uy+uh//2 : uy+uh, ux:ux+uw]
            roi_lower = result_image[ly : ly+lh//2, lx:lx+lw]
            avg_b, avg_g, avg_r = 0,0,0
            count = 0
            if roi_upper.size>0:
                m = cv.mean(roi_upper)
                avg_b+=m[0]; avg_g+=m[1]; avg_r+=m[2]; count+=1
            if roi_lower.size>0:
                m = cv.mean(roi_lower)
                avg_b+=m[0]; avg_g+=m[1]; avg_r+=m[2]; count+=1
            if count>0:
                color = (int(avg_b/count), int(avg_g/count), int(avg_r/count))
            else:
                color = (0,0,0)

        cv.fillPoly(result_image, [pts], color)
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
