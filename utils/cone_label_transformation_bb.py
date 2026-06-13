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

def locate_and_match_cones_in_bbox(segmented, bbox, class_name="", min_area=30, x_cluster_thres=80):
    """Find and match components in bbox. For big_orange_cone, handles 3 components (2 labels + body).
    Returns dict of matched pairs: {(upper_stats, upper_centroid): (lower_stats, lower_centroid), ...}
    """
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
                'area': area,
                'stats': (stats[i][0]+x1, stats[i][1]+y1, stats[i][2], stats[i][3], area)
            })

    if len(components) < 2:
        return {}

    # Sort by area (largest first)
    components.sort(key=lambda c: c['area'], reverse=True)

    # Check if this is a big orange cone (expects 3 components: 2 labels + body)
    print(class_name)
    is_big_orange = 'large_orange_cone' in class_name.lower()

    if is_big_orange and len(components) >= 3:
        # Take the 3 largest components
        largest_three = components[:3]

        # Validate areas - all should be substantial relative to the largest
        largest_area = largest_three[0]['area']
        for comp in largest_three[1:]:
            if comp['area'] / largest_area < 0.20:
                # One component is too small, fall back to 2-component matching
                is_big_orange = False
                break

        if is_big_orange:
            # Sort by y-coordinate to get top, middle, bottom
            largest_three.sort(key=lambda c: c['cy'])

            # Create two trapezoid pairs: (top->middle) and (middle->bottom)
            matched = {}

            for i in range(2):  # Two pairs for big orange cone
                upper_comp = largest_three[i]
                lower_comp = largest_three[i + 1]

                upper = (upper_comp['stats'], (upper_comp['cx'] + x1, upper_comp['cy'] + y1))
                lower = (lower_comp['stats'], (lower_comp['cx'] + x1, lower_comp['cy'] + y1))
                matched[upper] = lower

            return matched

    # Standard 2-component matching (regular cones or fallback)
    if len(components) < 2:
        return {}

    largest_two = components[:2]

    # Validate that the smaller component is at least 20% of the larger component's area
    larger_area = largest_two[0]['area']
    smaller_area = largest_two[1]['area']
    area_ratio = smaller_area / larger_area

    if area_ratio < 0.20:
        # Second component is too small - likely noise, not a label
        return {}

    # Sort these two by y-coordinate to determine upper and lower
    largest_two.sort(key=lambda c: c['cy'])

    # Include centroids for more robust trapezoid calculation
    upper = (largest_two[0]['stats'], (largest_two[0]['cx'] + x1, largest_two[0]['cy'] + y1))
    lower = (largest_two[1]['stats'], (largest_two[1]['cx'] + x1, largest_two[1]['cy'] + y1))

    return {upper: lower}

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
        # Unpack stats and centroids
        upper_stats, upper_centroid = upper
        lower_stats, lower_centroid = lower

        ux, uy, uw, uh, _ = upper_stats
        lx, ly, lw, lh, _ = lower_stats
        ucx, ucy = upper_centroid  # Upper component centroid
        lcx, lcy = lower_centroid  # Lower component centroid

        # Define trapezoid points
        pts = np.array([
            [ux, uy + int(uh*0.9)],                   # top-left
            [lx + int(lw*0.28), ly + int(lh*0.08)],   # bottom-left
            [lx + int(lw*0.75), ly + int(lh*0.08)],   # bottom-right
            [ux + int(uw*1.08), uy + int(uh*0.92)]     # top-right
        ], np.int32).reshape((-1,1,2))

        # Validate cone shape: top should be narrower than bottom
        top_width = pts[3][0][0] - pts[0][0][0]  # top-right x - top-left x
        bottom_width = pts[2][0][0] - pts[1][0][0]  # bottom-right x - bottom-left x

        if top_width > (bottom_width * 0.8):
            # Invalid cone shape - correct it by expanding the bottom
            # Preserve top width and make bottom 125% wider
            target_bottom_width = int(top_width * 1.25)
            top_center_x = (pts[3][0][0] + pts[0][0][0]) // 2

            pts[1][0][0] = top_center_x - target_bottom_width // 2  # bottom-left
            pts[2][0][0] = top_center_x + target_bottom_width // 2  # bottom-right

        # Validate trapezoid height: should be at least 100% of upper component height
        top_y = pts[0][0][1]  # y-coordinate of top edge (pts[0] and pts[3] have same y)
        bottom_y = pts[1][0][1]  # y-coordinate of bottom edge (pts[1] and pts[2] have same y)
        trapezoid_height = bottom_y - top_y
        min_height = uh  # Upper component height

        if trapezoid_height < min_height:
            # Extend bottom edge downward to meet minimum height requirement
            target_bottom_y = top_y + min_height
            pts[1][0][1] = target_bottom_y  # bottom-left y
            pts[2][0][1] = target_bottom_y  # bottom-right y

        if mode == "white":
            # Realistic white: preserve lighting by using luminance-based mapping
            # Create a mask for the trapezoid region
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv.fillPoly(mask, [pts], 255)

            # Extract original colors in trapezoid region
            y_coords, x_coords = np.where(mask == 255)

            # Calculate median color in trapezoid (represents typical label color)
            trapezoid_pixels = result_image[y_coords, x_coords]
            median_color = np.median(trapezoid_pixels, axis=0)

            # Color distance threshold - only transform pixels similar to median color
            color_threshold = 80  # Euclidean distance in BGR space

            for py, px in zip(y_coords, x_coords):
                original_pixel = result_image[py, px].astype(np.float32)

                # Calculate color distance from median
                color_distance = np.sqrt(np.sum((original_pixel - median_color) ** 2))

                # Only transform if pixel is similar to the label color
                if color_distance <= color_threshold:
                    # Calculate luminance (perceived brightness)
                    luminance = 0.299 * original_pixel[2] + 0.587 * original_pixel[1] + 0.114 * original_pixel[0]

                    # Map to white range: preserve relative brightness
                    # Dark areas (shadows) -> light gray (200-220)
                    # Bright areas (highlights) -> bright white (240-255)
                    min_white = 200
                    max_white = 255
                    normalized_lum = luminance / 255.0
                    white_value = int(min_white + normalized_lum * (max_white - min_white))
                    result_image[py, px] = [white_value - 5, white_value, white_value]  # BGR: slightly blue-tinted white

        elif mode == "black":
            # Realistic black: preserve lighting by using luminance-based mapping
            # Create a mask for the trapezoid region
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv.fillPoly(mask, [pts], 255)

            # Extract original colors in trapezoid region
            y_coords, x_coords = np.where(mask == 255)

            # Calculate median color in trapezoid (represents typical label color)
            trapezoid_pixels = result_image[y_coords, x_coords]
            median_color = np.median(trapezoid_pixels, axis=0)

            # Color distance threshold - only transform pixels similar to median color
            color_threshold = 80  # Euclidean distance in BGR space

            for py, px in zip(y_coords, x_coords):
                original_pixel = result_image[py, px].astype(np.float32)

                # Calculate color distance from median
                color_distance = np.sqrt(np.sum((original_pixel - median_color) ** 2))

                # Only transform if pixel is similar to the label color
                if color_distance <= color_threshold:
                    # Calculate luminance (perceived brightness)
                    luminance = 0.299 * original_pixel[2] + 0.587 * original_pixel[1] + 0.114 * original_pixel[0]

                    # Map to black range: preserve relative brightness
                    # Dark areas (shadows) -> pure black (0-20)
                    # Bright areas (highlights/edges) -> dark gray (20-60)
                    min_black = 0
                    max_black = 60
                    normalized_lum = luminance / 255.0
                    black_value = int(min_black + normalized_lum * (max_black - min_black))

                    # Apply slightly to all channels for realistic dark surface
                    result_image[py, px] = [black_value, black_value, black_value]
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
        class_name = det["class_name"]
        matched_cones = locate_and_match_cones_in_bbox(segmented, bbox, class_name)
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
