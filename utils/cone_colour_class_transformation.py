#!/usr/bin/env python3
"""
Transform cone colors based on their bounding box class labels.

Reads ground truth bounding boxes and applies color transformations
to convert cones from one class to another (e.g., yellow to blue).

Optionally uses segmentation masks to apply transformations only to
cone pixels (non-black pixels in the bottom half of segmented images),
avoiding transformation of background pixels within bounding boxes.

Class mapping:
- 0: blue_cone
- 1: yellow_cone
- 2: orange_cone / big_orange_cone
- 3: unknown_cone
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
    """Read bounding box labels from JSON file."""
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
# Color transformation functions
# ------------------------------------------------------------

def hsv_shift_color(roi, hue_shift, saturation_scale=1.0, value_scale=1.0):
    """
    Shift the hue of a ROI to change cone color.
    
    Args:
        roi: BGR image region
        hue_shift: Amount to shift hue (0-180 in OpenCV HSV)
        saturation_scale: Multiplier for saturation
        value_scale: Multiplier for value/brightness
    """
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV).astype(np.float32)
    
    # Shift hue
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    
    # Scale saturation and value
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_scale, 0, 255)
    
    hsv = hsv.astype(np.uint8)
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

def get_color_transformation(from_class, to_class):
    """
    Get HSV transformation parameters to convert from one cone class to another.
    
    Returns: (hue_shift, saturation_scale, value_scale)
    """
    # Approximate hue values in OpenCV HSV (0-180):
    # Yellow: ~15-30
    # Blue: ~100-120
    # Orange: ~5-15
    
    transformations = {
        # From yellow (class 1)
        (1, 0): (90, 1.0, 1.0),    # yellow -> blue: shift hue by ~90
        (1, 2): (-10, 1.0, 1.0),   # yellow -> orange: shift hue by -10
        
        # From blue (class 0)
        (0, 1): (-90, 1.0, 1.0),   # blue -> yellow: shift hue by -90
        (0, 2): (-100, 1.0, 1.0),  # blue -> orange: shift hue by -100
        
        # From orange (class 2)
        (2, 0): (100, 1.0, 1.0),   # orange -> blue: shift hue by 100
        (2, 1): (10, 1.0, 1.0),    # orange -> yellow: shift hue by 10
    }
    
    return transformations.get((from_class, to_class), (0, 1.0, 1.0))

# ------------------------------------------------------------
# Apply transformations to image
# ------------------------------------------------------------

def transform_cone_colors(image, detections, transformation_map, segmentation_mask=None):
    """
    Apply color transformations to cones based on their class.
    
    Args:
        image: Input BGR image
        detections: List of cone detections with class and bbox
        transformation_map: Dict mapping from_class -> to_class
        segmentation_mask: Optional segmentation mask to identify cone pixels (BGR image)
    """
    result = image.copy()
    
    for det in detections:
        from_class = det["class"]
        bbox = det["bbox"]
        
        # Skip if no transformation defined for this class
        if from_class not in transformation_map:
            continue
        
        to_class = transformation_map[from_class]
        
        # Skip if transforming to same class
        if from_class == to_class:
            continue
        
        # Extract bounding box
        x1, y1, x2, y2 = map(int, bbox)
        h, w = result.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Get ROI
        roi = result[y1:y2, x1:x2]
        
        if roi.size == 0:
            continue
        
        # Get transformation parameters
        hue_shift, sat_scale, val_scale = get_color_transformation(from_class, to_class)
        
        # Apply transformation
        transformed_roi = hsv_shift_color(roi, hue_shift, sat_scale, val_scale)
        
        # If segmentation mask is provided, only apply to non-black pixels
        if segmentation_mask is not None:
            mask_roi = segmentation_mask[y1:y2, x1:x2]
            
            # Create mask for non-black pixels (all channels > 0)
            non_black_mask = np.any(mask_roi != 0, axis=2)
            
            # Only apply transformation where mask is non-black
            result[y1:y2, x1:x2][non_black_mask] = transformed_roi[non_black_mask]
            
            pixels_transformed = np.sum(non_black_mask)
            print(f"  Transformed cone from class {from_class} to {to_class} at bbox [{x1},{y1},{x2},{y2}] ({pixels_transformed} pixels)")
        else:
            # Apply to entire ROI if no mask
            result[y1:y2, x1:x2] = transformed_roi
            print(f"  Transformed cone from class {from_class} to {to_class} at bbox [{x1},{y1},{x2},{y2}]")
    
    return result

# ------------------------------------------------------------
# Process images
# ------------------------------------------------------------

def process_image(image_path, label_file, output_dir, transformation_map, segmented_dir=None):
    """Process a single image with cone color transformations."""
    image = cv.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}")
        return
    
    detections = read_ground_truth_labels(label_file)
    if not detections:
        print(f"No GT labels for {image_path}")
        return
    
    # Load segmentation mask if directory is provided
    segmentation_mask = None
    if segmented_dir is not None:
        seg_path = os.path.join(segmented_dir, os.path.basename(image_path))
        if os.path.exists(seg_path):
            seg_image = cv.imread(seg_path)
            if seg_image is not None:
                # Extract bottom half of segmented image as mask
                height = seg_image.shape[0]
                segmentation_mask = seg_image[height//2:, :]
                print(f"  Using segmentation mask from {seg_path}")
            else:
                print(f"  Warning: Could not read segmentation image {seg_path}")
        else:
            print(f"  Warning: Segmentation image not found at {seg_path}")
    
    result = transform_cone_colors(image, detections, transformation_map, segmentation_mask)
    
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv.imwrite(out_path, result)
    print(f"Saved: {out_path}")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transform cone colors based on bounding box class labels"
    )
    parser.add_argument("--images_dir", required=True, help="Directory with input images")
    parser.add_argument("--labels_dir", required=True, help="Directory with bounding box JSON labels")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--segmented_dir", help="Optional directory with segmented images (for masking)")
    parser.add_argument(
        "--transformation",
        choices=["yellow_to_blue", "blue_to_yellow", "yellow_to_orange", 
                 "orange_to_blue", "blue_to_orange", "orange_to_yellow"],
        required=True,
        help="Type of color transformation to apply"
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define transformation map based on chosen transformation
    transformation_maps = {
        "yellow_to_blue": {1: 0},      # Transform yellow (1) to blue (0)
        "blue_to_yellow": {0: 1},      # Transform blue (0) to yellow (1)
        "yellow_to_orange": {1: 2},    # Transform yellow (1) to orange (2)
        "orange_to_blue": {2: 0},      # Transform orange (2) to blue (0)
        "blue_to_orange": {0: 2},      # Transform blue (0) to orange (2)
        "orange_to_yellow": {2: 1},    # Transform orange (2) to yellow (1)
    }
    
    transformation_map = transformation_maps[args.transformation]
    
    print(f"Applying transformation: {args.transformation}")
    print(f"Transformation map: {transformation_map}")
    if args.segmented_dir:
        print(f"Using segmentation masks from: {args.segmented_dir}")
    
    # Process all label files
    for label_file in os.listdir(args.labels_dir):
        if not label_file.endswith(".json"):
            continue
        
        # Find corresponding image
        image_name = label_file.replace(".json", "")
        image_path = os.path.join(args.images_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for {label_file}")
            continue
        
        label_path = os.path.join(args.labels_dir, label_file)
        print(f"\nProcessing {image_name}...")
        process_image(image_path, label_path, args.output_dir, transformation_map, args.segmented_dir)

if __name__ == "__main__":
    main()

