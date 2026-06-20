#!/usr/bin/env python3
"""
Transform cone colors based on their bounding box class labels.

Reads ground truth bounding boxes and applies color transformations
to convert cones from one class to another (e.g., yellow to blue).

Optionally uses segmentation masks to apply transformations only to
cone pixels (non-black pixels in the bottom half of segmented images),
avoiding transformation of background pixels within bounding boxes.

Supported label formats:
1. TXT format (auto-detects between two sub-formats):
   a) YOLO normalized: class_id center_x center_y width height (values 0-1)
      - File naming: image_name.txt (matches image_name.jpg/png)
      - Example: 0 0.786038 0.238587 0.005919 0.020652 (yellow cone)
   
   b) Absolute pixel: class_id x1 y1 x2 y2
      - File naming: image_name.txt
      - Example: 0 520 375 565 445 (yellow cone)

2. JSON format: Supervisely-style annotation
   - File naming: image_name.ext.json (e.g., image.png.json)

Class mapping:
- 0: yellow_cone
- 1: orange_cone
- 2: unknown_cone
- 3: blue_cone

Format detection: TXT files with all coordinate values <= 1.0 are treated as YOLO normalized format.
"""

import os
import cv2 as cv
import json
import numpy as np
import argparse
from metamorphic_rules.instantiate_rules import instantiate_rule

# ------------------------------------------------------------
# Ground truth reader
# ------------------------------------------------------------

def read_ground_truth_labels_txt(label_file, image_shape=None):
    """Read bounding box labels from TXT file.
    
    Supports two TXT formats:
    1. Absolute pixel format: class_id x1 y1 x2 y2
    2. YOLO normalized format: class_id center_x center_y width height (values 0-1)
    
    Auto-detects format based on coordinate values.
    """
    detections = []
    
    class_names = {
        0: 'yellow_cone',
        1: 'orange_cone',
        2: 'unknown_cone',
        3: 'blue_cone'
    }
    
    try:
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                try:
                    class_id = int(parts[0])
                    val1, val2, val3, val4 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    
                    # Auto-detect format: if all values <= 1.0, assume YOLO normalized format
                    if val1 <= 1.0 and val2 <= 1.0 and val3 <= 1.0 and val4 <= 1.0:
                        # YOLO format: class_id center_x center_y width height (normalized)
                        if image_shape is None:
                            print("Warning: YOLO format detected but image dimensions not provided")
                            continue
                        
                        img_height, img_width = image_shape
                        center_x, center_y, width, height = val1, val2, val3, val4
                        
                        # Convert to absolute pixels
                        center_x_abs = center_x * img_width
                        center_y_abs = center_y * img_height
                        width_abs = width * img_width
                        height_abs = height * img_height
                        
                        # Convert to x1,y1,x2,y2
                        x1 = center_x_abs - width_abs / 2
                        y1 = center_y_abs - height_abs / 2
                        x2 = center_x_abs + width_abs / 2
                        y2 = center_y_abs + height_abs / 2
                    else:
                        # Absolute pixel format
                        x1, y1, x2, y2 = val1, val2, val3, val4
                    
                    detections.append({
                        "class": class_id,
                        "class_name": class_names.get(class_id, 'unknown_cone'),
                        "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    })
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error reading {label_file}: {e}")
    
    return detections


def read_ground_truth_labels_json(label_file):
    """Read bounding box labels from JSON file."""
    detections = []
    with open(label_file, "r") as f:
        data = json.load(f)

    class_map = {
        'blue_cone': 3,
        'yellow_cone': 0,
        'orange_cone': 1,
        'big_orange_cone': 1,
        'unknown_cone': 2
    }

    for obj in data.get("objects", []):
        ext = obj.get("points", {}).get("exterior", [])
        if len(ext) < 2:
            continue
        (x1, y1), (x2, y2) = ext
        detections.append({
            "class": class_map.get(obj["classTitle"], 2),
            "class_name": obj["classTitle"],
            "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        })
    return detections


def read_ground_truth_labels(label_file, image_shape=None):
    """Read bounding box labels from either TXT or JSON format.
    
    Args:
        label_file: Path to label file
        image_shape: Tuple of (height, width) needed for YOLO format TXT files
    """
    if label_file.endswith('.txt'):
        return read_ground_truth_labels_txt(label_file, image_shape)
    elif label_file.endswith('.json'):
        return read_ground_truth_labels_json(label_file)
    else:
        print(f"Warning: Unknown label file format: {label_file}")
        return []

# ------------------------------------------------------------
# Color transformation functions
# ------------------------------------------------------------

def apply_hue_transformation(roi, hue_value):
    """
    Apply color channel transformation rule to a ROI.
    
    Args:
        roi: BGR image region
        hue_value: Hue value to apply to the image
    
    Returns:
        Transformed BGR image
    """
    # Convert BGR to RGB for the rule
    roi_rgb = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
    
    # Change hue of the image
    transformed_hsv = cv.cvtColor(roi_rgb, cv.COLOR_RGB2HSV)
    transformed_hsv[:, :, 0] = hue_value
    
    # Convert back to BGR
    return cv.cvtColor(transformed_hsv, cv.COLOR_HSV2BGR)


def get_hue_transformation(from_class, to_class):
    """
    Get hue value to convert from one cone class to another.
    
    Args:
        from_class: Source cone class (0=yellow, 1=orange, 2=unknown, 3=blue)
        to_class: Target cone class
    
    Returns:
        Hue value
    
    Color characteristics:
    - Yellow: Hue value 27
    - Blue: Hue value 112
    - Orange: Hue value 14
    """
    transformations = {
        # From yellow (class 0)
        (0, 3): 112,    # yellow -> blue
        (0, 1): 14,    # yellow -> orange
        
        # From blue (class 3)
        (3, 0): 27,    # blue -> yellow
        (3, 1): 14,   # blue -> orange
        
        # From orange (class 1)
        (1, 3): 112,    # orange -> blue
        (1, 0): 27,    # orange -> yellow
    }
    
    # Default: no transformation
    return transformations.get((from_class, to_class), 0.0)

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
    
    Returns:
        tuple: (result_image, transformations_applied)
            - result_image: Transformed image
            - transformations_applied: Number of cones that were transformed
    """
    result = image.copy()
    transformations_applied = 0
    
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
        
        # apply median blur to reduce noise
        roi = cv.medianBlur(roi, 15)
        
        # Get transformation parameters
        transformation = get_hue_transformation(from_class, to_class)
        
        # Check if this is a direct color replacement
        transformed_roi = apply_hue_transformation(roi, transformation)
        
        # If segmentation mask is provided, only apply to non-black pixels
        if segmentation_mask is not None:
            # Check if bbox is within mask bounds
            mask_h, mask_w = segmentation_mask.shape[:2]
            if y1 >= mask_h or x1 >= mask_w:
                print(f"  ⚠ Bbox [{x1},{y1},{x2},{y2}] outside mask bounds ({mask_w}x{mask_h}) - skipping")
                continue
            
            # Clip bbox to mask bounds
            mask_y2 = min(y2, mask_h)
            mask_x2 = min(x2, mask_w)
            
            if mask_y2 <= y1 or mask_x2 <= x1:
                print(f"  ⚠ Bbox [{x1},{y1},{x2},{y2}] invalid after clipping - skipping")
                continue
            
            # Extract the mask region corresponding to the ROI
            mask_roi = segmentation_mask[y1:mask_y2, x1:mask_x2]
            
            # Adjust transformed_roi size if clipped
            if mask_y2 < y2 or mask_x2 < x2:
                transformed_roi = transformed_roi[:mask_y2-y1, :mask_x2-x1]
            
            # Create mask for non-black pixels (cone pixels)
            # Check each pixel: True if ANY channel is non-zero
            non_black_mask = np.any(mask_roi != 0, axis=2)
            
            # Only apply transformation if there are non-black pixels
            non_black_count = np.sum(non_black_mask)
            if non_black_count == 0:
                print(f"  ⚠ No cone pixels in mask at bbox [{x1},{y1},{x2},{y2}] - skipping")
                continue

            result_roi = result[y1:mask_y2, x1:mask_x2]
            result_roi[non_black_mask] = transformed_roi[non_black_mask]
            
            # Apply morphological closing to reduce noise
            # Closing = dilation followed by erosion (fills small holes)
            kernel_size = 5
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Create 3-channel mask for processing
            mask_3ch = np.stack([non_black_mask, non_black_mask, non_black_mask], axis=2).astype(np.uint8) * 255
            
            # Apply closing to the mask
            closed_mask = cv.morphologyEx(mask_3ch[:, :, 0], cv.MORPH_CLOSE, kernel)
            closed_mask_bool = closed_mask > 0
            
            # Reapply with closed mask to reduce noise
            result_roi[closed_mask_bool] = transformed_roi[closed_mask_bool]
            
            result[y1:mask_y2, x1:mask_x2] = result_roi
            
            total_pixels = (mask_y2 - y1) * (mask_x2 - x1)
            print(f"  ✓ Transformed cone from class {from_class} to {to_class} at bbox [{x1},{y1},{x2},{y2}] ({non_black_count}/{total_pixels} pixels)")
        else:
            # No segmentation mask, transform entire ROI
            result[y1:y2, x1:x2] = transformed_roi
            print(f"  ⚠ Transformed entire bbox from class {from_class} to {to_class} at [{x1},{y1},{x2},{y2}] (no mask)")
        
        transformations_applied += 1
    
    return result, transformations_applied

# ------------------------------------------------------------
# Process images
# ------------------------------------------------------------

def process_image(image_path, label_file, output_dir, transformation_map, segmented_dir=None):
    """Process a single image with cone color transformations."""
    # Load segmentation mask if directory is provided
    segmentation_mask = None
    if segmented_dir is not None:
        seg_path = os.path.join(segmented_dir, os.path.basename(image_path))
        if os.path.exists(seg_path):
            seg_image = cv.imread(seg_path)
            if seg_image is not None:
                # Segmented image contains: top half = color image, bottom half = segmentation mask
                height = seg_image.shape[0]
                # Extract top half as the image to transform
                image = seg_image[:height//2, :]
                # Extract bottom half as mask
                segmentation_mask = seg_image[height//2:, :]
                print(f"  ✓ Using top half of segmented image as input: {image.shape}")
                print(f"  ✓ Using bottom half as segmentation mask: {segmentation_mask.shape}")
            else:
                print(f"  ✗ Warning: Could not read segmentation image {seg_path}")
                return
        else:
            print(f"  ✗ Warning: Segmentation image not found at {seg_path}")
            return
    else:
        # Fallback: read from original images directory (no segmentation mask)
        image = cv.imread(image_path)
        if image is None:
            print(f"Could not read {image_path}")
            return
        print("  ⚠ Warning: No segmentation directory provided - will transform entire bounding boxes")

    # Get image dimensions for YOLO format support
    image_shape = image.shape[:2]  # (height, width)

    detections = read_ground_truth_labels(label_file, image_shape)
    if not detections:
        print(f"No GT labels for {image_path}")
        return
    
    result, transformations_applied = transform_cone_colors(image, detections, transformation_map, segmentation_mask)
    
    # Skip saving if no transformations were applied
    if transformations_applied == 0:
        print("  No transformations applied - skipping image")
        return
    
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv.imwrite(out_path, result)
    print(f"Saved: {out_path} ({transformations_applied} cones transformed)")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transform cone colors based on bounding box class labels"
    )
    parser.add_argument("--images_dir", required=True, help="Directory with original input images")
    parser.add_argument("--labels_dir", required=True, help="Directory with bounding box labels (TXT or JSON)")
    parser.add_argument("--output_dir", required=True, help="Output directory for transformed images")
    parser.add_argument("--segmented_dir", required=True, help="Directory with segmented images (used to determine which images to process)")
    parser.add_argument(
        "--transformation",
        choices=["yellow_to_blue", "blue_to_yellow", "yellow_to_orange",
                 "orange_to_blue", "blue_to_orange", "orange_to_yellow",
                 "swap_sides"],
        required=True,
        help="Type of color transformation to apply"
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define transformation map based on chosen transformation
    # Class mapping: 0=yellow, 1=orange, 2=unknown, 3=blue
    transformation_maps = {
        "yellow_to_blue": {0: 3},      # Transform yellow (0) to blue (3)
        "blue_to_yellow": {3: 0},      # Transform blue (3) to yellow (0)
        "yellow_to_orange": {0: 1},    # Transform yellow (0) to orange (1)
        "orange_to_blue": {1: 3},      # Transform orange (1) to blue (3)
        "blue_to_orange": {3: 1},      # Transform blue (3) to orange (1)
        "orange_to_yellow": {1: 0},    # Transform orange (1) to yellow (0)
        "swap_sides": {0: 3, 3: 0},    # Swap yellow (0) ↔ blue (3)
    }
    
    transformation_map = transformation_maps[args.transformation]
    
    print(f"Applying transformation: {args.transformation}")
    print(f"Transformation map: {transformation_map}")
    
    if not args.segmented_dir:
        print("Error: --segmented_dir is required for processing")
        return
    
    print(f"Using segmentation masks from: {args.segmented_dir}")
    
    # Process all segmented images
    segmented_files = [f for f in os.listdir(args.segmented_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not segmented_files:
        print(f"Warning: No image files found in {args.segmented_dir}")
        return
    
    print(f"Found {len(segmented_files)} segmented images to process")
    
    for segmented_file in segmented_files:
        # Get base name without extension
        base_name = os.path.splitext(segmented_file)[0]
        
        # Find corresponding label file (could be .txt or .json)
        label_path = None
        # Try .txt first
        txt_label = os.path.join(args.labels_dir, base_name + '.txt')
        if os.path.exists(txt_label):
            label_path = txt_label
        else:
            # Try .json (format: base_name.ext.json)
            json_label = os.path.join(args.labels_dir, segmented_file + '.json')
            if os.path.exists(json_label):
                label_path = json_label
        
        if not label_path:
            print(f"Warning: No label file found for {segmented_file}")
            continue
        
        # Find corresponding original image in images_dir
        image_path = os.path.join(args.images_dir, segmented_file)
        
        if not os.path.exists(image_path):
            print(f"Warning: Original image not found for {segmented_file}: {image_path}")
            continue
        
        print(f"\nProcessing {segmented_file}...")
        process_image(image_path, label_path, args.output_dir, transformation_map, args.segmented_dir)

if __name__ == "__main__":
    main()

