#!/usr/bin/env python3
"""
Extract cone regions using ground truth labels and histogram backprojection.
Reads JSON annotation files and applies histogram backprojection to each labeled bounding box region.
This bypasses YOLO inference entirely and uses ground truth annotations.
"""

import numpy as np
import cv2 as cv
import os 
import json
import argparse
from pathlib import Path

def load_color_histograms(hist_samples_dir):
    """Load and compute histograms for different cone colors."""
    histograms = {}
    
    # Load sample images for each color
    color_samples = {
        'blue': 'blue_cones_samples2.PNG',
        'yellow': 'yellow_cones_samples2.png', 
        'orange': 'orange_cones_samples2.PNG'
    }
    
    for color, filename in color_samples.items():
        sample_path = os.path.join(hist_samples_dir, filename)
        if not os.path.exists(sample_path):
            print(f"Warning: Sample file {sample_path} not found, skipping {color} cones")
            continue
            
        roi = cv.imread(sample_path)
        if roi is None:
            print(f"Warning: Could not read {sample_path}")
            continue
            
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        
        # Calculate histogram
        hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
        histograms[color] = hist
        
    return histograms

def read_ground_truth_labels(label_file):
    """Read ground truth JSON annotation file and return list of cone detections."""
    detections = []
    
    if not os.path.exists(label_file):
        return detections
        
    try:
        with open(label_file, 'r') as f:
            data = json.load(f)
            
        # Extract objects from JSON
        if 'objects' in data:
            for obj in data['objects']:
                if 'classTitle' in obj and 'points' in obj:
                    class_title = obj['classTitle']
                    
                    # Extract bounding box coordinates
                    if 'exterior' in obj['points'] and len(obj['points']['exterior']) >= 2:
                        x1, y1 = obj['points']['exterior'][0]
                        x2, y2 = obj['points']['exterior'][1]
                        
                        # Ensure proper bounding box format (x1,y1 = top-left, x2,y2 = bottom-right)
                        bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                        
                        # Map class titles to indices
                        class_map = {
                            'blue_cone': 0,
                            'yellow_cone': 1, 
                            'orange_cone': 2,
                            'big_orange_cone': 2,  # Treat big orange same as orange
                            'unknown_cone': 3
                        }
                        
                        class_id = class_map.get(class_title, 3)  # Default to unknown
                        
                        detection = {
                            'class': class_id,
                            'class_name': class_title,
                            'confidence': 1.0,  # Ground truth has perfect confidence
                            'bbox': bbox
                        }
                        detections.append(detection)
                        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading {label_file}: {e}")
        return []
                        
    return detections

def apply_backprojection_to_bbox(image, bbox, class_name, histograms):
    """Apply histogram backprojection to a specific bounding box region."""
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Ensure bounding box is within image bounds
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1)) 
    x2 = max(x1+1, min(x2, w))
    y2 = max(y1+1, min(y2, h))
    
    # Extract region of interest
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        
    # Convert to HSV
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    
    # Apply backprojection based on cone class
    cone_color = None
    if 'blue' in class_name.lower():
        cone_color = 'blue'
    elif 'yellow' in class_name.lower():
        cone_color = 'yellow'  
    elif 'orange' in class_name.lower():
        cone_color = 'orange'
    
    # Apply backprojection for the specific color, or all colors if unknown
    if cone_color and cone_color in histograms:
        combined = cv.calcBackProject([hsv_roi], [0, 1], histograms[cone_color], [0, 180, 0, 256], 1)
    else:
        # If color is unknown or not available, try all colors
        backproject_results = []
        for color, hist in histograms.items():
            dst = cv.calcBackProject([hsv_roi], [0, 1], hist, [0, 180, 0, 256], 1)
            backproject_results.append(dst)
        
        if backproject_results:
            combined = backproject_results[0]
            for dst in backproject_results[1:]:
                combined = cv.max(combined, dst)
        else:
            combined = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    
    # Apply morphological operations
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    cv.filter2D(combined, -1, disc, combined)
    
    # Threshold
    ret, thresh = cv.threshold(combined, 5, 255, 0)
    
    # Morphological operations to clean up
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    thresh = cv.morphologyEx(thresh, cv.MORPH_ERODE, kernel)
    thresh = cv.morphologyEx(thresh, cv.MORPH_DILATE, kernel)
    
    return thresh

def process_image_with_gt_labels(image_path, label_file, histograms, output_dir):
    """Process a single image with its ground truth labels."""
    # Load image
    image = cv.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
        
    # Read ground truth labels
    detections = read_ground_truth_labels(label_file)
    if not detections:
        print(f"No ground truth labels found for {image_path}")
        return
        
    # Create output images
    result_image = image.copy()
    segmentation_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    print(f"Processing {len(detections)} ground truth labels for {os.path.basename(image_path)}")
    
    # Color map for visualization
    color_map = {
        0: (255, 0, 0),    # Blue cone -> Blue
        1: (0, 255, 255),  # Yellow cone -> Yellow  
        2: (0, 165, 255),  # Orange cone -> Orange
        3: (128, 128, 128) # Unknown -> Gray
    }
    
    # Process each detection
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        class_id = detection['class']
        class_name = detection['class_name']
        
        # Apply histogram backprojection to this bounding box
        bbox_mask = apply_backprojection_to_bbox(image, bbox, class_name, histograms)
        
        # Add to overall segmentation mask
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        if bbox_mask.size > 0:
            segmentation_mask[y1:y2, x1:x2] = cv.bitwise_or(
                segmentation_mask[y1:y2, x1:x2], bbox_mask)
        
        # Draw bounding box on result image with color coding
        color = color_map.get(class_id, (255, 255, 255))
        cv.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        cv.putText(result_image, f'{class_name}', 
                   (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Apply segmentation mask to original image
    segmentation_mask_3ch = cv.merge((segmentation_mask, segmentation_mask, segmentation_mask))
    segmented_image = cv.bitwise_and(image, segmentation_mask_3ch)
    
    # Create combined output with labels
    font = cv.FONT_HERSHEY_SIMPLEX
    label_img = np.zeros((60, image.shape[1], 3), dtype=np.uint8)
    cv.putText(label_img, 'Original', (10, 25), font, 0.7, (255, 255, 255), 2)
    
    bbox_img = np.zeros((60, image.shape[1], 3), dtype=np.uint8) 
    cv.putText(bbox_img, f'Ground Truth Bounding Boxes ({len(detections)} cones)', (10, 25), font, 0.7, (255, 255, 255), 2)
    
    seg_img = np.zeros((60, image.shape[1], 3), dtype=np.uint8)
    cv.putText(seg_img, 'Histogram Backprojection Result', (10, 25), font, 0.7, (255, 255, 255), 2)
    
    result_stack = np.vstack((
        label_img,
        image,           # Original
        bbox_img,
        result_image,    # With bounding boxes  
        seg_img,
        segmented_image  # Segmented regions
    ))
    
    # Save result
    output_filename = os.path.join(output_dir, f"gt_segmented_{os.path.basename(image_path)}")
    cv.imwrite(output_filename, result_stack)
    print(f"Saved ground truth segmentation result to {output_filename}")
    
    # Also save individual components
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv.imwrite(os.path.join(output_dir, f"gt_bbox_{base_name}.png"), result_image)
    cv.imwrite(os.path.join(output_dir, f"gt_mask_{base_name}.png"), segmentation_mask)

def main():
    parser = argparse.ArgumentParser(description="Extract cone regions using ground truth labels and histogram backprojection")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--labels_dir", type=str, required=True, help="Directory containing ground truth JSON label files") 
    parser.add_argument("--hist_samples_dir", type=str, required=True, help="Directory containing color histogram sample images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load color histograms
    print("Loading color histograms...")
    histograms = load_color_histograms(args.hist_samples_dir)
    if not histograms:
        print("No color histograms loaded. Please check histogram sample files.")
        return
    print(f"Loaded histograms for colors: {list(histograms.keys())}")
    
    # Process all images that have corresponding labels
    image_extensions = ('.jpg', '.jpeg', '.png')
    label_files = [f for f in os.listdir(args.labels_dir) if f.endswith('.json')]
    available_images = set(os.listdir(args.images_dir))
    
    processed_count = 0
    for label_file in label_files:
        # Extract base filename from label file (remove .json extension)
        # Label files are named like "amz_03812.png.json", so we need to remove ".json"
        image_filename = label_file.replace('.json', '')
        
        # Check if the corresponding image file exists
        if image_filename in available_images:
            image_path = os.path.join(args.images_dir, image_filename)
            label_path = os.path.join(args.labels_dir, label_file)
            
            process_image_with_gt_labels(image_path, label_path, histograms, args.output_dir)
            processed_count += 1
        else:
            print(f"No corresponding image found for label file {label_file} (looking for {image_filename})")
    
    print(f"\nProcessed {processed_count} images with ground truth labels")

if __name__ == "__main__":
    main()