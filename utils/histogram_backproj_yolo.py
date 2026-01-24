#!/usr/bin/env python3
"""
Extract cone regions using YOLO detections and histogram backprojection.
Reads YOLO detection files and applies histogram backprojection to each bounding box region.
"""

import numpy as np
import cv2 as cv
import os 
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

def read_yolo_detections(detection_file):
    """Read YOLO detection file and return list of detections."""
    detections = []
    
    if not os.path.exists(detection_file):
        return detections
        
    with open(detection_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) >= 6:
                detection = {
                    'class': int(parts[0]),
                    'confidence': float(parts[1]),
                    'bbox': [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
                }
                detections.append(detection)
                
    return detections

def apply_backprojection_to_bbox(image, bbox, histograms):
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
    
    # Apply backprojection for each color
    backproject_results = []
    for color, hist in histograms.items():
        dst = cv.calcBackProject([hsv_roi], [0, 1], hist, [0, 180, 0, 256], 1)
        backproject_results.append(dst)
    
    # Combine results from all colors
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
    ret, thresh = cv.threshold(combined, 20, 255, 0)
    
    # Morphological operations to clean up
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    thresh = cv.morphologyEx(thresh, cv.MORPH_ERODE, kernel)
    thresh = cv.morphologyEx(thresh, cv.MORPH_DILATE, kernel)
    
    return thresh

def process_image_with_detections(image_path, detection_file, histograms, output_dir):
    """Process a single image with its detections."""
    # Load image
    image = cv.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
        
    # Read detections
    detections = read_yolo_detections(detection_file)
    if not detections:
        print(f"No detections found for {image_path}")
        return
        
    # Create output images
    result_image = image.copy()
    segmentation_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    print(f"Processing {len(detections)} detections for {os.path.basename(image_path)}")
    
    # Process each detection
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_id = detection['class']
        
        # Apply histogram backprojection to this bounding box
        bbox_mask = apply_backprojection_to_bbox(image, bbox, histograms)
        
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
        
        # Draw bounding box on result image
        color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
        cv.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        cv.putText(result_image, f'C{class_id}:{confidence:.2f}', 
                   (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Apply segmentation mask to original image
    segmentation_mask_3ch = cv.merge((segmentation_mask, segmentation_mask, segmentation_mask))
    segmented_image = cv.bitwise_and(image, segmentation_mask_3ch)
    stacked_image = np.vstack((image, segmented_image))
    # Save the stacked image
    output_filename = os.path.join(output_dir, f"{os.path.basename(image_path)}")
    cv.imwrite(output_filename, stacked_image)
    print(f"Saved segmentation result to {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Extract cone regions using YOLO detections and histogram backprojection")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--detections_dir", type=str, required=True, help="Directory containing YOLO detection .txt files") 
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
    
    # Process all images
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(args.images_dir) 
                   if f.lower().endswith(image_extensions)]
    
    for image_file in image_files:
        image_path = os.path.join(args.images_dir, image_file)
        
        # Look for corresponding detection file
        detection_file = os.path.join(args.detections_dir, 
                                    os.path.splitext(image_file)[0] + '.txt')
        
        if os.path.exists(detection_file):
            process_image_with_detections(image_path, detection_file, histograms, args.output_dir)
        else:
            print(f"No detection file found for {image_file} (looking for {detection_file})")

if __name__ == "__main__":
    main()