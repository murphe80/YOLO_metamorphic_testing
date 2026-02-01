"""
Formats detection data for evaluation for compatibility with COCO API.

Detection format: class: int, confidence: float, bbox: [x1, y1, x2, y2]
e.g. 3 0.93 249.04 349.11 357.34 348.39 

COCO format: bbox: [x, y, width, height], category_id: int, score: float
"""

import json
import os
import argparse
from pathlib import Path


def xyxy_to_xywh(x1, y1, x2, y2):
    """
    Convert bbox from [x1, y1, x2, y2] to COCO format [x, y, width, height].
    
    Args:
        x1, y1: Top-left corner coordinates
        x2, y2: Bottom-right corner coordinates
        
    Returns:
        List [x, y, width, height] in COCO format
    """
    x = min(x1, x2)
    y = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return [x, y, width, height]


def parse_detection_txt(txt_file):
    """
    Parse a detection .txt file.
    
    Format per line: class confidence x1 y1 x2 y2
    
    Args:
        txt_file: Path to .txt file with detections
        
    Returns:
        List of detection dictionaries
    """
    detections = []
    
    try:
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    class_id = int(parts[0])
                    confidence = float(parts[1])
                    x1 = float(parts[2])
                    y1 = float(parts[3])
                    x2 = float(parts[4])
                    y2 = float(parts[5])
                    
                    # Convert bbox to COCO format
                    bbox = xyxy_to_xywh(x1, y1, x2, y2)
                    
                    detections.append({
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': bbox
                    })
    except FileNotFoundError:
        print(f"Warning: File not found: {txt_file}")
    except Exception as e:
        print(f"Error parsing {txt_file}: {e}")
    
    return detections


def format_detections(detection_dir, output_file, gt_coco_file):
    """
    Format detection data from .txt files to COCO JSON format.
    
    Args:
        detection_dir: Directory containing .txt detection files
        output_file: Output path for COCO format JSON
        gt_coco_file: Path to ground truth COCO JSON (to get image IDs and match filenames)
    """
    detection_path = Path(detection_dir)
    
    # Load ground truth to get image information
    print(f"Loading ground truth from: {gt_coco_file}")
    with open(gt_coco_file, 'r') as f:
        coco_gt = json.load(f)
    
    # Build mapping from image filename (without extension) to image ID
    image_map = {}
    for img in coco_gt['images']:
        filename = img['file_name']
        # Store both with and without extension for flexibility
        basename = Path(filename).stem
        image_map[basename] = img['id']
        image_map[filename] = img['id']
    
    # Process all .txt files in the detection directory
    coco_detections = []
    txt_files = sorted(detection_path.glob("*.txt"))
    
    if not txt_files:
        print(f"Warning: No .txt files found in {detection_dir}")
        return
    
    print(f"Processing {len(txt_files)} detection files...")
    
    for txt_file in txt_files:
        # Get corresponding image ID
        basename = txt_file.stem
        
        if basename not in image_map:
            print(f"Warning: No matching image for {txt_file.name}")
            continue
        
        image_id = image_map[basename]
        
        # Parse detections from txt file
        detections = parse_detection_txt(txt_file)
        
        # Convert to COCO format
        for det in detections:
            coco_detection = {
                'image_id': image_id,
                'category_id': det['class_id'] + 1,  # COCO categories are 1-indexed
                'bbox': det['bbox'],
                'score': det['confidence']
            }
            coco_detections.append(coco_detection)
    
    # Save to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(coco_detections, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Total detections: {len(coco_detections)}")
    print(f"Output saved to: {output_file}")
    
    return coco_detections


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Convert detection .txt files to COCO format JSON"
    )
    parser.add_argument(
        '--detection_dir',
        type=str,
        required=True,
        help='Directory containing .txt detection files'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output path for COCO format JSON file'
    )
    parser.add_argument(
        '--gt_coco',
        type=str,
        default='data/base_images/fsoco_sample/annotations_coco.json',
        help='Path to ground truth COCO JSON (for image ID mapping)'
    )
    
    args = parser.parse_args()
    
    format_detections(args.detection_dir, args.output_file, args.gt_coco)


if __name__ == "__main__":
    main()