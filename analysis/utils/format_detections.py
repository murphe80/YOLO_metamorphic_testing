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


def parse_detection_txt(txt_file, is_yolo_format=False, img_width=1920, img_height=1080):
    """
    Parse a detection .txt file.

    Detection format: class confidence x1 y1 x2 y2
    YOLO format: class x_center y_center width height (all normalized 0-1)

    Args:
        txt_file: Path to .txt file with detections
        is_yolo_format: If True, parse as YOLO format (normalized xywh)
        img_width: Image width for denormalizing YOLO format
        img_height: Image height for denormalizing YOLO format

    Returns:
        List of detection dictionaries
    """
    detections = []

    try:
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                if is_yolo_format and len(parts) >= 5:
                    # YOLO format: class x_center y_center width height (normalized)
                    class_id = int(parts[0])
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])

                    # Denormalize to absolute pixel coordinates
                    x_center = x_center_norm * img_width
                    y_center = y_center_norm * img_height
                    width = width_norm * img_width
                    height = height_norm * img_height

                    # Convert from center format to corner format
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2

                    # Convert bbox to COCO format
                    bbox = xyxy_to_xywh(x1, y1, x2, y2)

                    detections.append({
                        'class_id': class_id,
                        'confidence': 1.0,  # Ground truth has confidence 1.0
                        'bbox': bbox
                    })

                elif not is_yolo_format and len(parts) >= 6:
                    # Detection format: class confidence x1 y1 x2 y2
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


def format_detections(detection_dir, output_file, gt_coco_file=None, as_ground_truth=False, is_yolo_format=False, img_width=1920, img_height=1080):
    """
    Format detection data from .txt files to COCO JSON format.

    Args:
        detection_dir: Directory containing .txt detection files
        output_file: Output path for COCO format JSON
        gt_coco_file: Optional path to ground truth COCO JSON (to get image IDs and match filenames)
                     If None, image IDs will be auto-generated from filenames
        as_ground_truth: If True, output full COCO format with images/annotations/categories
                        If False, output just array of annotations (for detection results)
        is_yolo_format: If True, parse files as YOLO format (class x_center y_center width height, normalized)
                       rather than detection format (class conf x1 y1 x2 y2)
        img_width: Image width for denormalizing YOLO format coordinates
        img_height: Image height for denormalizing YOLO format coordinates
    """
    detection_path = Path(detection_dir)

    # Model class mapping to COCO category IDs
    # Model outputs: 0=yellow, 1=orange, 2=large_orange, 3=blue, 4=other, 5=unknown
    # COCO GT uses: 1=yellow, 2=blue, 3=orange, 4=large_orange
    MODEL_TO_COCO_CATEGORY = {
        0: 1,  # yellow_cone -> yellow_cone
        1: 3,  # orange_cone -> orange_cone
        2: 4,  # large_orange_cone -> large_orange_cone
        3: 2,  # blue_cone -> blue_cone
        4: 5,  # other_cone (not in GT categories)
        5: 6,  # unknown_cone (not in GT categories)
    }
    
    # Build image mapping
    image_map = {}
    
    if gt_coco_file:
        # Load ground truth to get image information
        print(f"Loading ground truth from: {gt_coco_file}")
        with open(gt_coco_file, 'r') as f:
            coco_gt = json.load(f)
        
        # Build mapping from image filename (without extension) to image ID
        # Include both original names and potential transformed variants
        for img in coco_gt['images']:
            filename = img['file_name']
            # Store both with and without extension for flexibility
            basename = Path(filename).stem
            image_map[basename] = img['id']
            image_map[filename] = img['id']
            
            # Also store without common file extensions for flexibility
            # This helps match "amz_00016" with "amz_00016.jpg" or "amz_00016.png"
            base_no_ext = basename.split('.')[0]
            if base_no_ext != basename:
                image_map[base_no_ext] = img['id']
    else:
        # Auto-generate image IDs from detection filenames
        print("No ground truth provided, auto-generating image IDs from detection files")
        txt_files = sorted(detection_path.glob("*.txt"))
        for idx, txt_file in enumerate(txt_files, start=1):
            basename = txt_file.stem
            # Strip transformation suffixes to get original image name
            # e.g., "amz_00840_imgColorWheel" -> "amz_00840"
            original_name = basename
            parts = basename.split('_')
            # Try to strip common transformation suffixes
            if len(parts) > 2 and parts[-1].startswith('img'):
                original_name = '_'.join(parts[:-1])
            
            image_map[basename] = idx
            image_map[original_name] = idx
    
    # Process all .txt files in the detection directory
    coco_detections = []
    image_id_to_stem = {}
    txt_files = sorted(detection_path.glob("*.txt"))
    
    if not txt_files:
        print(f"Warning: No .txt files found in {detection_dir}")
        return
    
    print(f"Processing {len(txt_files)} detection files...")
    
    for txt_file in txt_files:
        # Get corresponding image ID
        basename = txt_file.stem
        
        # Try to match with original basename first
        matched_basename = None
        if basename in image_map:
            matched_basename = basename
        else:
            # Try stripping common transformation suffixes
            # e.g., "amz_02840_imgResolution" -> "amz_02840"
            # Pattern: split on underscore and try progressively shorter names
            parts = basename.split('_')
            for i in range(len(parts), 0, -1):
                candidate = '_'.join(parts[:i])
                if candidate in image_map:
                    matched_basename = candidate
                    break
        
        if matched_basename is None:
            print(f"Warning: No matching image for {txt_file.name}")
            continue
        
        image_id = image_map[matched_basename]
        if image_id not in image_id_to_stem:
            image_id_to_stem[image_id] = basename

        # Parse detections from txt file
        detections = parse_detection_txt(txt_file, is_yolo_format=is_yolo_format, img_width=img_width, img_height=img_height)
        
        # Convert to COCO format
        for det in detections:
            # Map model class ID to COCO category ID
            model_class = det['class_id']
            if model_class not in MODEL_TO_COCO_CATEGORY:
                print(f"Warning: Unknown model class {model_class} in {txt_file.name}")
                continue

            coco_detection = {
                'id': len(coco_detections) + 1,  # Unique annotation ID
                'image_id': image_id,
                'category_id': MODEL_TO_COCO_CATEGORY[model_class],  # Map model class to COCO category
                'bbox': det['bbox'],
                'score': det['confidence'],
                'area': det['bbox'][2] * det['bbox'][3],  # width * height
                'iscrowd': 0  # Required by COCO evaluation
            }
            coco_detections.append(coco_detection)
    
    # Save to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if as_ground_truth:
        # Create full COCO format structure (for use as ground truth in evaluation)
        coco_output = {
            "images": [],
            "annotations": coco_detections,
            "categories": []
        }
        
        # Add image entries for all unique image_ids
        image_ids = set(det['image_id'] for det in coco_detections)
        for img_id in sorted(image_ids):
            stem = image_id_to_stem.get(img_id, f"image_{img_id}")
            coco_output['images'].append({
                "id": img_id,
                "file_name": f"{stem}.jpg",
                "height": 1080,  # Placeholder dimensions
                "width": 1920
            })
        
        # Add category entries (assuming categories 1-4 based on typical FSOCO)
        for cat_id in range(1, 5):
            coco_output['categories'].append({
                "id": cat_id,
                "name": f"category_{cat_id}",
                "supercategory": "cone"
            })

        with open(output_path, 'w') as f:
            json.dump(coco_output, f, indent=2)
    else:
        # Output just array of detections (for use with loadRes)
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
        default=None,
        help='Path to ground truth COCO JSON (for image ID mapping). If not provided, IDs will be auto-generated.'
    )
    parser.add_argument(
        '--as_ground_truth',
        action='store_true',
        help='Output full COCO format with images/annotations/categories (for use as GT in evaluation)'
    )
    parser.add_argument(
        '--yolo_format',
        action='store_true',
        help='Parse files as YOLO format (class x_center y_center width height, normalized)'
    )
    parser.add_argument(
        '--img_width',
        type=int,
        default=1920,
        help='Image width for denormalizing YOLO format (default: 1920)'
    )
    parser.add_argument(
        '--img_height',
        type=int,
        default=1080,
        help='Image height for denormalizing YOLO format (default: 1080)'
    )

    args = parser.parse_args()

    format_detections(args.detection_dir, args.output_file, args.gt_coco, args.as_ground_truth,
                     args.yolo_format, args.img_width, args.img_height)


if __name__ == "__main__":
    main()