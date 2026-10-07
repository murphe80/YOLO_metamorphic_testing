"""
Formats ground truth data for evaluation for compatibility with COCO API.

This script converts the custom bounding box format to COCO format which is required by pycocotools.
"""

import json
import os
from pathlib import Path
from datetime import datetime


def convert_bbox_to_coco(points):
    """
    Convert bounding box from custom format to COCO format.
    
    Custom format: exterior: [[x1, y1], [x2, y2]] (top-left, bottom-right)
    COCO format: [x, y, width, height] where (x,y) is top-left corner
    
    Args:
        points: Dictionary with 'exterior' key containing bbox coordinates
        
    Returns:
        List [x, y, width, height] in COCO format
    """
    exterior = points['exterior']
    x1, y1 = exterior[0]
    x2, y2 = exterior[1]
    
    # Calculate top-left corner and dimensions
    x = min(x1, x2)
    y = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    
    return [x, y, width, height]


def convert_to_coco_format(input_dir, output_file, image_dir=None):
    """
    Convert custom ground truth format to COCO format.
    
    Args:
        input_dir: Directory containing JSON files with bounding boxes
        output_file: Output path for COCO format JSON file
        image_dir: Optional directory containing the actual images (for verification)
    """
    input_path = Path(input_dir)
    
    # Initialize COCO format structure
    coco_format = {
        "info": {
            "description": "FSOCO Sample Dataset - Converted to COCO format",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "FSOCO",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Track categories and assign IDs
    category_map = {}
    category_id_counter = 1
    
    # Track annotation IDs
    annotation_id_counter = 1
    
    # Process each JSON file
    json_files = sorted(input_path.glob("*.json"))
    
    for image_id, json_file in enumerate(json_files, start=1):
        # Skip if this is already a COCO format file
        if 'coco' in json_file.name.lower() or json_file.name == 'annotations.json':
            print(f"Skipping COCO format file: {json_file.name}")
            continue
            
        print(f"Processing: {json_file.name}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Skip if this looks like a COCO format file (has 'images' and 'annotations' keys)
        if 'images' in data and 'annotations' in data:
            print(f"Skipping COCO format file: {json_file.name}")
            continue
        
        # Extract image filename from JSON filename
        # e.g., "amz_00016.jpg.json" -> "amz_00016.jpg"
        image_filename = json_file.name.replace('.json', '')
        
        # Add image information
        image_info = {
            "id": image_id,
            "file_name": image_filename,
            "height": data['size']['height'],
            "width": data['size']['width']
        }
        coco_format['images'].append(image_info)
        
        # Process each object (annotation) in the image
        for obj in data['objects']:
            class_title = obj['classTitle']
            
            # Add category if not seen before
            if class_title not in category_map:
                category_map[class_title] = category_id_counter
                coco_format['categories'].append({
                    "id": category_id_counter,
                    "name": class_title,
                    "supercategory": "cone"  # You can adjust this
                })
                category_id_counter += 1
            
            # Convert bounding box to COCO format
            bbox = convert_bbox_to_coco(obj['points'])
            area = bbox[2] * bbox[3]  # width * height
            
            # Create annotation
            annotation = {
                "id": annotation_id_counter,
                "image_id": image_id,
                "category_id": category_map[class_title],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": []  # Empty for bounding boxes only
            }
            coco_format['annotations'].append(annotation)
            annotation_id_counter += 1
    
    # Save COCO format JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Output saved to: {output_file}")
    print(f"Total images: {len(coco_format['images'])}")
    print(f"Total annotations: {len(coco_format['annotations'])}")
    print(f"Categories: {list(category_map.keys())}")
    
    return coco_format


def main():
    """Main function to run the conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert custom ground truth format to COCO format"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/base_images/fsoco_sample/bounding_boxes',
        help='Directory containing JSON files with bounding boxes'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='data/base_images/fsoco_sample/annotations_coco.json',
        help='Output path for COCO format JSON file'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Optional: Directory containing the actual images'
    )
    
    args = parser.parse_args()
    
    # Convert paths to absolute if running from project root
    script_dir = Path(__file__).parent.parent.parent
    input_dir = script_dir / args.input_dir if not Path(args.input_dir).is_absolute() else args.input_dir
    output_file = script_dir / args.output_file if not Path(args.output_file).is_absolute() else args.output_file
    
    convert_to_coco_format(input_dir, output_file, args.image_dir)


if __name__ == "__main__":
    main()
