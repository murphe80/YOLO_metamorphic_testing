#!/usr/bin/env python3
"""
Convert Supervisely JSON annotations to YOLO txt label format.

Input format (JSON):
- Each file contains image dimensions and objects with absolute bounding box coordinates
- Objects have classTitle: "yellow_cone", "blue_cone", "orange_cone", "large_orange_cone"
- Bounding boxes defined by exterior points: [[x1, y1], [x2, y2]]

Output format (YOLO txt):
- Each line: <class_id> <x_center> <y_center> <width> <height>
- All coordinates normalized to [0, 1] relative to image dimensions
- Class mapping: 0=yellow_cone, 1=orange_cone, 3=blue_cone

Usage:
    python utils/convert_json_to_yolo_labels.py
    python utils/convert_json_to_yolo_labels.py --input-dir path/to/json --output-dir path/to/labels
"""

import argparse
import json
from pathlib import Path

# Class name to ID mapping (from analysis/utils/cone_color_analysis.py)
CLASS_MAP = {
    "yellow_cone": 0,
    "orange_cone": 1,
    "large_orange_cone": 2,  # Map large_orange_cone to orange_cone
    "blue_cone": 3,
}


def convert_bbox_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """
    Convert absolute bounding box coordinates to YOLO format.

    Args:
        x1, y1: Top-left corner (absolute pixels)
        x2, y2: Bottom-right corner (absolute pixels)
        img_width, img_height: Image dimensions

    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0, 1]
    """
    # Calculate center and dimensions
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    # Normalize to [0, 1]
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return x_center_norm, y_center_norm, width_norm, height_norm


def convert_json_to_yolo(json_path, output_path):
    """
    Convert a single JSON annotation file to YOLO txt format.

    Args:
        json_path: Path to input JSON file
        output_path: Path to output txt file

    Returns:
        Number of objects converted
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract image dimensions
    img_width = data['size']['width']
    img_height = data['size']['height']

    # Convert each object
    yolo_lines = []
    skipped_objects = []

    for obj in data['objects']:
        class_title = obj['classTitle']

        # Check if class is recognized
        if class_title not in CLASS_MAP:
            skipped_objects.append(class_title)
            continue

        class_id = CLASS_MAP[class_title]

        # Extract bounding box coordinates
        exterior = obj['points']['exterior']
        x1, y1 = exterior[0]
        x2, y2 = exterior[1]

        # Convert to YOLO format
        x_center, y_center, width, height = convert_bbox_to_yolo(
            x1, y1, x2, y2, img_width, img_height
        )

        # Format: class x_center y_center width height
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)

    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(yolo_lines))
        if yolo_lines:  # Add trailing newline if file is not empty
            f.write('\n')

    # Report skipped objects
    if skipped_objects:
        print(f"  Warning: Skipped unknown classes in {json_path.name}: {set(skipped_objects)}")

    return len(yolo_lines)


def main():
    parser = argparse.ArgumentParser(
        description='Convert Supervisely JSON annotations to YOLO txt format'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/base_images/fsoco_sample/bounding_boxes'),
        help='Directory containing JSON annotation files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/base_images/fsoco_sample/labels'),
        help='Directory to write YOLO txt files'
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    # Find all JSON files
    json_files = sorted(args.input_dir.glob('*.json'))

    if not json_files:
        print(f"Error: No JSON files found in {args.input_dir}")
        return 1

    print(f"Converting {len(json_files)} JSON files...")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print()

    # Convert each file
    total_objects = 0
    for json_path in json_files:
        # Output filename: remove .jpg.json extension, add .txt
        # e.g., amz_00016.jpg.json -> amz_00016.txt
        stem = json_path.stem  # "amz_00016.jpg"
        if stem.endswith('.jpg'):
            stem = stem[:-4]  # "amz_00016"
        output_filename = f"{stem}.txt"
        output_path = args.output_dir / output_filename

        num_objects = convert_json_to_yolo(json_path, output_path)
        total_objects += num_objects
        print(f"✓ {json_path.name} -> {output_filename} ({num_objects} objects)")

    print()
    print(f"✓ Conversion complete!")
    print(f"  Files converted: {len(json_files)}")
    print(f"  Total objects: {total_objects}")
    print(f"  Output directory: {args.output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
