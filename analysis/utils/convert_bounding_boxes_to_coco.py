#!/usr/bin/env python3
"""
Convert bounding box JSON files to COCO format annotations.

Input: Directory with individual .jpg.json / .png.json files
Output: Single annotations_coco.json file in COCO format
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_bbox_json(json_path: Path) -> Dict:
    """Parse a single bounding box JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def convert_bbox_to_coco_format(exterior_points: List[List[float]]) -> List[float]:
    """
    Convert bounding box from exterior points to COCO format.

    Input: [[x1, y1], [x2, y2]] (top-left, bottom-right)
    Output: [x, y, width, height] where (x, y) is top-left corner
    """
    x1, y1 = exterior_points[0]
    x2, y2 = exterior_points[1]

    x = min(x1, x2)
    y = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    return [x, y, width, height]


def get_category_id(class_title: str, categories: Dict[str, int]) -> int:
    """Get or create category ID for a class title."""
    if class_title not in categories:
        categories[class_title] = len(categories) + 1
    return categories[class_title]


def convert_dataset_to_coco(bbox_dir: Path, images_dir: Path) -> Dict:
    """
    Convert all bounding box files to COCO format.

    Args:
        bbox_dir: Directory containing .json bounding box files
        images_dir: Directory containing corresponding image files

    Returns:
        COCO format dictionary
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    categories = {}  # class_title -> category_id
    annotation_id = 1
    image_id = 1

    # Process all JSON files in bbox directory
    bbox_files = sorted(bbox_dir.glob("*.json"))

    print(f"Found {len(bbox_files)} bounding box files")

    for bbox_file in bbox_files:
        # Determine image filename from bbox filename
        # Format: image_name.jpg.json or image_name.png.json
        bbox_name = bbox_file.name

        # Try different image extensions
        image_file = None
        for ext in ['.jpg', '.png', '.jpeg']:
            if bbox_name.endswith(f"{ext}.json"):
                image_name = bbox_name[:-5]  # Remove .json
                potential_image = images_dir / image_name
                if potential_image.exists():
                    image_file = potential_image
                    break

        if not image_file:
            print(f"Warning: Could not find image for {bbox_file.name}")
            continue

        # Parse bounding box JSON
        try:
            bbox_data = parse_bbox_json(bbox_file)
        except Exception as e:
            print(f"Error parsing {bbox_file}: {e}")
            continue

        # Add image info
        image_info = {
            "id": image_id,
            "file_name": image_file.name,
            "width": bbox_data["size"]["width"],
            "height": bbox_data["size"]["height"]
        }
        coco_data["images"].append(image_info)

        # Add annotations for all objects in this image
        for obj in bbox_data.get("objects", []):
            if obj["geometryType"] != "rectangle":
                print(f"Warning: Skipping non-rectangle object in {bbox_file.name}")
                continue

            # Convert bbox format
            exterior = obj["points"]["exterior"]
            bbox_coco = convert_bbox_to_coco_format(exterior)

            # Get category ID
            class_title = obj["classTitle"]
            category_id = get_category_id(class_title, categories)

            # Calculate area
            area = bbox_coco[2] * bbox_coco[3]

            # Add annotation
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox_coco,
                "area": area,
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation)
            annotation_id += 1

        image_id += 1

    # Add categories to COCO data
    for class_title, category_id in sorted(categories.items(), key=lambda x: x[1]):
        coco_data["categories"].append({
            "id": category_id,
            "name": class_title,
            "supercategory": "cone"
        })

    return coco_data


def main():
    parser = argparse.ArgumentParser(
        description="Convert bounding box JSON files to COCO format"
    )
    parser.add_argument(
        "--bbox-dir",
        type=str,
        required=True,
        help="Directory containing bounding box JSON files"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing corresponding image files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for COCO annotations JSON file"
    )

    args = parser.parse_args()

    bbox_dir = Path(args.bbox_dir)
    images_dir = Path(args.images_dir)
    output_path = Path(args.output)

    # Validate input directories
    if not bbox_dir.exists():
        print(f"Error: Bounding box directory not found: {bbox_dir}")
        return 1

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return 1

    print(f"Converting bounding boxes from: {bbox_dir}")
    print(f"Using images from: {images_dir}")

    # Convert to COCO format
    coco_data = convert_dataset_to_coco(bbox_dir, images_dir)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"\n✓ COCO annotations saved to: {output_path}")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")

    # Print categories
    print("\nCategories:")
    for cat in coco_data['categories']:
        count = sum(1 for ann in coco_data['annotations'] if ann['category_id'] == cat['id'])
        print(f"  {cat['id']}: {cat['name']} ({count} instances)")

    return 0


if __name__ == "__main__":
    exit(main())
