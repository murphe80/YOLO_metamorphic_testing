# Analyze cone color detection results
# Update ground truth to be expected cone color (as a result of the transformation)

# For example, if the transformation is yellow_to_blue, the yellow cones in the ground truth should be updated to blue.

import copy
import json
from pathlib import Path

CLASS_NAMES = {
    0: "yellow_cone",
    1: "orange_cone",
    2: "unknown_cone",
    3: "blue_cone",
}

# Generic category names written by format_detections.py follow the FSOCO
# evaluation order, which is different from the model's class-id order.
GENERIC_CATEGORY_NAMES = {
    1: "yellow_cone",
    2: "blue_cone",
    3: "orange_cone",
    4: "big_orange_cone",
}

# Transformation name -> (source_class_id, target_class_id) or list of tuples for bidirectional
# e.g. yellow_to_blue: cones that were yellow (0) are now blue (3) in the transformed image
TRANSFORMATION_MAP = {
    "yellow_to_blue": [(0, 3)],
    "blue_to_yellow": [(3, 0)],
    "yellow_to_orange": [(0, 1)],
    "orange_to_blue": [(1, 3)],
    "blue_to_orange": [(3, 1)],
    "orange_to_yellow": [(1, 0)],
}


def _build_name_to_id(categories):
    """Map category ``name`` -> id, and map semantic cone names when GT uses category_1..category_4."""
    name_to_id = {}
    for cat in categories:
        cid = cat["id"]
        name = cat["name"]
        name_to_id[name] = cid
        if isinstance(name, str) and name.startswith("category_"):
            suffix = name[len("category_") :]
            if suffix.isdigit():
                semantic_name = GENERIC_CATEGORY_NAMES.get(int(suffix))
                if semantic_name:
                    name_to_id[semantic_name] = cid
    return name_to_id


def _wrap_detection_list_as_coco(annotations_list):
    """Bare JSON array from format_detections (no --as_ground_truth) -> minimal full COCO dict."""
    annotations = copy.deepcopy(annotations_list)
    image_ids = sorted({a["image_id"] for a in annotations})
    images = [
        {
            "id": iid,
            "file_name": f"image_{iid}.jpg",
            "height": 1080,
            "width": 1920,
        }
        for iid in image_ids
    ]
    categories = [
        {"id": k, "name": f"category_{k}", "supercategory": "cone"}
        for k in range(1, len(CLASS_NAMES) + 1)
    ]
    return {"images": images, "annotations": annotations, "categories": categories}


def _ensure_full_coco(data, ground_truth_file):
    if isinstance(data, list):
        return _wrap_detection_list_as_coco(data)
    if isinstance(data, dict) and "categories" in data and "annotations" in data:
        return data
    raise TypeError(
        f"{ground_truth_file}: expected full COCO object with 'annotations' and 'categories', "
        f"or a JSON array of annotations; got {type(data).__name__}."
    )


def update_ground_truth_cone_color(ground_truth_file, transformation, output_path):
    """
    Update the ground truth cone color based on the transformation. Format in COCO format.

    For each annotation whose category is the source cone color, change it to the target
    cone color (e.g. for yellow_to_blue, all yellow_cone annotations become blue_cone).

    Args:
        ground_truth_file: Path to the ground truth COCO JSON file to update.
        transformation: Transformation name (e.g. yellow_to_blue, blue_to_yellow).
        output_path: Path to save the updated ground truth annotations_coco.json file.
    """
    if transformation not in TRANSFORMATION_MAP:
        raise ValueError(
            f"Unknown transformation: {transformation}. "
            f"Must be one of: {list(TRANSFORMATION_MAP.keys())}"
        )
    transformations = TRANSFORMATION_MAP[transformation]

    with open(ground_truth_file, "r") as f:
        raw = json.load(f)
    coco = copy.deepcopy(_ensure_full_coco(raw, ground_truth_file))

    name_to_id = _build_name_to_id(coco["categories"])

    # Build mapping: source_category_id -> target_category_id
    # For each transformation rule (source_class, target_class), find the corresponding category IDs
    category_mapping = {}  # source_category_id -> target_category_id

    for source_class, target_class in transformations:
        source_name = CLASS_NAMES[source_class]
        target_name = CLASS_NAMES[target_class]

        source_category_id = name_to_id.get(source_name)
        if source_category_id is None:
            raise ValueError(
                f"Ground truth has no category '{source_name}'. "
                f"Categories present: {list(name_to_id.keys())}"
            )

        target_category_id = name_to_id.get(target_name)
        if target_category_id is None:
            # Create new category if needed
            sample = coco["categories"][0]["name"]
            if (
                isinstance(sample, str)
                and sample.startswith("category_")
                and sample[len("category_") :].isdigit()
            ):
                generic_ids = {
                    name: category_id
                    for category_id, name in GENERIC_CATEGORY_NAMES.items()
                }
                target_category_id = generic_ids[target_name]
                coco["categories"].append(
                    {
                        "id": target_category_id,
                        "name": f"category_{target_category_id}",
                        "supercategory": coco["categories"][0].get("supercategory", "cone"),
                    }
                )
            else:
                max_id = max(coco["categories"], key=lambda c: c["id"])["id"]
                target_category_id = max_id + 1
                coco["categories"].append(
                    {
                        "id": target_category_id,
                        "name": target_name,
                        "supercategory": coco["categories"][0].get("supercategory", "cone"),
                    }
                )
            name_to_id[target_name] = target_category_id
            name_to_id[coco["categories"][-1]["name"]] = target_category_id

        category_mapping[source_category_id] = target_category_id

    # Update all annotations in a single pass based on their original category_id
    updated = 0
    for ann in coco["annotations"]:
        original_category_id = ann["category_id"]
        if original_category_id in category_mapping:
            ann["category_id"] = category_mapping[original_category_id]
            updated += 1

    print(f"Updated {updated} annotations for transformation: {transformation}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Saved to {output_path}")
    return coco


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Update ground truth cone color to expected color after transformation (COCO format)."
    )
    parser.add_argument("--ground_truth_file", required=True, help="Path to ground truth COCO JSON.")
    parser.add_argument(
        "--transformation",
        required=True,
        choices=list(TRANSFORMATION_MAP.keys()),
        help="Transformation name (e.g. yellow_to_blue).",
    )
    parser.add_argument("--output_path", required=True, help="Path for updated COCO JSON.")
    args = parser.parse_args()

    update_ground_truth_cone_color(
        args.ground_truth_file,
        args.transformation,
        args.output_path,
    )


if __name__ == "__main__":
    main()
