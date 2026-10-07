"""Reviewed hue mappings and source-class-specific evaluation cohorts.

COCO category IDs here are the canonical IDs from format_detections.py.
Large orange cones are excluded at every hue setting because their geometry
differs from the regular cone classes.
"""

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re

from analysis.utils.detection_matching import compute_iou

NAMES = {1: "yellow", 2: "blue", 3: "orange", 4: "large orange"}
# source blue, orange, yellow; None means excluded by manual assessment.
TARGETS = [
    (2, 3, 1), (2, None, None), (None, None, None),
    (None, None, None), (3, None, 2), (None, None, None),
    (1, None, None), (None, None, None), (None, 3, 3), (2, 3, 1),
]
OBSERVED = [
    ("blue", "orange", "yellow"), ("blue", "yellow", "green"),
    ("highly saturated blue", "green", "green"), ("pink", "cyan", "cyan"),
    ("dark orange", "green/teal", "blue"), ("orange", "darker blue", "purple"),
    ("yellow", "pink", "purple"), ("green", "pink", "pink"),
    ("green", "orange", "orange"), ("blue", "orange", "yellow"),
]
POLICY = (
    "One result per source class; no pooled score. Expected reference labels are "
    "remapped before category-aware evaluation. Target-class predictions whose "
    "best spatial reference (IoU >= 0.5) belongs to another source class are "
    "ignored for this cohort. Unmatched target-class predictions remain false "
    "positive candidates in each applicable cohort; their source colour cannot "
    "be inferred. Large orange references are excluded because their geometry differs; other/unknown references are also excluded."
)


def is_color_wheel(variant):
    return bool(re.fullmatch(r"imgColorWheel_?[0-9]", variant))


def mapping_for_key(key):
    """Resolve a source-specific bootstrap key to its reviewed mapping."""
    variant, separator, shift = key.partition("__")
    if not separator or not (is_color_wheel(variant) or variant == "original"):
        return None
    record = metadata("imgColorWheel_0" if variant == "original" else variant)
    return next(m for m in record["included_mappings"] if m["key"] == shift)


def baseline_key(variant_key):
    mapping = mapping_for_key(variant_key)
    if mapping is None:
        return "original"
    source = mapping["source"].replace(" ", "_")
    return f"original__{source}_to_{source}"


def metadata(variant):
    if not is_color_wheel(variant):
        raise ValueError(f"No reviewed hue mapping for {variant}")
    index = int(variant[-1])
    included, excluded = [], []
    for source, target, observed in zip((2, 3, 1), TARGETS[index], OBSERVED[index]):
        record = {
            "source_category_id": source,
            "source": NAMES[source],
            "target_category_id": target,
            "target": NAMES[target] if target else observed,
        }
        record["mapping"] = f"{record['source']} → {record['target']}"
        record["key"] = f"{record['source'].replace(' ', '_')}_to_{record['target'].replace(' ', '_')}"
        if index == 1 and source == 3:
            record["exclusion_reason"] = "Orange-to-yellow mapping excluded from analysis by user request."
        if index == 5 and source == 2:
            record["exclusion_reason"] = "Blue-to-orange analysis uses imgColorWheel_4 instead, by user request."
        if index == 4 and source == 3:
            record["exclusion_reason"] = (
                "Unreliable orange-to-blue mapping: transformed orange cones often "
                "appear green/teal rather than blue."
            )
        (included if target else excluded).append(record)
    return {
        "variant_id": variant, "hue_rotation": index * 20,
        "status": "included" if included else "excluded",
        "included_mappings": included, "excluded_mappings": excluded,
        "excluded_classes": ["large_orange_cone", "other_cone", "unknown_cone"],
        "evaluation_policy": POLICY,
    }


def annotations(data):
    return data if isinstance(data, list) else data["annotations"]


def grouped(data):
    result = defaultdict(list)
    for ann in annotations(data):
        result[ann["image_id"]].append(ann)
    return result


def remap_reference(reference, mapping, remap=True):
    result = deepcopy(reference)
    source, target = mapping["source_category_id"], mapping["target_category_id"]
    result["annotations"] = [a for a in result["annotations"] if a["category_id"] == source]
    for a in result["annotations"]:
        a["category_id"] = target if remap else source
    category = target if remap else source
    result["categories"] = [{"id": category, "name": NAMES[category]}]
    return result


def select_predictions(predictions, reference, mapping, transformed=True, fallback=None,
                       include_wrong_classes=False):
    """Ignore objects belonging to other source cohorts without hiding background FPs.

    Ground truth is the primary spatial reference for overlap/confidence; original
    predictions provide fallback locations for objects absent from GT.
    Overlap metrics include all predicted classes so classification errors can
    enter trans_only. AP and confidence callers retain target-class selection.
    """
    refs = grouped(reference)
    fallback_refs = grouped(fallback) if fallback is not None else {}
    source = mapping["source_category_id"]
    category = mapping["target_category_id"] if transformed else source
    selected = []
    for ann in annotations(predictions):
        if not include_wrong_classes and ann["category_id"] != category:
            continue
        owner = None
        for candidates in (refs.get(ann["image_id"], []), fallback_refs.get(ann["image_id"], [])):
            best = max(candidates, key=lambda r: compute_iou(ann["bbox"], r["bbox"]), default=None)
            if best is not None and compute_iou(ann["bbox"], best["bbox"]) >= 0.5:
                owner = best["category_id"]
                break
        if owner is None or owner == source:
            selected.append(dict(ann))
    return selected


def evaluate_files(gt_path, dt_path, variant, image_dir=None):
    """Evaluate each included shift separately, with explicit excluded settings."""
    from tempfile import TemporaryDirectory
    from analysis.coco_eval import evaluate_coco

    reference = json.loads(Path(gt_path).read_text())
    predictions = json.loads(Path(dt_path).read_text())
    result = metadata(variant)
    result["class_shifts"] = {}
    keys = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
    with TemporaryDirectory(prefix="color_wheel_eval_") as directory:
        gt, dt = Path(directory) / "gt.json", Path(directory) / "dt.json"
        for mapping in result["included_mappings"]:
            expected = remap_reference(reference, mapping)
            selected = select_predictions(predictions, reference, mapping)
            gt.write_text(json.dumps(expected))
            dt.write_text(json.dumps(selected))
            values = evaluate_coco(gt, dt, image_dir=image_dir)
            result["class_shifts"][mapping["key"]] = {
                **mapping, "reference_count": len(expected["annotations"]),
                "prediction_count": len(selected),
                **dict(zip(keys, (float(v) if v >= 0 else None for v in values))),
            }
    return result
