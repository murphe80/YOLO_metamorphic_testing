#!/usr/bin/env python3
"""
Detection matching utilities for computing overlap between detection sets.

Implements three-way matching between original detections, transformed detections,
and ground truth annotations using IoU (Intersection over Union) thresholds.

Matching algorithm based on pycocotools COCOeval.evaluateImg approach:
- Sort detections by confidence score (highest first)
- Greedy assignment: each detection finds its best IoU match
- One-to-one constraint: each detection matched at most once
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate IoU between two bounding boxes in COCO format [x, y, width, height].

    Args:
        bbox1: First bounding box [x, y, width, height]
        bbox2: Second bounding box [x, y, width, height]

    Returns:
        IoU score between 0 and 1
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to corner coordinates
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    # Calculate intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def load_detections_by_image(coco_json_path: Path) -> Dict[int, List[Dict]]:
    """
    Load COCO JSON detections and group by image_id.

    Args:
        coco_json_path: Path to COCO format JSON file

    Returns:
        Dictionary mapping image_id to list of detections
    """
    with open(coco_json_path) as f:
        data = json.load(f)

    # Handle both full COCO format (with 'annotations' key) and direct list
    if isinstance(data, list):
        detections = data
    elif 'annotations' in data:
        detections = data['annotations']
    else:
        detections = []

    # Group by image_id
    by_image = defaultdict(list)
    for det in detections:
        by_image[det['image_id']].append(det)

    return dict(by_image)


def match_detection_pairs(
    original_dets: List[Dict],
    transformed_dets: List[Dict],
    iou_threshold: float = 0.5,
    require_category_match: bool = False
) -> List[Tuple[int, int]]:
    """
    Match original detections to transformed detections using COCO-style greedy
    assignment.

    Original detections are considered from highest to lowest confidence. Each
    is paired with the unmatched transformed detection having the greatest IoU
    above the threshold. Returned tuples contain indices into the input lists.

    Args:
        require_category_match: If True, a candidate pair is only eligible when
            both detections share the same category_id (in addition to passing
            the IoU threshold). Defaults to False to preserve purely spatial
            matching for existing callers (e.g. persistent-FP tracking).
    """
    original_sorted = sorted(
        enumerate(original_dets),
        key=lambda item: item[1].get('score', 1.0),
        reverse=True
    )
    matched_transformed = set()
    pairs = []

    for original_index, original_detection in original_sorted:
        best_iou = iou_threshold
        best_transformed_index = -1

        for transformed_index, transformed_detection in enumerate(transformed_dets):
            if transformed_index in matched_transformed:
                continue

            if require_category_match and (
                transformed_detection.get('category_id') != original_detection.get('category_id')
            ):
                continue

            iou = compute_iou(
                original_detection['bbox'],
                transformed_detection['bbox']
            )
            if iou > best_iou:
                best_iou = iou
                best_transformed_index = transformed_index

        if best_transformed_index != -1:
            pairs.append((original_index, best_transformed_index))
            matched_transformed.add(best_transformed_index)

    return pairs


def match_three_way_per_image(
    original_dets: List[Dict],
    transformed_dets: List[Dict],
    gt_dets: List[Dict],
    iou_threshold: float = 0.5,
    require_category_match: bool = True
) -> Dict[str, int]:
    """
    Perform three-way matching of detections for a single image using COCO evaluation approach.

    Algorithm (based on pycocotools COCOeval.evaluateImg, lines 273-296):
    1. Sort detections by confidence score (highest first)
    2. For each detection, find best IoU match via greedy assignment
    3. Match original→GT, transformed→GT, original→transformed independently
    4. Combine matches to identify 7 Venn regions

    This ensures high-confidence detections get priority in matching, preventing
    low-confidence spurious detections from "stealing" matches from true positives.

    Args:
        original_dets: Detections from original image
        transformed_dets: Detections from transformed image
        gt_dets: Ground truth annotations (assumed score=1.0)
        iou_threshold: Minimum IoU to consider a match (default: 0.5)
        require_category_match: If True (default), a match additionally
            requires both detections to share the same category_id. Without
            this, a detection whose predicted class flips after transformation
            (e.g. a color transform changing a cone's perceived color) is
            still counted as a retained true positive purely because its box
            didn't move, which contradicts category-aware evaluation metrics
            like COCO AP.

    Returns:
        Dictionary with counts for each Venn region:
        - orig_trans_gt: All three (TPs maintained)
        - orig_trans: Original + Transformed only (consistent FPs)
        - orig_gt: Original + GT only (TPs lost during transformation)
        - trans_gt: Transformed + GT only (TPs recovered)
        - orig_only: Original only (FPs that disappeared)
        - trans_only: Transformed only (spurious FPs introduced)
        - gt_only: GT only (missed detections)
    """
    # Sort by confidence (highest first) - COCO approach (coco_eval.py lines 173, 259)
    # Ground truth has no scores, treat as 1.0
    orig_sorted = sorted(
        enumerate(original_dets),
        key=lambda x: x[1].get('score', 1.0),
        reverse=True
    )
    trans_sorted = sorted(
        enumerate(transformed_dets),
        key=lambda x: x[1].get('score', 1.0),
        reverse=True
    )

    # Tracking dictionaries for two-way matches
    orig_to_gt = {}      # orig_idx -> gt_idx
    trans_to_gt = {}     # trans_idx -> gt_idx
    orig_to_trans = {}   # orig_idx -> trans_idx

    # Step 1: Match original → GT (COCO-style greedy, coco_eval.py lines 274-296)
    matched_gt = set()
    for orig_idx, orig_det in orig_sorted:
        best_iou = iou_threshold
        best_gt_idx = -1

        for gt_idx, gt_det in enumerate(gt_dets):
            if gt_idx in matched_gt:
                continue

            if require_category_match and gt_det.get('category_id') != orig_det.get('category_id'):
                continue

            iou = compute_iou(orig_det['bbox'], gt_det['bbox'])
            # Use '>' not '>=' for consistency with COCO (coco_eval.py line 286)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx != -1:
            orig_to_gt[orig_idx] = best_gt_idx
            matched_gt.add(best_gt_idx)

    # Step 2: Match transformed → GT (independent matching)
    matched_gt_trans = set()
    for trans_idx, trans_det in trans_sorted:
        best_iou = iou_threshold
        best_gt_idx = -1

        for gt_idx, gt_det in enumerate(gt_dets):
            if gt_idx in matched_gt_trans:
                continue

            if require_category_match and gt_det.get('category_id') != trans_det.get('category_id'):
                continue

            iou = compute_iou(trans_det['bbox'], gt_det['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx != -1:
            trans_to_gt[trans_idx] = best_gt_idx
            matched_gt_trans.add(best_gt_idx)

    # Step 3: Match original → transformed
    orig_to_trans.update(
        match_detection_pairs(
            original_dets, transformed_dets, iou_threshold,
            require_category_match=require_category_match
        )
    )

    # Step 4: Assign to Venn regions based on matches
    regions = {
        'orig_trans_gt': 0,
        'orig_trans': 0,
        'orig_gt': 0,
        'trans_gt': 0,
        'orig_only': 0,
        'trans_only': 0,
        'gt_only': 0
    }

    # Track which detections we've assigned to a region
    assigned_orig = set()
    assigned_trans = set()
    assigned_gt = set()

    # Three-way matches: orig matched to trans, and both match same GT
    for orig_idx in range(len(original_dets)):
        if orig_idx not in orig_to_trans:
            continue
        trans_idx = orig_to_trans[orig_idx]

        orig_gt = orig_to_gt.get(orig_idx)
        trans_gt = trans_to_gt.get(trans_idx)

        if orig_gt is not None and orig_gt == trans_gt:
            regions['orig_trans_gt'] += 1
            assigned_orig.add(orig_idx)
            assigned_trans.add(trans_idx)
            assigned_gt.add(orig_gt)

    # Two-way: orig + trans (no GT match)
    for orig_idx, trans_idx in orig_to_trans.items():
        if orig_idx not in assigned_orig and trans_idx not in assigned_trans:
            regions['orig_trans'] += 1
            assigned_orig.add(orig_idx)
            assigned_trans.add(trans_idx)

    # Two-way: orig + GT (trans lost this detection)
    for orig_idx, gt_idx in orig_to_gt.items():
        if orig_idx not in assigned_orig and gt_idx not in assigned_gt:
            regions['orig_gt'] += 1
            assigned_orig.add(orig_idx)
            assigned_gt.add(gt_idx)

    # Two-way: trans + GT (orig missed, trans recovered)
    for trans_idx, gt_idx in trans_to_gt.items():
        if trans_idx not in assigned_trans and gt_idx not in assigned_gt:
            regions['trans_gt'] += 1
            assigned_trans.add(trans_idx)
            assigned_gt.add(gt_idx)

    # Single-source only
    regions['orig_only'] = len(original_dets) - len(assigned_orig)
    regions['trans_only'] = len(transformed_dets) - len(assigned_trans)
    regions['gt_only'] = len(gt_dets) - len(assigned_gt)

    return regions


def aggregate_three_way_overlap(
    original_coco_path: Path,
    transformed_coco_path: Path,
    gt_coco_path: Path,
    iou_threshold: float = 0.5,
    require_category_match: bool = True
) -> Dict[str, float]:
    """
    Compute image-normalized three-way overlap statistics across all images.

    Each image contributes exactly 1.0 total weight across the seven Venn
    regions (split proportionally across whichever regions its detections
    fall into), so images with many cones do not dominate images with few
    cones.

    Args:
        original_coco_path: Path to original detections COCO JSON
        transformed_coco_path: Path to transformed detections COCO JSON
        gt_coco_path: Path to ground truth annotations COCO JSON
        iou_threshold: Minimum IoU to consider a match (default: 0.5)
        require_category_match: Passed through to match_three_way_per_image;
            when True (default), matches also require equal category_id so
            that a detection whose predicted class changes after
            transformation is not miscounted as a retained true positive.

    Returns:
        Dictionary with image-normalized aggregate weights for all 7 Venn regions
    """
    # Load detections grouped by image
    original_by_img = load_detections_by_image(original_coco_path)
    transformed_by_img = load_detections_by_image(transformed_coco_path)
    gt_by_img = load_detections_by_image(gt_coco_path)

    # Aggregate per-image proportions across all images
    total_regions = {
        'orig_trans_gt': 0.0,
        'orig_trans': 0.0,
        'orig_gt': 0.0,
        'trans_gt': 0.0,
        'orig_only': 0.0,
        'trans_only': 0.0,
        'gt_only': 0.0
    }

    # Get all image IDs that appear in any source
    all_image_ids = set(original_by_img.keys()) | set(transformed_by_img.keys()) | set(gt_by_img.keys())

    for image_id in all_image_ids:
        orig_dets = original_by_img.get(image_id, [])
        trans_dets = transformed_by_img.get(image_id, [])
        gt_dets = gt_by_img.get(image_id, [])

        # Match detections for this image
        image_regions = match_three_way_per_image(
            orig_dets, trans_dets, gt_dets, iou_threshold, require_category_match
        )

        image_total = sum(image_regions.values())
        if image_total == 0:
            continue

        # Accumulate normalized per-image region weights. Using the raw
        # fraction (rather than round()) preserves partial contributions from
        # images with mixed outcomes instead of collapsing each image to a
        # single "winning" region and silently dropping ties/near-ties.
        for region, count in image_regions.items():
            total_regions[region] += count / image_total

    return total_regions


def compute_retention_metrics(regions: Dict[str, float]) -> Dict[str, float]:
    """
    Compute retention and consistency metrics from Venn region weights.

    Args:
        regions: Dictionary with weights for all 7 Venn regions

    Returns:
        Dictionary with computed metrics:
        - tp_retention_rate: Proportion of true positives maintained
        - fp_consistency_rate: Proportion of false positives that are consistent
        - detection_loss_rate: Proportion of original detections lost
        - new_hallucination_rate: Proportion of transformed detections that are new false positives
        - new_tp_rate: Proportion of original-missed GT cones recovered by transformed detections
    """
    metrics = {}

    # True positive retention: maintained original TPs / all original TPs.
    # In Venn region terms: (orig ∩ gt ∩ trans) / (orig ∩ gt).
    orig_tp_count = regions['orig_trans_gt'] + regions['orig_gt']

    if orig_tp_count > 0:
        metrics['tp_retention_rate'] = regions['orig_trans_gt'] / orig_tp_count
    else:
        metrics['tp_retention_rate'] = 0.0

    # False positive consistency: (orig ∩ trans only) / (all false positives)
    all_orig_trans = regions['orig_only'] + regions['trans_only'] +regions['orig_trans']
    if all_orig_trans > 0:
        metrics['fp_consistency_rate'] = regions['orig_trans'] / all_orig_trans
    else:
        metrics['fp_consistency_rate'] = 0.0

    # Detection loss: share of all original detections (TP or FP) that have no
    # matching transformed detection. This includes lost TPs (orig_gt) and
    # disappearing FPs (orig_only).
    total_orig = (regions['orig_trans_gt'] + regions['orig_trans'] +
                  regions['orig_gt'] + regions['orig_only'])
    if total_orig > 0:
        metrics['detection_loss_rate'] = (
            regions['orig_gt'] + regions['orig_only']
        ) / total_orig
    else:
        metrics['detection_loss_rate'] = 0.0

    # New hallucination rate: transformed detections that match neither original nor GT
    total_trans = (regions['orig_trans_gt'] + regions['orig_trans'] +
                   regions['trans_gt'] + regions['trans_only'])
    if total_trans > 0:
        metrics['new_hallucination_rate'] = regions['trans_only'] / total_trans
    else:
        metrics['new_hallucination_rate'] = 0.0

    # New true positive rate: GT cones missed by original but recovered by transformed
    original_missed_gt_count = regions['trans_gt'] + regions['gt_only']
    if original_missed_gt_count > 0:
        metrics['new_tp_rate'] = regions['trans_gt'] / original_missed_gt_count
    else:
        metrics['new_tp_rate'] = 0.0

    return metrics
