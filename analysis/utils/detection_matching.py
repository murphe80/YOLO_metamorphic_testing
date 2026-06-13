#!/usr/bin/env python3
"""
Detection matching utilities for computing overlap between detection sets.

Implements three-way matching between original detections, transformed detections,
and ground truth annotations using IoU (Intersection over Union) thresholds.
"""

import json
import numpy as np
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


def match_three_way_per_image(
    original_dets: List[Dict],
    transformed_dets: List[Dict],
    gt_dets: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, int]:
    """
    Perform three-way matching of detections for a single image.

    Uses a greedy approach:
    1. Find all three-way matches (detection present in all three sources)
    2. Find two-way matches among remaining detections
    3. Assign remaining detections to single-source regions

    Args:
        original_dets: Detections from original image
        transformed_dets: Detections from transformed image
        gt_dets: Ground truth annotations
        iou_threshold: Minimum IoU to consider a match

    Returns:
        Dictionary with counts for each Venn region:
        - orig_trans_gt: All three
        - orig_trans: Original + Transformed only
        - orig_gt: Original + GT only
        - trans_gt: Transformed + GT only
        - orig_only: Original only
        - trans_only: Transformed only
        - gt_only: GT only
    """
    # Track which detections have been matched
    matched_orig = set()
    matched_trans = set()
    matched_gt = set()

    # Initialize region counts
    regions = {
        'orig_trans_gt': 0,
        'orig_trans': 0,
        'orig_gt': 0,
        'trans_gt': 0,
        'orig_only': 0,
        'trans_only': 0,
        'gt_only': 0
    }

    # Step 1: Find three-way matches (Original ∩ Transformed ∩ GT)
    for i, orig in enumerate(original_dets):
        if i in matched_orig:
            continue

        best_trans_idx = None
        best_trans_iou = 0.0

        # Find best matching transformed detection
        for j, trans in enumerate(transformed_dets):
            if j in matched_trans:
                continue
            iou = compute_iou(orig['bbox'], trans['bbox'])
            if iou >= iou_threshold and iou > best_trans_iou:
                best_trans_iou = iou
                best_trans_idx = j

        if best_trans_idx is None:
            continue

        # Now check if this pair also matches a GT detection
        best_gt_idx = None
        best_gt_iou = 0.0

        for k, gt in enumerate(gt_dets):
            if k in matched_gt:
                continue
            iou_orig_gt = compute_iou(orig['bbox'], gt['bbox'])
            iou_trans_gt = compute_iou(transformed_dets[best_trans_idx]['bbox'], gt['bbox'])

            # Both original and transformed should match GT
            if (iou_orig_gt >= iou_threshold and
                iou_trans_gt >= iou_threshold and
                iou_orig_gt > best_gt_iou):
                best_gt_iou = iou_orig_gt
                best_gt_idx = k

        if best_gt_idx is not None:
            # Three-way match found!
            regions['orig_trans_gt'] += 1
            matched_orig.add(i)
            matched_trans.add(best_trans_idx)
            matched_gt.add(best_gt_idx)

    # Step 2a: Find two-way matches (Original ∩ Transformed)
    for i, orig in enumerate(original_dets):
        if i in matched_orig:
            continue

        for j, trans in enumerate(transformed_dets):
            if j in matched_trans:
                continue

            iou = compute_iou(orig['bbox'], trans['bbox'])
            if iou >= iou_threshold:
                regions['orig_trans'] += 1
                matched_orig.add(i)
                matched_trans.add(j)
                break

    # Step 2b: Find two-way matches (Original ∩ GT)
    for i, orig in enumerate(original_dets):
        if i in matched_orig:
            continue

        for k, gt in enumerate(gt_dets):
            if k in matched_gt:
                continue

            iou = compute_iou(orig['bbox'], gt['bbox'])
            if iou >= iou_threshold:
                regions['orig_gt'] += 1
                matched_orig.add(i)
                matched_gt.add(k)
                break

    # Step 2c: Find two-way matches (Transformed ∩ GT)
    for j, trans in enumerate(transformed_dets):
        if j in matched_trans:
            continue

        for k, gt in enumerate(gt_dets):
            if k in matched_gt:
                continue

            iou = compute_iou(trans['bbox'], gt['bbox'])
            if iou >= iou_threshold:
                regions['trans_gt'] += 1
                matched_trans.add(j)
                matched_gt.add(k)
                break

    # Step 3: Count unmatched detections (single-source regions)
    regions['orig_only'] = len(original_dets) - len(matched_orig)
    regions['trans_only'] = len(transformed_dets) - len(matched_trans)
    regions['gt_only'] = len(gt_dets) - len(matched_gt)

    return regions


def aggregate_three_way_overlap(
    original_coco_path: Path,
    transformed_coco_path: Path,
    gt_coco_path: Path,
    iou_threshold: float = 0.5
) -> Dict[str, int]:
    """
    Compute aggregate three-way overlap statistics across all images.

    Args:
        original_coco_path: Path to original detections COCO JSON
        transformed_coco_path: Path to transformed detections COCO JSON
        gt_coco_path: Path to ground truth annotations COCO JSON
        iou_threshold: Minimum IoU to consider a match (default: 0.5)

    Returns:
        Dictionary with aggregate counts for all 7 Venn regions
    """
    # Load detections grouped by image
    original_by_img = load_detections_by_image(original_coco_path)
    transformed_by_img = load_detections_by_image(transformed_coco_path)
    gt_by_img = load_detections_by_image(gt_coco_path)

    # Aggregate results across all images
    total_regions = {
        'orig_trans_gt': 0,
        'orig_trans': 0,
        'orig_gt': 0,
        'trans_gt': 0,
        'orig_only': 0,
        'trans_only': 0,
        'gt_only': 0
    }

    # Get all image IDs that appear in any source
    all_image_ids = set(original_by_img.keys()) | set(transformed_by_img.keys()) | set(gt_by_img.keys())

    for image_id in all_image_ids:
        orig_dets = original_by_img.get(image_id, [])
        trans_dets = transformed_by_img.get(image_id, [])
        gt_dets = gt_by_img.get(image_id, [])

        # Match detections for this image
        image_regions = match_three_way_per_image(
            orig_dets, trans_dets, gt_dets, iou_threshold
        )

        # Accumulate counts
        for region, count in image_regions.items():
            total_regions[region] += count

    return total_regions


def compute_retention_metrics(regions: Dict[str, int]) -> Dict[str, float]:
    """
    Compute retention and consistency metrics from Venn region counts.

    Args:
        regions: Dictionary with counts for all 7 Venn regions

    Returns:
        Dictionary with computed metrics:
        - tp_retention_rate: Proportion of true positives maintained
        - fp_consistency_rate: Proportion of false positives that are consistent
        - detection_loss_rate: Proportion of original detections lost
        - spurious_detection_rate: Proportion of new spurious detections
    """
    metrics = {}

    # True positive retention: (orig ∩ gt ∩ trans + orig ∩ gt) / (all orig that match GT)
    orig_tp_count = regions['orig_trans_gt'] + regions['orig_gt']
    all_gt_matched = regions['orig_trans_gt'] + regions['orig_gt'] + regions['trans_gt']

    if all_gt_matched > 0:
        metrics['tp_retention_rate'] = orig_tp_count / all_gt_matched
    else:
        metrics['tp_retention_rate'] = 0.0

    # False positive consistency: (orig ∩ trans only) / (all matched orig-trans pairs)
    all_orig_trans = regions['orig_trans_gt'] + regions['orig_trans']
    if all_orig_trans > 0:
        metrics['fp_consistency_rate'] = regions['orig_trans'] / all_orig_trans
    else:
        metrics['fp_consistency_rate'] = 0.0

    # Detection loss: detections in original that disappeared
    total_orig = (regions['orig_trans_gt'] + regions['orig_trans'] +
                  regions['orig_gt'] + regions['orig_only'])
    if total_orig > 0:
        metrics['detection_loss_rate'] = regions['orig_only'] / total_orig
    else:
        metrics['detection_loss_rate'] = 0.0

    # Spurious detection rate: new detections in transformed
    total_trans = (regions['orig_trans_gt'] + regions['orig_trans'] +
                   regions['trans_gt'] + regions['trans_only'])
    if total_trans > 0:
        metrics['spurious_detection_rate'] = regions['trans_only'] / total_trans
    else:
        metrics['spurious_detection_rate'] = 0.0

    return metrics
