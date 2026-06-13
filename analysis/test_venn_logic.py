#!/usr/bin/env python3
"""
Test script to verify Venn diagram matching logic with synthetic data.
"""

import json
import tempfile
from pathlib import Path

from utils.detection_matching import (
    compute_iou,
    aggregate_three_way_overlap,
    compute_retention_metrics
)


def test_iou():
    """Test IoU computation."""
    print("Testing IoU computation...")

    # Identical boxes
    bbox1 = [100, 100, 50, 50]
    assert compute_iou(bbox1, bbox1) == 1.0, "Identical boxes should have IoU = 1.0"

    # Non-overlapping boxes
    bbox2 = [200, 200, 50, 50]
    assert compute_iou(bbox1, bbox2) == 0.0, "Non-overlapping boxes should have IoU = 0.0"

    # Partially overlapping boxes (50% overlap in one dimension)
    bbox3 = [125, 100, 50, 50]  # Shifted 25 pixels right (half of 50)
    iou = compute_iou(bbox1, bbox3)
    expected = 0.333  # Approximately 1/3
    assert abs(iou - expected) < 0.01, f"50% overlap should give IoU ≈ {expected}, got {iou}"

    print("✓ IoU tests passed")


def test_three_way_matching():
    """Test three-way matching with synthetic data."""
    print("\nTesting three-way matching...")

    # Create synthetic detection data
    # Image 1: Perfect consistency (all detections match across all three)
    original_dets = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 100, 50, 50],
            "score": 0.9
        },
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [200, 200, 50, 50],
            "score": 0.85
        }
    ]

    # Transformed: Same as original (perfect metamorphic consistency)
    transformed_dets = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 100, 50, 50],
            "score": 0.88
        },
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [200, 200, 50, 50],
            "score": 0.87
        }
    ]

    # GT: Same as original (perfect accuracy)
    gt_dets = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 100, 50, 50]
        },
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [200, 200, 50, 50]
        }
    ]

    # Write to temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_path = Path(tmpdir) / "original.json"
        trans_path = Path(tmpdir) / "transformed.json"
        gt_path = Path(tmpdir) / "gt.json"

        with open(orig_path, 'w') as f:
            json.dump(original_dets, f)
        with open(trans_path, 'w') as f:
            json.dump(transformed_dets, f)
        with open(gt_path, 'w') as f:
            json.dump(gt_dets, f)

        # Run three-way matching
        stats = aggregate_three_way_overlap(orig_path, trans_path, gt_path, iou_threshold=0.5)

        print(f"  Region counts: {stats}")

        # Assertions for perfect consistency case
        assert stats['orig_trans_gt'] == 2, "Should have 2 three-way matches"
        assert stats['orig_trans'] == 0, "Should have no FP matches"
        assert stats['orig_gt'] == 0, "Should have no detection loss"
        assert stats['trans_gt'] == 0, "Should have no recovery"
        assert stats['orig_only'] == 0, "Should have no original-only FPs"
        assert stats['trans_only'] == 0, "Should have no transformed-only FPs"
        assert stats['gt_only'] == 0, "Should have no missed detections"

    print("✓ Three-way matching test passed")


def test_detection_loss():
    """Test case with detection loss."""
    print("\nTesting detection loss scenario...")

    original_dets = [
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [200, 200, 50, 50], "score": 0.85}
    ]

    # Transformed: Lost one detection
    transformed_dets = [
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "score": 0.88}
    ]

    # GT: Both are real
    gt_dets = [
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50]},
        {"image_id": 1, "category_id": 1, "bbox": [200, 200, 50, 50]}
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_path = Path(tmpdir) / "original.json"
        trans_path = Path(tmpdir) / "transformed.json"
        gt_path = Path(tmpdir) / "gt.json"

        with open(orig_path, 'w') as f:
            json.dump(original_dets, f)
        with open(trans_path, 'w') as f:
            json.dump(transformed_dets, f)
        with open(gt_path, 'w') as f:
            json.dump(gt_dets, f)

        stats = aggregate_three_way_overlap(orig_path, trans_path, gt_path, iou_threshold=0.5)
        metrics = compute_retention_metrics(stats)

        print(f"  Region counts: {stats}")
        print(f"  Metrics: {metrics}")

        # One detection maintained, one lost
        assert stats['orig_trans_gt'] == 1, "Should have 1 three-way match"
        assert stats['orig_gt'] == 1, "Should have 1 detection lost (orig ∩ GT only)"
        assert stats['trans_gt'] == 0, "No recovery"

        # TP retention should be 50% (1 out of 2 TPs maintained)
        # Note: detection_loss_rate counts orig_only (FPs that disappeared), not orig_gt (TPs that disappeared)
        # For TP retention, we look at orig_trans_gt / (orig_trans_gt + orig_gt)
        tp_retention = stats['orig_trans_gt'] / (stats['orig_trans_gt'] + stats['orig_gt'])
        assert abs(tp_retention - 0.5) < 0.01, f"Expected 50% TP retention, got {tp_retention}"

    print("✓ Detection loss test passed")


def test_spurious_detection():
    """Test case with spurious detections."""
    print("\nTesting spurious detection scenario...")

    original_dets = [
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "score": 0.9}
    ]

    # Transformed: Added a spurious detection
    transformed_dets = [
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "score": 0.88},
        {"image_id": 1, "category_id": 1, "bbox": [200, 200, 50, 50], "score": 0.65}  # Spurious
    ]

    # GT: Only one real object
    gt_dets = [
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50]}
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_path = Path(tmpdir) / "original.json"
        trans_path = Path(tmpdir) / "transformed.json"
        gt_path = Path(tmpdir) / "gt.json"

        with open(orig_path, 'w') as f:
            json.dump(original_dets, f)
        with open(trans_path, 'w') as f:
            json.dump(transformed_dets, f)
        with open(gt_path, 'w') as f:
            json.dump(gt_dets, f)

        stats = aggregate_three_way_overlap(orig_path, trans_path, gt_path, iou_threshold=0.5)
        metrics = compute_retention_metrics(stats)

        print(f"  Region counts: {stats}")
        print(f"  Metrics: {metrics}")

        assert stats['orig_trans_gt'] == 1, "Should have 1 three-way match"
        assert stats['trans_only'] == 1, "Should have 1 spurious detection"

        # Spurious rate should be 50% (1 out of 2 transformed detections is spurious)
        assert abs(metrics['spurious_detection_rate'] - 0.5) < 0.01, f"Expected 50% spurious, got {metrics['spurious_detection_rate']}"

    print("✓ Spurious detection test passed")


def main():
    """Run all tests."""
    print("=" * 80)
    print("VENN DIAGRAM LOGIC VERIFICATION")
    print("=" * 80)

    test_iou()
    test_three_way_matching()
    test_detection_loss()
    test_spurious_detection()

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    print("\nThe Venn diagram matching logic is working correctly!")
    print("You can now run the visualization scripts on your actual data.")


if __name__ == "__main__":
    main()
