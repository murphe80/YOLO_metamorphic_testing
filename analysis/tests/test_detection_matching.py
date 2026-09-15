import unittest

from analysis.utils.detection_matching import compute_retention_metrics


class RetentionMetricTests(unittest.TestCase):
    def test_detection_loss_includes_vanished_true_and_false_positives(self):
        regions = {
            "orig_trans_gt": 4.0,
            "orig_trans": 2.0,
            "orig_gt": 3.0,
            "trans_gt": 0.0,
            "orig_only": 1.0,
            "trans_only": 0.0,
            "gt_only": 0.0,
        }

        metrics = compute_retention_metrics(regions)

        # Four of ten original detections vanished: three TPs and one FP.
        self.assertAlmostEqual(metrics["detection_loss_rate"], 0.4)

    def test_detection_loss_is_zero_when_there_are_no_original_detections(self):
        regions = {
            "orig_trans_gt": 0.0,
            "orig_trans": 0.0,
            "orig_gt": 0.0,
            "trans_gt": 2.0,
            "orig_only": 0.0,
            "trans_only": 1.0,
            "gt_only": 1.0,
        }

        metrics = compute_retention_metrics(regions)

        self.assertEqual(metrics["detection_loss_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
