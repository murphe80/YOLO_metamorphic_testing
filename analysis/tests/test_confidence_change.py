import json
import tempfile
import unittest
from pathlib import Path

from analysis.visualisation import plot_confidence_change


def _detection(image_id, bbox, score):
    return {
        "image_id": image_id,
        "category_id": 1,
        "bbox": bbox,
        "score": score,
    }


class ConfidenceChangeTests(unittest.TestCase):
    def _write_inputs(self, root):
        model_dir = root / "yolox"
        original_dir = model_dir / "original"
        variant_dir = model_dir / "imgExample_0"
        original_dir.mkdir(parents=True)
        variant_dir.mkdir()

        ground_truth = {
            "annotations": [
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
                {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10]},
            ]
        }
        original = [
            _detection(1, [0, 0, 10, 10], 0.90),
            _detection(1, [30, 30, 10, 10], 0.50),
            _detection(2, [0, 0, 10, 10], 0.80),
        ]
        transformed = [
            _detection(1, [0, 0, 10, 10], 0.70),
            _detection(1, [30, 30, 10, 10], 0.40),
        ]
        gt_path = root / "gt.json"
        gt_path.write_text(json.dumps(ground_truth))
        (original_dir / "detections_for_gt.json").write_text(json.dumps(original))
        (variant_dir / "detections_for_gt.json").write_text(json.dumps(transformed))
        return model_dir, gt_path

    def test_lost_tp_uses_threshold_based_confidence_decrease(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            model_dir, gt_path = self._write_inputs(root)
            summary = plot_confidence_change.confidence_summary(
                model_dir, "imgExample_0", "imgExample", gt_path, 0.5
            )

        self.assertAlmostEqual(
            summary.values[plot_confidence_change.LOST_TP_CHANGE][0], 0.55
        )
        self.assertAlmostEqual(
            summary.values[plot_confidence_change.RETAINED_TP_CHANGE][0], -0.2
        )
        self.assertAlmostEqual(
            summary.values[plot_confidence_change.PERSISTENT_FP_CHANGE][0], -0.1
        )

    def test_lost_tp_series_and_plot_are_model_specific(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _, gt_path = self._write_inputs(root)
            _, line_series, lost_tp_series, setting_labels, counts = (
                plot_confidence_change.collect_series(
                    "imgExample",
                    [{"variant_id": "imgExample_0", "strength": 1.0}],
                    ["yolox"],
                    root,
                    gt_path,
                    0.5,
                )
            )
            output_path = (
                root
                / "confidence_change"
                / "lost_TP"
                / "yolox"
                / "imgExample.png"
            )
            generated = plot_confidence_change.plot_lost_tp_rule(
                "imgExample",
                "yolox",
                "Strength",
                lost_tp_series["yolox"],
                setting_labels,
                output_path,
            )

            self.assertTrue(generated)
            self.assertTrue(output_path.is_file())

        retained_points = line_series[plot_confidence_change.RETAINED_TP_CHANGE]["yolox"]
        self.assertEqual(retained_points[0][0], 0.0)
        self.assertAlmostEqual(retained_points[0][1], -0.2)
        self.assertEqual(lost_tp_series["yolox"][0][0], 0.0)
        self.assertAlmostEqual(lost_tp_series["yolox"][0][1][0], 0.55)
        self.assertEqual(counts[plot_confidence_change.LOST_TP_CHANGE], 1)


if __name__ == "__main__":
    unittest.main()
