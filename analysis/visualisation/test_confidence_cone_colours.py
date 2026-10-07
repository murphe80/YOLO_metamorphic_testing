"""Regression coverage for confidence matching under cone recolouring."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.visualisation.plot_confidence_change import (
    confidence_summary, collect_series, RETAINED_TP_CHANGE, LOST_TP_CHANGE,
    PERSISTENT_FP_CHANGE,
)


class ConeColourConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.gt = self.root / "gt.json"
        self.gt.write_text(json.dumps({
            "categories": [{"id": i, "name": f"category_{i}"} for i in (1, 2, 3)],
            "annotations": [self.detection(2, 1)],
        }))

    @staticmethod
    def detection(category, score):
        return {"image_id": 42, "category_id": category, "score": score,
                "bbox": [0, 0, 20, 20]}

    def write_detections(self, variant, category, score):
        path = self.root / "model" / variant / "detections_for_gt.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([self.detection(category, score)]))

    def summary(self, variant="blue_to_orange"):
        with contextlib.redirect_stdout(io.StringIO()):
            return confidence_summary(self.root / "model", variant, "cone_colour", self.gt, 0.5).values

    def test_correct_colour_change_is_retained(self):
        self.write_detections("original", 2, 0.9)
        self.write_detections("blue_to_orange", 3, 0.8)
        result = self.summary()
        self.assertAlmostEqual(result[RETAINED_TP_CHANGE][0], -0.1)
        self.assertEqual(result[LOST_TP_CHANGE], [])
        self.assertEqual(result[PERSISTENT_FP_CHANGE], [])
        self.assertEqual(json.loads(self.gt.read_text())["annotations"][0]["category_id"], 2)

    def test_unchanged_blue_prediction_is_lost_tp(self):
        self.write_detections("original", 2, 0.9)
        self.write_detections("blue_to_orange", 2, 0.8)
        result = self.summary()
        self.assertEqual(result[RETAINED_TP_CHANGE], [])
        self.assertAlmostEqual(result[LOST_TP_CHANGE][0], 0.65)

    def test_original_wrong_colour_does_not_become_retained_tp(self):
        self.write_detections("original", 3, 0.9)
        self.write_detections("blue_to_orange", 3, 0.8)
        self.assertTrue(all(not values for values in self.summary().values()))

    def test_persistent_fp_uses_correct_labels_for_each_image(self):
        self.write_detections("original", 1, 0.9)
        self.write_detections("blue_to_orange", 2, 0.8)
        self.assertAlmostEqual(self.summary()[PERSISTENT_FP_CHANGE][0], -0.1)

    def test_multiple_rules_share_categorical_axis(self):
        self.write_detections("original", 2, 0.9)
        self.write_detections("blue_to_orange", 3, 0.8)
        self.write_detections("blue_to_yellow", 1, 0.7)
        with contextlib.redirect_stdout(io.StringIO()):
            label, lines, _, ticks, counts = collect_series(
                "cone_colour", [{"variant_id": name} for name in ("blue_to_orange", "blue_to_yellow")],
                ["model"], self.root, self.gt, 0.5,
            )
        self.assertEqual(label, "Cone colour rule")
        self.assertEqual(ticks, ["Blue → Orange", "Blue → Yellow"])
        self.assertEqual([x for x, _ in lines[RETAINED_TP_CHANGE]["model"]], [0, 1])
        self.assertEqual(counts[RETAINED_TP_CHANGE], 2)

    def test_non_colour_rule_keeps_original_gt(self):
        self.write_detections("original", 2, 0.9)
        self.write_detections("imgGrayscale", 2, 0.8)
        self.assertAlmostEqual(self.summary("imgGrayscale")[RETAINED_TP_CHANGE][0], -0.1)


if __name__ == "__main__":
    unittest.main()
