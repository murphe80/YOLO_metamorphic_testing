"""Regression coverage for confidence matching under cone recolouring."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.visualisation.plot_confidence_change import (
    confidence_summary, collect_series, RETAINED_TP_CHANGE, LOST_TP_CHANGE,
    PERSISTENT_FP_CHANGE, NEW_FP_CONFIDENCE, plot_new_fp_rule,
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
        self.assertEqual(result[NEW_FP_CONFIDENCE], [])
        self.assertEqual(json.loads(self.gt.read_text())["annotations"][0]["category_id"], 2)

    def test_unchanged_blue_prediction_is_lost_tp(self):
        self.write_detections("original", 2, 0.9)
        self.write_detections("blue_to_orange", 2, 0.8)
        result = self.summary()
        self.assertEqual(result[RETAINED_TP_CHANGE], [])
        self.assertAlmostEqual(result[LOST_TP_CHANGE][0], 0.65)
        self.assertEqual(result[NEW_FP_CONFIDENCE], [0.8])

    def test_original_wrong_colour_does_not_become_retained_tp(self):
        self.write_detections("original", 3, 0.9)
        self.write_detections("blue_to_orange", 3, 0.8)
        self.assertTrue(all(not values for values in self.summary().values()))

    def test_persistent_fp_uses_correct_labels_for_each_image(self):
        self.write_detections("original", 1, 0.9)
        self.write_detections("blue_to_orange", 2, 0.8)
        self.assertAlmostEqual(self.summary()[PERSISTENT_FP_CHANGE][0], -0.1)
        self.assertEqual(self.summary()[NEW_FP_CONFIDENCE], [])

    def test_new_fps_exclude_persistent_matches_and_true_positives(self):
        self.write_detections("original", 2, .9)
        self.write_detections("imgGrayscale_0", 2, .8)
        original_path = self.root / 'model/original/detections_for_gt.json'
        transformed_path = self.root / 'model/imgGrayscale_0/detections_for_gt.json'
        fp = dict(self.detection(1, .6), bbox=[100, 100, 20, 20])
        original_path.write_text(json.dumps([self.detection(2, .9), fp]))
        transformed_path.write_text(json.dumps([
            self.detection(2, .8), dict(fp, score=.7),
            dict(fp, score=.4, bbox=[200, 200, 20, 20]),
            dict(fp, score=.3),  # duplicate FP cannot share the persistent match
        ]))
        result = self.summary('imgGrayscale_0')
        self.assertEqual(sorted(result[NEW_FP_CONFIDENCE]), [.3, .4])
        self.assertEqual(len(result[PERSISTENT_FP_CHANGE]), 1)
        self.assertEqual(len(result[RETAINED_TP_CHANGE]), 1)

    def test_new_fp_collection_and_plot_include_cone_colour(self):
        self.write_detections('original', 2, .9)
        self.write_detections('blue_to_orange', 2, .8)
        new_series = {'model': []}
        with contextlib.redirect_stdout(io.StringIO()):
            label, _, _, ticks, counts = collect_series(
                'cone_colour', [{'variant_id': 'blue_to_orange'}],
                ['model'], self.root, self.gt, .5, new_fp_series=new_series)
        self.assertEqual(new_series, {'model': [(0., [.8])]})
        self.assertEqual(counts[NEW_FP_CONFIDENCE], 1)
        out = self.root / 'new_FP/cone_colour.pdf'
        self.assertTrue(plot_new_fp_rule('cone_colour', label, new_series, ticks, out))
        self.assertGreater(out.stat().st_size, 1000)
        self.assertTrue(plot_new_fp_rule('cone_colour', label, {'model': [(0., [])]}, ticks, out))

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
