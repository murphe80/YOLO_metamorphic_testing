"""Regression tests for class-changing rules in detection overlap analysis."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.utils.calculate_detection_overlap_metrics import (
    calculate_statistics_for_model_transformation,
)


class ConeColourOverlapTests(unittest.TestCase):
    def calculate(self, rule, source, predicted, generic=False, category_aware=True,
                  extra_references=()):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            categories = [
                {"id": cid, "name": f"category_{cid}" if generic else name}
                for cid, name in [(1, "yellow_cone"), (2, "blue_cone"), (3, "orange_cone")]
            ]
            original = {"categories": categories, "annotations": [
                {"image_id": 1, "category_id": source, "bbox": [0, 0, 20, 20]}
            ] + list(extra_references)}
            original_path = root / "model/original/detections_original_coco.json"
            gt_path = root / "gt.json"
            trans_path = root / "model" / rule / "detections_coco.json"
            for path, data in [
                (original_path, original), (gt_path, original),
                (trans_path, [{"image_id": 1, "category_id": predicted,
                               "bbox": [0, 0, 20, 20], "score": 0.9}]),
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            before = original_path.read_bytes(), gt_path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                result = calculate_statistics_for_model_transformation(
                    "model", rule, root, gt_path, require_category_match=category_aware
                )
            self.assertEqual(before, (original_path.read_bytes(), gt_path.read_bytes()))
            return result

    def test_expected_colour_changes_are_retained(self):
        colours = {"yellow": 1, "blue": 2, "orange": 3}
        for source, source_id in colours.items():
            for target, target_id in colours.items():
                if source == target:
                    continue
                for generic in (False, True):
                    with self.subTest(source=source, target=target, generic=generic):
                        result = self.calculate(f"{source}_to_{target}", source_id, target_id, generic)
                        self.assertEqual(result["tp_retention_rate"], 1)
                        self.assertEqual(result["new_hallucination_rate"], 0)
                        self.assertEqual(result["detection_loss_rate"], 0)

    def test_unchanged_source_colour_is_an_error(self):
        result = self.calculate("yellow_to_orange", 1, 1)
        self.assertEqual(result["new_hallucination_rate"], 1)
        self.assertEqual(result["tp_retention_rate"], 0)

    def test_other_rules_still_require_original_class(self):
        self.assertEqual(self.calculate("imgGrayscale", 2, 3)["new_hallucination_rate"], 1)
        self.assertEqual(self.calculate("imgGrayscale", 2, 2)["tp_retention_rate"], 1)

    def test_spatial_matching_ignores_colour(self):
        result = self.calculate("yellow_to_orange", 1, 1, category_aware=False)
        self.assertEqual(result["tp_retention_rate"], 1)

    def test_untransformed_classes_do_not_enter_source_retention(self):
        extras = [{"image_id": 1, "category_id": category, "bbox": [40*i, 0, 20, 20]}
                  for i, category in enumerate((2, 3, 4), 1)]
        result = self.calculate("yellow_to_blue", 1, 2, extra_references=extras)
        self.assertEqual(result["tp_retention_rate"], 1)
        self.assertEqual(result["orig_gt"], 0)
        self.assertEqual(result["cohort"], "source_only")

    def test_cone_and_hue_share_ownership_and_wrong_class_policy(self):
        def det(category, x, score=.9):
            return dict(image_id=1, category_id=category, bbox=[x, 0, 20, 20], score=score)
        gt = {"annotations": [det(1, 0), det(2, 40), det(4, 80)]}
        original = {"annotations": gt["annotations"] + [det(3, 120), det(1, 160)]}
        transformed = [det(2, 0), det(1, 0, .7), det(2, 40), det(1, 80),
                       det(2, 120), det(1, 160), det(3, 200)]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = [(root/"gt.json", gt),
                     (root/"model/original/detections_original_coco.json", original)]
            files.extend((root/"model"/rule/"detections_coco.json", transformed)
                         for rule in ("yellow_to_blue", "imgColorWheel_4"))
            for path, data in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            cone = calculate_statistics_for_model_transformation("model", "yellow_to_blue", root, root/"gt.json")
            hue = calculate_statistics_for_model_transformation("model", "imgColorWheel_4", root, root/"gt.json")["class_shifts"]["yellow_to_blue"]
            for key in hue:
                self.assertEqual(cone[key], hue[key], key)
            self.assertEqual(cone["tp_retention_rate"], 1)
            self.assertAlmostEqual(cone["new_hallucination_rate"], .75)


if __name__ == "__main__":
    unittest.main()
