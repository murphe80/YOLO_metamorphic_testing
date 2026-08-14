import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from analysis.coco_eval import evaluate_coco
from analysis.utils.format_detections import format_detections


def _ground_truth(image_count):
    return {
        "images": [
            {
                "id": image_id,
                "file_name": f"image_{image_id}.jpg",
                "width": 100,
                "height": 100,
            }
            for image_id in range(1, image_count + 1)
        ],
        "annotations": [
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            }
            for image_id in range(1, image_count + 1)
        ],
        "categories": [{"id": 1, "name": "cone"}],
    }


def _perfect_detection(image_id=1):
    return {
        "image_id": image_id,
        "category_id": 1,
        "bbox": [10, 10, 20, 20],
        "score": 0.99,
    }


class CocoImageUniverseTests(unittest.TestCase):
    def _evaluate(self, gt, detections, image_dir=None):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            gt_path = temporary_path / "gt.json"
            dt_path = temporary_path / "detections.json"
            gt_path.write_text(json.dumps(gt))
            dt_path.write_text(json.dumps(detections))
            with contextlib.redirect_stdout(io.StringIO()):
                return evaluate_coco(gt_path, dt_path, image_dir=image_dir)

    def test_list_results_use_all_declared_ground_truth_images(self):
        stats = self._evaluate(_ground_truth(2), [_perfect_detection(1)])

        # One perfect result and one valid zero-detection image. The old code
        # inferred the universe from the result list and incorrectly returned 1.
        self.assertAlmostEqual(float(stats[0]), 51 / 101, places=12)
        self.assertAlmostEqual(float(stats[8]), 0.5, places=12)

    def test_explicit_image_directory_keeps_zero_detection_images(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            image_dir = Path(temporary_dir)
            (image_dir / "image_1.jpg").touch()
            # Extension differences must not change dataset image identity.
            (image_dir / "image_2.png").touch()
            stats = self._evaluate(
                _ground_truth(3),
                [_perfect_detection(1)],
                image_dir=image_dir,
            )

        self.assertAlmostEqual(float(stats[0]), 51 / 101, places=12)
        self.assertAlmostEqual(float(stats[8]), 0.5, places=12)

    def test_empty_list_is_a_valid_zero_detection_result(self):
        stats = self._evaluate(_ground_truth(2), [])

        self.assertEqual(float(stats[0]), 0.0)
        self.assertEqual(float(stats[8]), 0.0)


class DetectionFormattingUniverseTests(unittest.TestCase):
    def test_source_images_are_declared_when_detection_files_are_empty_or_missing(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            image_dir = root / "images"
            detection_dir = root / "detections"
            image_dir.mkdir()
            detection_dir.mkdir()

            for image_name in ("image_1.jpg", "image_2.png", "image_3.jpg"):
                (image_dir / image_name).touch()
            (detection_dir / "image_1.txt").write_text("0 0.9 10 10 30 30\n")
            (detection_dir / "image_2.txt").touch()
            # image_3 intentionally has no result file at all.

            output_path = root / "formatted.json"
            with contextlib.redirect_stdout(io.StringIO()):
                format_detections(
                    detection_dir,
                    output_path,
                    as_ground_truth=True,
                    image_dir=image_dir,
                )
            formatted = json.loads(output_path.read_text())

        self.assertEqual(len(formatted["images"]), 3)
        self.assertEqual(len(formatted["annotations"]), 1)
        self.assertEqual(
            {image["file_name"] for image in formatted["images"]},
            {"image_1.jpg", "image_2.png", "image_3.jpg"},
        )


if __name__ == "__main__":
    unittest.main()
