"""Size strata must preserve matching and the existing per-image weighting."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from analysis.utils.calculate_detection_overlap_metrics import calculate_statistics_for_model_transformation
from analysis.utils.detection_matching import (
    SIZE_BINS, aggregate_three_way_overlap, compute_retention_metrics,
    match_three_way_per_image, size_bin,
)


def detection(image, x, width=20, height=None):
    return dict(image_id=image, category_id=1, score=.9,
                bbox=[x, 0, width, width if height is None else height])


class OverlapSizeTests(unittest.TestCase):
    def aggregate(self, original, transformed, gt):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('original.json', 'transformed.json', 'gt.json')]
            for path, rows in zip(paths, (original, transformed, gt)):
                path.write_text(json.dumps(rows))
            legacy = aggregate_three_way_overlap(*paths)
            sized = aggregate_three_way_overlap(*paths, include_sizes=True)
            self.assertEqual(legacy, {k: v for k, v in sized.items() if k != 'size_breakdown'})
            for metric, total in compute_retention_metrics(legacy).items():
                breakdown = sized['size_breakdown']['metrics'][metric]
                self.assertAlmostEqual(sum(breakdown['rate_contributions'].values()), total)
            return sized

    def test_coco_cutoffs_are_disjoint(self):
        for area, expected in [(0, 'small'), (1023, 'small'), (1024, 'medium'),
                               (9215, 'medium'), (9216, 'large'), (1e11, 'large')]:
            self.assertEqual(size_bin(area), expected)
        for area in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                size_bin(area)

    def test_all_seven_regions_and_empty_input(self):
        boxes = [detection(1, i * 1000, [20, 60, 120][i % 3]) for i in range(7)]
        original = [boxes[i] for i in (0, 1, 2, 4)]
        transformed = [boxes[i] for i in (0, 1, 3, 5)]
        gt = [boxes[i] for i in (0, 2, 3, 6)]
        self.assertTrue(all(v == 1 for v in match_three_way_per_image(original, transformed, gt).values()))
        sized = self.aggregate(original, transformed, gt)
        for region in sized['size_breakdown']['regions'].values():
            self.assertEqual(sum(region['counts'].values()), 1)
            self.assertAlmostEqual(sum(region['weighted_counts'].values()), 1 / 7)
            self.assertAlmostEqual(sum(region['proportions'].values()), 1)
        empty = self.aggregate([], [], [])
        for metric in empty['size_breakdown']['metrics'].values():
            self.assertEqual(metric['proportions'], dict.fromkeys(SIZE_BINS, 0))

    def test_crowded_images_do_not_dominate_size_proportions(self):
        original, transformed, gt = [], [], []
        for image, count, width in [(1, 100, 20), (2, 1, 120)]:
            retained = [detection(image, i * 1000, width) for i in range(count)]
            new = [detection(image, (i + count) * 1000, width) for i in range(count)]
            original.extend(retained)
            gt.extend(retained)
            transformed.extend(retained + new)
        result = self.aggregate(original, transformed, gt)
        metric = result['size_breakdown']['metrics']['new_hallucination_rate']
        self.assertEqual(metric['counts'], dict(small=100, medium=0, large=1))
        self.assertAlmostEqual(metric['proportions']['small'], .5)
        self.assertAlmostEqual(metric['proportions']['large'], .5)
        self.assertAlmostEqual(metric['rate_contributions']['small'], .25)
        self.assertAlmostEqual(metric['rate_contributions']['large'], .25)

    def test_geometric_sizes_use_transformed_pixels_including_lost_objects(self):
        for rule, params, width, scale, expected in [
            ('imgStretch_9', dict(width_ratio=10, height_ratio=10), 20, 10, 'large'),
            ('imgResolution_0', dict(size_ratio=.1), 120, .1, 'small'),
        ]:
            with self.subTest(rule=rule), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                original = [detection(1, 0, width), detection(1, 1000, width)]
                matching = [detection(1, 0, width), detection(1, 2000, width)]
                native = [{**row, 'bbox': [v * scale for v in row['bbox']], 'area': 1}
                          for row in matching]  # stale prediction area must be ignored
                files = {
                    'model/original/detections_original_coco.json': original,
                    'gt.json': original,
                    f'model/{rule}/detections_coco.json': native,
                    f'model/{rule}/detections_coco_transformed.json': matching,
                    f'model/{rule}/configs.json': params,
                }
                for name, data in files.items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data))
                with contextlib.redirect_stdout(io.StringIO()):
                    result = calculate_statistics_for_model_transformation('model', rule, root, root / 'gt.json')
                for metric in ('new_hallucination_rate', 'tp_retention_rate', 'detection_loss_rate'):
                    parts = result['size_breakdown']['metrics'][metric]
                    self.assertEqual(parts['counts'][expected], 1)
                    self.assertEqual(parts['proportions'][expected], 1)
                    self.assertEqual(parts['rate_contributions'][expected], result[metric])


if __name__ == '__main__':
    unittest.main()
