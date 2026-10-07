"""Regression coverage for reviewed hue mappings and many-to-one cohorts."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.utils.color_wheel import metadata, select_predictions, evaluate_files, baseline_key
from analysis.utils.calculate_detection_overlap_metrics import calculate_statistics_for_model_transformation
from analysis.visualisation.plot_confidence_change import confidence_summary, RETAINED_TP_CHANGE, LOST_TP_CHANGE


class ColorWheelTests(unittest.TestCase):
    def test_reviewed_mappings(self):
        expected = [
            {2: 2, 3: 3, 1: 1}, {2: 2}, {}, {},
            {2: 3, 1: 2}, {}, {2: 1}, {},
            {3: 3, 1: 3}, {2: 2, 3: 3, 1: 1},
        ]
        for index, mapping in enumerate(expected):
            record = metadata(f"imgColorWheel_{index}")
            self.assertEqual({m['source_category_id']: m['target_category_id'] for m in record['included_mappings']}, mapping)
            self.assertEqual(record['hue_rotation'], index * 20)
            self.assertIn('large_orange_cone', record['excluded_classes'])
            self.assertTrue(all(m['source_category_id'] != 4 for m in record['included_mappings'] + record['excluded_mappings']))
            self.assertEqual(record['status'], 'included' if mapping else 'excluded')

    def test_sample_equivalence_keeps_mappings_separate(self):
        from analysis.utils.calculate_sample_equivalence import load_config, DEFAULT_CONFIG, variant_pairs
        config = load_config(DEFAULT_CONFIG, ['imgColorWheel'])
        self.assertEqual(load_config(DEFAULT_CONFIG, list(config)), config)
        pairs = variant_pairs(config)
        self.assertEqual(len(pairs), 12)
        for rule, variant in pairs:
            self.assertIn('__', rule)
            self.assertNotIn(variant.split('__')[0], ['imgColorWheel_2', 'imgColorWheel_3', 'imgColorWheel_5', 'imgColorWheel_7'])
        self.assertEqual(baseline_key('imgColorWheel_4__yellow_to_blue'), 'original__yellow_to_yellow')
        self.assertTrue(all('orange_to_blue' not in variant for _, variant in pairs))
        self.assertTrue(all('orange_to_yellow' not in variant for _, variant in pairs))
        self.assertTrue(all('large_orange' not in variant for _, variant in pairs))

    def fixture(self, root, variant='imgColorWheel_4', target=2):
        refs = [dict(id=i, image_id=1, category_id=c, bbox=[i*40, 0, 20, 20], area=400, iscrowd=0, score=.8)
                for i, c in enumerate((3, 4, 1, 2, 5, 6), 1)]
        gt = {'images': [{'id': 1, 'file_name': 'one.jpg', 'width': 400, 'height': 100}],
              'categories': [{'id': i, 'name': str(i)} for i in range(1, 7)], 'annotations': refs}
        # Excluded cones also predict the target, and a background FP must not
        # disappear from precision scoring.
        detections = [{**a, 'category_id': target, 'score': .9} for a in refs]
        detections.append(dict(id=7, image_id=1, category_id=target, bbox=[320, 0, 20, 20], score=.95))
        for path, data in [('gt.json', gt), ('model/original/detections_original_coco.json', gt),
                           ('model/original/detections_for_gt.json', refs),
                           (f'model/{variant}/detections_coco.json', detections),
                           (f'model/{variant}/detections_for_gt.json', detections)]:
            dest = root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(data))
        return gt, detections

    def test_shared_target_separate_sources_and_exclusions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gt, detections = self.fixture(root, 'imgColorWheel_8', 3)
            before = json.dumps(gt)
            for mapping in metadata('imgColorWheel_8')['included_mappings']:
                selected = select_predictions(detections, gt, mapping)
                self.assertEqual(len(selected), 2)  # This source + background FP.
            self.assertEqual(json.dumps(gt), before)
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_files(root/'gt.json', root/'model/imgColorWheel_8/detections_coco.json', 'imgColorWheel_8')
                overlap = calculate_statistics_for_model_transformation('model', 'imgColorWheel_8', root, root/'gt.json')
            self.assertNotIn('AP', result)
            self.assertEqual(len(result['class_shifts']), 2)
            for shift in result['class_shifts'].values():
                self.assertAlmostEqual(shift['AR100'], 1)
                self.assertLess(shift['AP'], 1)  # Background FP still penalised.
                self.assertEqual(shift['reference_count'], 1)
            for shift in overlap['class_shifts'].values():
                self.assertEqual(shift['tp_retention_rate'], 1)
            for mapping in result['included_mappings']:
                summary = confidence_summary(root/'model', 'imgColorWheel_8', 'imgColorWheel', root/'gt.json', .5, mapping)
                self.assertEqual(len(summary.values[RETAINED_TP_CHANGE]), 1)
                self.assertEqual(summary.values[LOST_TP_CHANGE], [])

    def test_hue_80_excludes_unreliable_orange_cohort(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gt, detections = self.fixture(root)
            record = metadata('imgColorWheel_4')
            orange = next(m for m in record['excluded_mappings'] if m['source_category_id'] == 3)
            self.assertIsNone(orange['target_category_id'])
            self.assertIn('green/teal', orange['exclusion_reason'])
            mapping = next(m for m in record['included_mappings'] if m['key'] == 'yellow_to_blue')
            selected = select_predictions(detections, gt, mapping)
            self.assertEqual([a['id'] for a in selected], [3, 7])
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_files(root/'gt.json', root/'model/imgColorWheel_4/detections_coco.json', 'imgColorWheel_4')
                overlap = calculate_statistics_for_model_transformation('model', 'imgColorWheel_4', root, root/'gt.json')
            self.assertEqual(set(result['class_shifts']), {'blue_to_orange', 'yellow_to_blue'})
            self.assertEqual(set(overlap['class_shifts']), {'blue_to_orange', 'yellow_to_blue'})

    def test_excluded_variant_has_no_scores(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            result = evaluate_files(root/'gt.json', root/'model/imgColorWheel_4/detections_coco.json', 'imgColorWheel_2')
            self.assertEqual(result['class_shifts'], {})
            self.assertNotIn('AP', result)

    def test_wrong_class_is_lost_tp_and_new_hallucination(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gt, _ = self.fixture(root)
            # The blue cone becomes yellow, although hue=80 expects orange.
            # Predictions at every other source location must be excluded.
            detections = [{**a, 'category_id': 1} for a in gt['annotations']]
            path = root/'model/imgColorWheel_4/detections_coco.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(detections))
            result = calculate_statistics_for_model_transformation(
                'model', 'imgColorWheel_4', root, root/'gt.json')
            shift = result['class_shifts']['blue_to_orange']
            self.assertEqual(shift['tp_retention_rate'], 0)
            self.assertEqual(shift['new_hallucination_rate'], 1)
            self.assertEqual(shift['orig_gt'], .5)
            self.assertEqual(shift['trans_only'], .5)
            # A correct orange detection alongside the wrong yellow detection
            # retains the TP, but yellow still contributes to numerator and
            # both predictions contribute to the hallucination denominator.
            blue = next(a for a in gt['annotations'] if a['category_id'] == 2)
            detections.append({**blue, 'category_id': 3})
            path.write_text(json.dumps(detections))
            mixed = calculate_statistics_for_model_transformation(
                'model', 'imgColorWheel_4', root, root/'gt.json')
            shift = mixed['class_shifts']['blue_to_orange']
            self.assertEqual(shift['tp_retention_rate'], 1)
            self.assertEqual(shift['new_hallucination_rate'], .5)

    def test_overlap_keeps_wrong_class_background_and_fallback_ownership(self):
        with TemporaryDirectory() as directory:
            gt, _ = self.fixture(Path(directory))
            mapping = metadata('imgColorWheel_4')['included_mappings'][0]
            predictions = [dict(image_id=1, category_id=1, bbox=[400+i*40, 0, 20, 20])
                           for i in range(3)]
            fallback = [{**predictions[0], 'category_id': 2},
                        {**predictions[1], 'category_id': 3}]
            selected = select_predictions(predictions, gt, mapping, fallback=fallback,
                                          include_wrong_classes=True)
            self.assertEqual(selected, [predictions[0], predictions[2]])
            self.assertEqual(select_predictions(predictions, gt, mapping), [])

    def test_unchanged_predictions_fail_when_a_colour_shift_is_expected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gt, _ = self.fixture(root)
            path = root/'model/imgColorWheel_4/detections_coco.json'
            path.write_text(json.dumps(gt['annotations']))
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_files(root/'gt.json', path, 'imgColorWheel_4')
            for shift in result['class_shifts'].values():
                self.assertEqual(shift['AP'], 0)
                self.assertEqual(shift['AR100'], 0)


if __name__ == '__main__':
    unittest.main()
