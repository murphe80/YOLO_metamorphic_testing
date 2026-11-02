"""Check three-way reassessment using existing RQ2b and validation JSON only."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.rq2 import rq2b_validate_brightness_method as rq


def scores(value):
    return dict.fromkeys(rq.BRIGHTNESS_METHODS, value)


def records():
    return [dict(model='yolov8', image_name=f'calibration_{i}.png',
                 scores_right=scores(dark), scores_left=scores(bright),
                 ap_right=.5, ap_left=.9 if i == 0 else .5)
            for i, (dark, bright) in enumerate(zip([.1, .2, .3], [.7, .8, .9]))]


def observations():
    return [dict(image_name=f'validation_{i}.png', kind='transformed',
                 side='dark' if i % 2 else 'bright', gamma=2 if i % 2 else .5,
                 ap=ap, actual_failure=ap < .5, brightness_scores=scores(value),
                 method_predictions={'old_prediction': True})
            for i, (ap, value) in enumerate((ap, value) for ap in [.8, .2] for value in [.5, .2, .99])]


class AssessmentTests(unittest.TestCase):
    def fixture(self, root):
        boundary = root / 'boundaries.json'
        evaluation = root / 'evaluation.json'
        boundary.write_text(json.dumps(dict(models=['yolov8'], per_image_results=records())))
        evaluation.write_text(json.dumps(dict(
            schema_version=3, model='yolov8', dataset='val', ap_threshold=.5,
            validation_observations=observations(),
            analysis_filter={'minimum_ap_inclusive': .1},
            holdout_summary={'transformed_images': 6})))
        return boundary, evaluation

    def test_distribution_uses_all_saved_brightness_scores_and_sample_sd(self):
        bounds = rq.derive_ranges(records())
        for b in bounds.values():
            self.assertEqual(b['dark_boundary']['n'], 3)
            self.assertEqual(b['bright_boundary']['n'], 3)  # no new AP-based selection
            self.assertAlmostEqual(b['dark_boundary']['mean'], .2)
            self.assertAlmostEqual(b['dark_boundary']['sample_standard_deviation'], .1)
            self.assertAlmostEqual(b['lower'], .2204)
            self.assertAlmostEqual(b['upper'], .7796)
            self.assertTrue(b['valid_interval'])

    def test_load_boundaries_selects_model_and_handles_missing_side(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary, _ = self.fixture(root)
            data = json.loads(boundary.read_text())
            data['models'].append('yolov12')
            data['per_image_results'].append(dict(model='yolov12', image_name='other',
                                                 scores_left=scores(100), scores_right=scores(50)))
            data['per_image_results'][0]['scores_right'] = None
            boundary.write_text(json.dumps(data))
            selected = rq.load_boundary_records(boundary, 'yolov8')
            self.assertEqual(len(selected), 3)
            bounds = rq.derive_ranges(selected)['hsv_value']
            self.assertEqual(bounds['dark_boundary']['n'], 2)
            self.assertEqual(bounds['bright_boundary']['n'], 3)
            with self.assertRaises(ValueError):
                rq.load_boundary_records(boundary, 'yolov5')

    def test_predictions_endpoints_overlap_missing_samples_and_reversed_means(self):
        bounds = dict(available=True, unavailable_reason=None,
                      dark_boundary=dict(lower=.1, upper=.3), bright_boundary=dict(lower=.7, upper=.9))
        for value, expected in [(.05, 'out_of_range'), (.1, 'uncertain'), (.3, 'uncertain'),
                                (.5, 'in_range'), (.7, 'uncertain'), (.9, 'uncertain'), (.95, 'out_of_range')]:
            self.assertEqual(rq.predict_brightness(value, bounds)[0], expected)
        bounds['dark_boundary']['upper'] = .8
        self.assertEqual(rq.predict_brightness(.5, bounds)[0], 'uncertain')
        for b in rq.derive_ranges([]).values():
            self.assertEqual(rq.predict_brightness(.5, b)[0], 'uncertain')
        reversed_records = records()
        for row in reversed_records:
            row['scores_left'], row['scores_right'] = row['scores_right'], row['scores_left']
        self.assertTrue(all(not b['available'] for b in rq.derive_ranges(reversed_records).values()))

    def test_uncertain_excluded_from_confusion_and_accuracy(self):
        rows = observations()
        ranges = rq.derive_ranges(records(), sigma_multiplier=1)
        metric = rq.classify(rows, ranges)['hsv_value']
        self.assertEqual(metric['confusion_matrix']['counts'], [[1, 1], [1, 1]])
        self.assertEqual(metric['confusion_matrix']['predicted_labels'], ['in_range', 'out_of_range'])
        self.assertEqual(metric['definite_count'], 4)
        self.assertEqual(metric['uncertain_count'], 2)
        self.assertNotIn('uncertain_correct', metric)
        self.assertNotIn('overall_correctness', metric)
        self.assertAlmostEqual(metric['coverage'], 4 / 6)
        self.assertEqual(metric['selective_accuracy'], .5)
        self.assertAlmostEqual(metric['correct_definite_fraction'], 2 / 6)
        self.assertAlmostEqual(metric['false_in_range_rate'], 1 / 3)
        uncertain_prediction = rows[1]['method_predictions']['hsv_value']
        self.assertIsNone(uncertain_prediction['predicted_failure'])
        self.assertIsNone(uncertain_prediction['correct'])
        self.assertAlmostEqual(uncertain_prediction['ap_residual'], .3)
        residual = metric['uncertain_residual_summary']
        self.assertAlmostEqual(residual['mean'], 0)
        self.assertAlmostEqual(residual['mean_absolute'], .3)
        self.assertAlmostEqual(residual['rmse'], .3)
        metric = rq.classify([], ranges)['hsv_value']
        self.assertIsNone(metric['selective_accuracy'])
        self.assertEqual(metric['uncertain_residual_summary'], {'n': 0})

    def test_uncertain_residuals_are_continuous_and_not_correctness_labels(self):
        aps = [0, .45, .48, .5, .52, .55, 1]
        rows = [dict(ap=ap, actual_failure=ap < .5, brightness_scores=scores(.2)) for ap in aps]
        ranges = rq.derive_ranges(records())
        metric = rq.classify(rows, ranges)['hsv_value']
        for row, ap in zip(rows, aps):
            prediction = row['method_predictions']['hsv_value']
            self.assertIsNone(prediction['correct'])
            self.assertAlmostEqual(prediction['ap_residual'], ap - .5)
        self.assertEqual(metric['confusion_matrix']['counts'], [[0, 0], [0, 0]])
        self.assertIsNone(metric['selective_accuracy'])
        self.assertEqual(metric['coverage'], 0)
        self.assertEqual(metric['uncertain_residual_summary']['n'], 7)
        self.assertAlmostEqual(metric['uncertain_residual_summary']['mean_absolute'], 1.14 / 7)
        definite = [dict(ap=ap, actual_failure=ap < .5, brightness_scores=scores(value))
                    for ap, value in [(.48, .99), (.52, .5), (.5, .5), (.5, .99)]]
        metric = rq.classify(definite, ranges)['hsv_value']
        self.assertEqual(metric['confusion_matrix']['counts'], [[2, 1], [0, 1]])
        self.assertEqual(metric['uncertain_residual_summary'], {'n': 0})
        self.assertEqual([r['method_predictions']['hsv_value']['correct'] for r in definite],
                         [True, True, True, False])
        with tempfile.TemporaryDirectory() as directory:
            for name, observations_ in [('uncertain', rows), ('definite', definite), ('empty', [])]:
                path = Path(directory) / f'{name}.pdf'
                rq.plot_uncertain_residuals(path, dict(dataset='val', model='yolov8',
                    operating_ranges=ranges, validation_observations=observations_))
                self.assertGreater(path.stat().st_size, 1000)

    def test_invalid_or_missing_inputs_fail_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary, evaluation = self.fixture(root)
            with self.assertRaises(FileNotFoundError):
                rq.load_validation_results(root / 'missing.json', 'val', 'yolov8')
            with self.assertRaises(ValueError):
                rq.load_validation_results(evaluation, 'val', 'yolov12')
            with self.assertRaises(ValueError):
                rq.load_validation_results(evaluation, 'test', 'yolov8')
            data = json.loads(evaluation.read_text())
            data['validation_observations'][0]['brightness_scores'].pop('hsv_value')
            evaluation.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'invalid brightness scores'):
                rq.load_validation_results(evaluation, 'val', 'yolov8')

    def test_main_reassesses_saved_schema3_without_source_images_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary, evaluation = self.fixture(root)
            before = (boundary.read_bytes(), evaluation.read_bytes())
            args = rq.parse_args(['--rq2b-results', str(boundary), '--validation-results', str(evaluation),
                                 '--output-dir', str(root / 'output'), '--sigma-multiplier', '1'])
            with patch.object(rq, 'parse_args', return_value=args), \
                 patch.object(rq.cv2, 'imread', side_effect=AssertionError('Read an image')), \
                 patch('utils.imgDarkBright_tuning.brightness_methods.calculate_all_brightness_scores',
                       side_effect=AssertionError('Recalculated brightness')):
                rq.main()
            self.assertEqual(before, (boundary.read_bytes(), evaluation.read_bytes()))
            out = root / 'output/val/yolov8/three_class'
            data = json.loads((out / 'validation_results.json').read_text())
            self.assertEqual(data['schema_version'], 6)
            self.assertEqual(data['evaluation_rule']['uncertain'],
                             dict(residual='AP - 0.5', reference_ap=.5, correctness_assessed=False))
            self.assertFalse(data['inference_rerun'])
            self.assertFalse(data['brightness_scoring_rerun'])
            self.assertEqual(data['analysis_filter'], {'minimum_ap_inclusive': .1})
            self.assertEqual(data['holdout_summary'], {'transformed_images': 6})
            self.assertEqual(data['calibration']['matching_image_names'], 0)
            for row in data['validation_observations']:
                self.assertNotIn('old_prediction', row['method_predictions'])
            for m in data['metrics'].values():
                self.assertEqual(m['prediction_counts'], dict(in_range=2, uncertain=2, out_of_range=2))
            report = (out / 'report.md').read_text()
            self.assertIn('without additional AP filtering', report)
            self.assertIn('AP - 0.5', report)
            self.assertNotIn('0.45 < AP < 0.55', report)
            self.assertIn('Only definite predictions contribute', report)
            self.assertNotIn('first success-to-failure', report)
            for name in ['brightness_validation.pdf', 'boundary_distributions.pdf',
                         'confusion_matrices_overall.pdf', 'confusion_matrices_brightened.pdf',
                         'confusion_matrices_darkened.pdf', 'uncertain_residuals.pdf']:
                self.assertGreater((out / name).stat().st_size, 1000)
                self.assertTrue((out / name).read_bytes().startswith(b'%PDF-'))
                self.assertNotIn(b'/Subtype /Image', (out / name).read_bytes())

    def test_legacy_binary_consumers_remain_supported(self):
        rows = observations()
        ranges = {'hsv_value': dict(lower=.3, upper=.7, valid_interval=True)}
        self.assertIn('tp', rq.classify(rows, ranges)['hsv_value'])
        data = dict(validation_observations=copy.deepcopy(rows), operating_ranges=ranges, ap_threshold=.5)
        self.assertIn('Brightened images', rq.directional_report(data))


if __name__ == '__main__':
    unittest.main()
