"""Check source exclusion and the distinction between class and spatial errors."""

import unittest

from analysis.visualisation.plot_color_wheel_hallucination import calculate_mapping
from analysis.visualisation.plot_cone_color_hallucination import calculate_rule


class ColorWheelHallucinationTests(unittest.TestCase):
    @staticmethod
    def detection(category, x, score=.9):
        return dict(category_id=category, bbox=[x, 0, 20, 20], score=score)

    def test_excluded_sources_and_fallback_do_not_enter_bars(self):
        d = self.detection
        gt = {'one': [d(1, 0), d(3, 40)]}
        original = {'one': gt['one'] + [d(3, 80)]}
        transformed = {'one': [d(2, 0), d(1, 0, .7), d(2, 40), d(1, 80), d(4, 120)]}
        result = calculate_mapping(original, transformed, gt, 'imgColorWheel_4', 'yellow_to_blue')
        self.assertEqual(result['detections'], 3)
        self.assertEqual(result['counts'], [2, 1, 1, 0, 0, 1])
        self.assertAlmostEqual(result['percentages'][0], 200 / 3)
        self.assertAlmostEqual(result['percentages'][1], 100 / 3)
        cone_result = calculate_rule(original, transformed, gt, 'yellow_to_blue')
        for key in ('detections', 'counts', 'percentages', 'images', 'other_count'):
            self.assertEqual(cone_result[key], result[key], key)

    def test_unreliable_mapping_is_rejected(self):
        for variant, key in [('imgColorWheel_4', 'orange_to_blue'),
                             ('imgColorWheel_1', 'orange_to_yellow')]:
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, 'not an included'):
                calculate_mapping({}, {}, {}, variant, key)

    def test_original_cone_rule_still_remaps_expected_classes(self):
        d = self.detection
        result = calculate_rule({'one': [d(2, 0)]}, {'one': [d(3, 0)]},
                                {'one': [d(2, 0)]}, 'blue_to_orange')
        self.assertEqual(result['counts'], [0, 1, 0, 0, 0, 0])
        self.assertEqual(result['percentages'], [0, 100, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
