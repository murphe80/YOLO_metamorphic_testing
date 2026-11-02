"""Saved experiment settings must survive later transformation-config changes."""
import unittest

from analysis.visualisation.render_sample_equivalence import rebuild_plot_config


class SavedPlotConfigTests(unittest.TestCase):
    def test_preserves_legacy_and_cohort_settings_without_expanding_them(self):
        rows = [
            dict(rule='imgColorWheel', variant_id='imgColorWheel_0',
                 varying_parameter='hue_rotation', setting={'hue_rotation': 0}),
            dict(rule='imgColorWheel__blue_to_yellow',
                 variant_id='imgColorWheel_6__blue_to_yellow',
                 varying_parameter='hue_rotation', setting={'hue_rotation': 120}),
            dict(rule='imgGrayscale', variant_id='imgGrayscale_0',
                 varying_parameter=None, setting={}),
        ]
        config = rebuild_plot_config(rows + rows)
        self.assertEqual(list(config), [row['rule'] for row in rows])
        for row in rows:
            self.assertEqual(config[row['rule']]['variants'],
                             [{**row['setting'], 'variant_id': row['variant_id']}])

    def test_rejects_conflicting_saved_settings(self):
        row = dict(rule='imgDarkBright', variant_id='imgDarkBright_0',
                   varying_parameter='gamma', setting={'gamma': .5})
        with self.assertRaisesRegex(ValueError, 'Inconsistent saved setting'):
            rebuild_plot_config([row, {**row, 'setting': {'gamma': 2}}])


if __name__ == '__main__':
    unittest.main()
