"""Check that charts use additive contributions and reject stale analysis."""
import unittest

from analysis.visualisation.plot_detection_overlap_metrics import metric_size_contributions


class SizeChartTests(unittest.TestCase):
    def test_stacks_use_rate_contributions_not_within_bar_proportions(self):
        metric = 'new_hallucination_rate'
        record = {metric: .2, 'size_breakdown': {'metrics': {metric: {
            'proportions': dict(small=.6, medium=.3, large=.1),
            'rate_contributions': dict(small=.12, medium=.06, large=.02),
        }}}}
        self.assertEqual(metric_size_contributions(record, metric), [.12, .06, .02])
        record['size_breakdown']['metrics'][metric]['rate_contributions']['small'] = .6
        with self.assertRaisesRegex(ValueError, 'do not sum'):
            metric_size_contributions(record, metric)

    def test_legacy_results_require_regeneration(self):
        with self.assertRaisesRegex(ValueError, 'rerun'):
            metric_size_contributions({'new_tp_rate': .2}, 'new_tp_rate')


if __name__ == '__main__':
    unittest.main()
