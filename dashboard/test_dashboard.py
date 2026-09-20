"""Integration checks for result isolation, metric semantics and the HTTP boundary."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from data import build, metric_values
from serve import Handler


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.roots = {'paper': self.root / 'original_results', 'local': self.root / 'results'}
        for path in self.roots.values():
            path.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_sources_are_isolated_and_raw_stats_take_precedence(self):
        dataset = self.roots['paper'] / 'rq1/test'
        self.write(dataset / 'custom_model/imgDarkBright_0/original_vs_transformed/stats.json',
                   {'AP': .7, 'AR100': .8})
        self.write(dataset / 'parameter_variations/metrics_original_vs_transformed.json', {
            'comparison': 'original_vs_transformed', 'rules': {'imgDarkBright': {'settings': [{
                'variant_id': 'imgDarkBright_0', 'setting': {'gamma': 42},
                'models': {'custom_model': {'AP': .2, 'AR': .3}}}]}}})
        paper = build('paper', self.roots)
        row = paper['datasets']['test']['rows'][0]
        self.assertEqual((row['AP'], row['AR']), (.7, .8))
        self.assertEqual(row['setting'], {'gamma': 42})
        self.assertEqual(build('local', self.roots)['datasets'], {})

    def test_missing_metrics_are_not_zero(self):
        self.assertEqual(metric_values({'AP': -1, 'AR100': float('nan')}), {'AP': None, 'AR': None})
        self.assertEqual(metric_values({'AP': 0, 'AR100': 1}), {'AP': 0, 'AR': 1})

    def test_malformed_result_does_not_hide_other_results(self):
        p = self.roots['paper'] / 'rq1/test/model/imgGrayscale_0/original_vs_transformed/stats.json'
        p.parent.mkdir(parents=True)
        p.write_text('{unfinished')
        result = build('paper', self.roots)
        self.assertEqual(len(result['warnings']), 1)
        self.assertIsNone(result['datasets']['test']['rows'][0]['AP'])

    def test_sample_metadata_and_classifications_are_preserved(self):
        self.write(self.roots['local'] / 'rq3/sample_equivalence/results.json', {
            'metadata': {'sample_fraction': .2, 'equivalence_margin': .025},
            'parameter_results': [{'classification': 'inconclusive', 'metric': 'AP'}]})
        result = build('local', self.roots)['rq3']
        self.assertEqual(result['metadata']['sample_fraction'], .2)
        self.assertEqual(result['parameter_results'][0]['classification'], 'inconclusive')

    def test_refresh_reads_new_results(self):
        self.assertFalse(build('local', self.roots)['comparisons'])
        self.write(self.roots['local'] / 'rq2/rq2a/comparison/model/comparison.json',
                   {'summary': {'num_images': 7}})
        self.assertEqual(build('local', self.roots)['comparisons']['model']['num_images'], 7)

    def test_http_serves_only_dashboard_and_selected_results(self):
        (self.root / 'secret.json').write_text('{"secret": true}')
        (self.roots['paper'] / 'escape.json').symlink_to(self.root / 'secret.json')
        self.write(self.roots['paper'] / 'allowed.json', {'ok': True})
        with patch('serve.ROOTS', self.roots):
            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(base + '/') as response:
                    self.assertIn(b'Research dashboard', response.read())
                with urlopen(base + '/results/paper/allowed.json') as response:
                    self.assertEqual(json.load(response), {'ok': True})
                for route in ('/results/paper/%2e%2e/secret.json', '/results/paper/escape.json',
                              '/results/nope/allowed.json', '/data.py', '/api/nope'):
                    with self.assertRaises(HTTPError):
                        urlopen(base + route)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
