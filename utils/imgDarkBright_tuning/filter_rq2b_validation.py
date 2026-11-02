"""Filter validation for rq2b to only include images with AP >= 0.1."""
import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rq2b_validation_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'analysis/results/rq2/rq2b/validation'
# Load only analysis/plotting definitions: no inference imports or execution.
source_code = ROOT / 'scripts/rq2/rq2b_validate_brightness_method.py'
names = {'div', 'fmt', 'classify', 'classify_binary', 'predict_brightness',
         'confusion_metrics', 'plot_confusion_matrices', 'plot_confusion_matrices_binary',
         'plot', 'plot_binary'}
tree = ast.parse(source_code.read_text())
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
exec(compile(tree, str(source_code), 'exec'), globals())

def key(row):
    return row['image_name'], row['kind'], row['gamma']

def main():
    global BRIGHTNESS_METHODS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', choices=('own', 'shared'), default='own',
                        help='own (default): filter each model by its AP; shared: use YOLOv8 AP for both')
    parser.add_argument('--model', 
                        help='YOLO model to filter')
    parser.add_argument('--dataset', default='competition_dataset',
                        help='Validation results dataset directory (default: competition_dataset)')
    args = parser.parse_args()
    base = BASE / args.dataset
    mode = args.mode
    reference_path = base / 'yolov8/validation_results.json'
    reference_rows = {}
    if mode == 'shared':
        reference = json.loads(reference_path.read_text())
        reference_rows = {key(row): row for row in reference['validation_observations']}
    models = (args.model,) if args.model else ('yolov8', 'yolov12')
    for model in models:
        source = base / model / 'validation_results.json'
        source_bytes = source.read_bytes()
        original = json.loads(source_bytes)
        if original.get('schema_version', 0) >= 4:
            raise ValueError('This binary AP-filtering utility accepts schema <= 3 only. '
                             'Use the unfiltered three-way calibration report for schema 4.')
        data = copy.deepcopy(original)
        rows = original['validation_observations']
        retained = [row for row in data['validation_observations']
                    if (row['ap'] >= .1 if mode == 'own' else
                        key(row) in reference_rows and reference_rows[key(row)]['ap'] >= .1)]
        retained_keys = {key(row) for row in retained}
        transformed = [row for row in retained if row['kind'] != 'original']
        data['validation_observations'] = retained
        data['analysis_filter'] = {
            'minimum_ap_inclusive': .1,
            'selection_basis': 'each model AP' if mode == 'own' else 'YOLOv8 AP for matching image/kind/gamma',
            'source_results': str(source),
            'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'selection_source': str(source if mode == 'own' else reference_path),
            'observations_before': len(rows),
            'observations_retained': len(retained),
            'observations_excluded': len(rows) - len(retained),
            'inference_rerun': False,
            'original_screening_note': 'Historical screening records, not analysis observations.',
        }
        data['gamma_search']['selection'] = 'Only image/gamma observations satisfying analysis_filter are retained; no interpolation.'
        data['gamma_search']['retained_values'] = sorted({row['gamma'] for row in transformed})
        searches = []
        for search in data['gamma_searches']:
            search['search_evaluations'] = [row for row in search['search_evaluations']
                if (search['image_name'], 'transformed', row['gamma']) in retained_keys]
            search['retained_evaluations'] = len(search['search_evaluations'])
            if search['search_evaluations']:
                searches.append(search)
        data['gamma_searches'] = searches
        summary = data['holdout_summary']
        summary['retained_originals'] = len(retained) - len(transformed)
        summary['transformed_images'] = len(transformed)
        summary['transformed_failures'] = sum(row['actual_failure'] for row in transformed)
        summary['transformed_successes'] = len(transformed) - summary['transformed_failures']
        data['metrics'] = classify(retained, data['operating_ranges'])
        data['confusion_metrics'] = confusion_metrics(data)
        data.pop('sample_images', None)
        BRIGHTNESS_METHODS = list(data['operating_ranges'])
        out = base / model / 'filtered_ap_0.1'
        out.mkdir(exist_ok=True)
        (out / 'validation_results.json').write_text(json.dumps(data, indent=2) + '\n')
        plot(out / 'brightness_validation.pdf', retained, data['operating_ranges'], data['ap_threshold'])
        plot_confusion_matrices(out, data)
        # Check saved output and independently count all confusion-matrix cells.
        saved = json.loads((out / 'validation_results.json').read_text())
        assert len(saved['validation_observations']) == len(retained)
        assert all((r['ap'] if mode == 'own' else reference_rows[key(r)]['ap']) >= .1 for r in retained)
        assert sum(len(s['search_evaluations']) for s in searches) == len(transformed)
        for subset, metrics in saved['confusion_metrics'].items():
            selected = retained if subset == 'overall' else [r for r in transformed if r['side'] == {'brightened': 'bright', 'darkened': 'dark'}[subset]]
            for method, m in metrics.items():
                assert m['n'] == len(selected)
                bounds = saved['operating_ranges'][method]
                if not bounds['valid_interval']:
                    continue
                counts = dict(tp=0, fp=0, tn=0, fn=0)
                for row in selected:
                    outside = not bounds['lower'] <= row['brightness_scores'][method] <= bounds['upper']
                    fail = row['ap'] < saved['ap_threshold']
                    counts['tp' if outside and fail else 'fp' if outside else 'fn' if fail else 'tn'] += 1
                assert all(m[k] == v for k, v in counts.items())
        assert source.read_bytes() == source_bytes
        for path in out.glob('*.pdf'):
            assert path.read_bytes().startswith(b'%PDF-')
        print(json.dumps({'model': model, **data['analysis_filter'], 'output': str(out)}), flush=True)


if __name__ == "__main__":
    main()
