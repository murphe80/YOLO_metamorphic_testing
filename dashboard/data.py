"""Read analysis products without importing the inference stack or changing results."""
import json
import math
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOTS = {'paper': PROJECT / 'analysis/original_results', 'local': PROJECT / 'analysis/results'}


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def read(path, warnings):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('expected a JSON object')
        return clean(value)
    except (OSError, ValueError) as exc:
        warnings.append(f'{path.name}: {exc}')
        return {}


def metric_values(stats):
    return {k: v if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1 else None
            for k, v in {'AP': stats.get('AP'), 'AR': stats.get('AR', stats.get('AR100'))}.items()}


def build(source, roots=None):
    root = (roots or ROOTS)[source]
    warnings = []
    def load(p):
        return read(p, warnings)
    def relative(p):
        return p.relative_to(root).as_posix()
    # Only inspect known analysis output locations: RQ2 also contains millions of inference files.
    figures = set()
    for pattern in (
        'rq1/*/parameter_variations/*/*.png', 'rq1/*/detection_overlap_metrics/plots/*.png',
        'rq1/*/confidence_change/*.png', 'rq1/*/cone_colour_analysis/*.png',
        'rq2/rq2a/comparison/*/*.png', 'rq2/rq2b/brightness_scoring/*/*/*.png',
        'rq2/rq2b/validation/*/*.png', 'rq2/rq2c/performance_distributions/*.png',
        'rq3/sample_equivalence/plots/*.png',
    ):
        figures.update(relative(p) for p in root.glob(pattern) if p.is_file())
    config_path = PROJECT / 'config/brute_force_transformations.json'
    config = load(config_path) if config_path.exists() else {}
    variants = {v['variant_id']: (rule, {k: x for k, x in v.items() if k != 'variant_id'})
                for rule, values in config.items() if isinstance(values, list)
                for v in values if isinstance(v, dict) and 'variant_id' in v}
    datasets = {}
    for dataset_dir in sorted((root / 'rq1').glob('*')):
        if not dataset_dir.is_dir():
            continue
        rows = {}
        baselines = {}
        for p in sorted(dataset_dir.glob('*/*/*/stats.json')):
            model, variant, comparison = p.relative_to(dataset_dir).parts[:3]
            if comparison not in ('original_vs_transformed', 'transformed_vs_gt'):
                continue
            rule, setting = variants.get(variant, (re.sub(r'_\d+$', '', variant), {}))
            rows[(model, variant, comparison)] = dict(model=model, variant=variant, rule=rule,
                setting=setting, comparison=comparison, **metric_values(load(p)))
        for p in sorted(dataset_dir.glob('*/original_vs_gt/stats.json')):
            baselines[p.parent.parent.name] = metric_values(load(p))
        # Exports preserve the original parameter labels, even if the current config changes.
        for p in sorted(dataset_dir.glob('parameter_variations/metrics_*.json')):
            d = load(p)
            comparison = d.get('comparison', p.stem.removeprefix('metrics_'))
            for rule, group in d.get('rules', {}).items():
                for setting in group.get('settings', []):
                    for model, values in setting.get('models', {}).items():
                        key = (model, setting['variant_id'], comparison)
                        row = rows.get(key, dict(model=model, variant=setting['variant_id'], rule=rule,
                            comparison=comparison, **metric_values(values)))
                        row['setting'] = setting.get('setting', {})
                        rows[key] = row
        if rows or baselines or any(f.startswith(f'rq1/{dataset_dir.name}/') for f in figures):
            datasets[dataset_dir.name] = dict(rows=list(rows.values()), baselines=baselines)
    comparisons = {}
    for p in sorted(root.glob('rq2/rq2a/comparison/*/comparison.json')):
        comparisons[p.parent.name] = load(p).get('summary', {})
    validation = {}
    for p in sorted(root.glob('rq2/rq2b/validation/*/validation_results.json')):
        d = load(p)
        validation[p.parent.name] = {k: d.get(k, {}) for k in ('metrics', 'holdout_summary', 'ap_threshold')}
    search = {}
    for p in sorted(root.glob('rq2/rq2a/*/*/results.json')):
        d = load(p)
        search.setdefault(p.parent.name, {})[p.parent.parent.name] = {
            k: d.get(k) for k in ('dataset', 'target_ap', 'processed_images', 'skipped_images', 'failed_images')}
    rq3_path = root / 'rq3/sample_equivalence/results.json'
    rq3 = load(rq3_path) if rq3_path.exists() else {}
    return clean(dict(source=source, root=f'analysis/{root.name}', datasets=datasets,
        figures=sorted(figures), comparisons=comparisons, validation=validation, search=search,
        rq3={k: rq3.get(k, {}) for k in ('metadata', 'validation', 'parameter_results', 'auc_results')} if rq3 else {},
        warnings=warnings))
