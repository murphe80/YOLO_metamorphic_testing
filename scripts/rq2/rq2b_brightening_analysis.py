#!/usr/bin/env python3
"""Compare brightness methods on brightened validation images without retuning."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from scripts.rq2.rq2b_fusion_analysis import metrics, paired_bootstrap
from scripts.rq2.rq2b_validate_brightness_method import directional_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    d = json.loads(args.validation.read_text()); obs = d['validation_observations']
    ranges = {m: b for m, b in d['operating_ranges'].items() if b['valid_interval']}
    bright = [o for o in obs if o['side'] == 'bright']
    names = sorted({o['image_name'] for o in bright}, key=lambda n: (int(re.search(r'frame_\d+_(\d+)', n)[1]), n))
    ids = {n: i for i, n in enumerate(names)}
    def evaluate(rows):
        y = np.array([o['ap'] >= d['ap_threshold'] for o in rows])
        preds = {m: np.array([b['lower'] <= o['brightness_scores'][m] <= b['upper'] for o in rows]) for m,b in ranges.items()}
        return y, preds, {m: metrics(y, pred) for m, pred in preds.items()}
    y, predictions, scores = evaluate(bright)
    groups = np.array([ids[o['image_name']] for o in bright])
    ci = {metric: paired_bootstrap(y, predictions, groups, baseline='rms_contrast', metric=metric)
          for metric in ('accuracy', 'balanced_accuracy')}
    by_gamma = {}
    for gamma in sorted({o['gamma'] for o in bright}):
        rows = [o for o in bright if o['gamma'] == gamma]
        _, _, table = evaluate(rows)
        values = [o['brightness_scores']['rms_contrast'] for o in rows]
        by_gamma[str(gamma)] = {'successes': sum(o['ap'] >= d['ap_threshold'] for o in rows),
                              'failures': sum(o['ap'] < d['ap_threshold'] for o in rows),
                              'rms_min': min(values), 'rms_max': max(values), 'metrics': table}
    strata = {side: evaluate([o for o in obs if o['side'] == side])[2]['rms_contrast']
              for side in ('original', 'bright', 'dark')}
    rb = ranges['rms_contrast']
    locations = {label: {'n':len(rows), 'successes':sum(o['ap']>=d['ap_threshold'] for o in rows)}
                 for label,rows in [(label,[o for o in bright if predicate(o['brightness_scores']['rms_contrast'])])
                    for label,predicate in [('below',lambda x:x<rb['lower']),('inside',lambda x:rb['lower']<=x<=rb['upper']),('above',lambda x:x>rb['upper'])]]}
    result = dict(model=d['model'], source=str(args.validation), source_images=len(names),
                  n=len(bright), successes=int(y.sum()), failures=int((~y).sum()),
                  operating_ranges=ranges, bright_metrics=scores, rms_by_side=strata,
                  per_gamma=by_gamma, rms_score_locations=locations,
                  paired_delta_vs_RMS_95pct_block_intervals=ci)
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    def pct(v): return 'N/A' if v is None else f'{100*v:.2f}%'
    lines=[f"# {d['model']}: performance on gamma-brightened images",'',
           f"{len(bright)} transformations from {len(names)} originals, with {int(y.sum())} successes (AP ≥ {d['ap_threshold']}) and {int((~y).sum())} failures. Fixed calibration mean boundaries; no fitting or threshold selection.",'',
           '| Method | Accuracy | Balanced accuracy | Successful images accepted | Failures rejected | MCC |',
           '|---|---:|---:|---:|---:|---:|']
    for m,s in sorted(scores.items(),key=lambda kv:kv[1]['balanced_accuracy'],reverse=True):
        lines.append(f"| {m} | {pct(s['accuracy'])} | {pct(s['balanced_accuracy'])} | {pct(s['success_acceptance'])} | {pct(s['failure_recall'])} | {s['mcc']:.3f} |")
    lines+=['','## RMS by transformation direction','','| Direction | N | Accuracy | Balanced accuracy | Success acceptance | Failure rejection |','|---|---:|---:|---:|---:|---:|']
    for side,s in strata.items(): lines.append(f"| {side} | {s['n']} | {pct(s['accuracy'])} | {pct(s['balanced_accuracy'])} | {pct(s['success_acceptance'])} | {pct(s['failure_recall'])} |")
    lines+=['','## Brightening by gamma','','| Gamma | Successes | Failures | RMS range | Accepted successes | Rejected successes | Rejected failures | Accepted failures |','|---|---:|---:|---|---:|---:|---:|---:|']
    for gamma,row in by_gamma.items():
        s=row['metrics']['rms_contrast']
        lines.append(f"| {float(gamma):.2f} | {row['successes']} | {row['failures']} | {row['rms_min']:.4f}–{row['rms_max']:.4f} | {s['accepted_successes']} | {s['rejected_successes']} | {s['rejected_failures']} | {s['accepted_failures']} |")
    lines+=['','## Interpretation and uncertainty','',
            f"RMS fixed operating interval: [{rb['lower']:.6f}, {rb['upper']:.6f}]. Bright-image score locations: {locations}.",'',
            'Positive means AP ≥ 0.5. Success acceptance is not object-detection recall. Balanced accuracy is the mean of success acceptance and failure rejection. Source images were screened for original AP ≥ 0.5, so results apply to that subset and this gamma grid. Bright and dark distributions differ, making direction comparisons descriptive rather than controlled causal comparisons.','',
            'Intervals below use 1,000 paired resamples of contiguous blocks of ten timestamp-ordered originals (all gamma variants stay together). Only eight blocks are available; intervals are exploratory, not adjusted for multiple comparisons, and do not prove cross-dataset generalisation.','',
            '| Comparator | RMS accuracy improvement, 95% interval | RMS balanced-accuracy improvement, 95% interval |','|---|---|---|']
    for m in ranges:
        if m=='rms_contrast': continue
        a,b=ci['accuracy'][m],ci['balanced_accuracy'][m]
        lines.append(f'| {m} | [{-100*a[1]:+.2f}, {-100*a[0]:+.2f}] pp | [{-100*b[1]:+.2f}, {-100*b[0]:+.2f}] pp |')
    lines += ['', directional_report(d)]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__': main()
