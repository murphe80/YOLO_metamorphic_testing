# Research dashboard

From the project root, run:

```bash
python3 dashboard/serve.py
```

Open **http://127.0.0.1:8080**. Requires Python 3.9+; no packages, frontend build, internet connection, or model weights are needed. Use `--port 8081` if the default port is occupied. The server binds to localhost by default.

**Paper results** loads `analysis/results`. Refresh after generating or updating analysis outputs. Analysis files are read only; the dashboard neither runs experiments nor rewrites results. Source paths embedded in archived JSON are not followed.

- **Model robustness:** an expandable original-image baseline and an “Explore a transformation” section with parameter response, detection overlap, confidence and cone-colour figures. Dataset, comparison and AP/AR selectors focus the view. Use the transformation selector to choose what to inspect.
- **Brightness tuning:** search efficiency and threshold agreement, brightness-score validation, and original/transformed accuracy distributions. Model and available search-method selectors reveal the existing figures.
- **Fewer test runs:** actual sample fraction and image counts, equivalence outcomes, and full-versus-sample response curves with uncertainty. AP/AR and consistency/degradation selectors keep the two analyses distinct.

Figures are reused directly from analysis outputs and link to their original resolution. The dashboard creates a response chart from metrics when an RQ1 plot is absent. Missing stages and empty results are shown explicitly. Run the corresponding `end_to_end_scripts/run_rq1_analysis.sh`, `run_rq2.sh`, or `run_rq3.sh` workflow to generate outputs (see the project README for prerequisites and options).

## Interpretation

Missing, non-finite and COCO `-1` metrics are unavailable, not zero. AP uses COCO AP@[.50:.95]; AR uses `AR100` (or exported `AR`). Detection consistency is not ground-truth accuracy. Baseline scores refer to the selected dataset, not an external COCO benchmark.

Raw `stats.json` values take precedence over metric exports; exported settings provide the historical parameter labels when available. For raw-only results, labels use the current transformation config. RQ2 summary cards use saved matched-image comparison statistics; mean per-image speedup differs from the ratio of mean runtimes. Holdout validation uses the `filtered_ap_0.1` subfolder for both metrics and figures and is shown only where it exists, independently of the selected scoring search method. RQ3 classifications and metadata come directly from the saved bootstrap results, including the actual sample fraction and validation discrepancies. Inconclusive outcomes are not evidence of a difference. The dashboard does not infer failure-image overlap, train/test generalisation, or subsumption from aggregate statistics.

The loader scans known analysis-output paths, avoiding RQ2's large per-image inference directories. No generated manifest is required. For RQ2/RQ3, outputs should retain the pipeline's standard directory layout. Distinct reruns should be archived outside the active `analysis/results` tree before replacement if they need preserving; the dashboard presents that tree as the current experiment set.

## Verification

```bash
python3 -m unittest discover -s dashboard -p 'test_*.py'
```
