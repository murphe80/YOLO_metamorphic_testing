# End-to-end workflows

- `run_rq1.sh`: all RQ1 experiments followed by the complete RQ1 analysis.
- `run_rq1_base_experiment.sh`: base metamorphic transformations, inference,
  and (unless skipped) base analysis.
- `run_rq1_cone_colour_experiment.sh`: cone-colour transformations, inference,
  and (unless skipped) cone-colour analysis.
- `run_rq1_analysis.sh`: rerun all RQ1 analysis and plots from existing results.
- `run_rq2.sh`: RQ2a search/comparison, RQ2b brightness scoring, and RQ2c
  performance-distribution analysis.
- `run_rq3.sh`: full-vs-subsample bootstrap equivalence analysis, including
  the RQ3 JSON results, report, and plots.

All commands resolve the repository root from their own location, so they can
be launched from any working directory.
