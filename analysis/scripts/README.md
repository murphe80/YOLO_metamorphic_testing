# Analysis runners

- `run_base_metamorphic_analysis.sh` evaluates base metamorphic results.
- `run_cone_colour_analysis.sh` evaluates cone-colour results and generates
  `confidence_change/cone_colour.pdf`, combining the selected colour rules on
  one x axis. Experiment runners calling this script also generate the plot.
- `generate_detection_overlap_metrics.sh` is a convenience runner for retained,
  lost, and new detection metrics on the test dataset.

The complete multi-stage analysis workflow is
`../../end_to_end_scripts/run_rq1_analysis.sh`.

To regenerate only the combined confidence plot from existing GT-aligned
detections, run from the repository root:

```bash
python3 analysis/visualisation/plot_confidence_change.py --dataset test --rules cone_colour
```

Individual colour rule names can also be passed to `--rules` to select a subset
for the same combined plot. Running without `--rules` includes available colour
rules alongside the configured base transformations.
