# Analysis result layout

Generated analysis outputs are grouped by research question:

```text
analysis/results/
├── rq1/
│   └── <dataset>/
│       ├── <model>/<rule>/<comparison>/stats.json
│       ├── confidence_change/
│       ├── detection_overlap_metrics/
│       └── parameter_variations/
├── rq2/
│   ├── rq2a/{brute_force,golden_section,comparison}/
│   ├── rq2b/{brightness_scoring,validation}/
│   └── rq2c/performance_distributions/
└── rq3/
    └── sample_equivalence/{results.json,report.md,plots/}
```

