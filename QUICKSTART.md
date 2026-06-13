# Venn Diagram Visualization - Quick Start Guide

## What This Does

Creates three-circle Venn diagrams showing detection overlap between:
- **Original** image detections
- **Transformed** image detections
- **Ground truth** annotations

This helps answer **RQ1.a**: Do metamorphic tests reveal different failures than traditional transformations?

## Installation & Setup

```bash
# 1. Setup environment (creates venv_venn with all dependencies)
bash scripts/setup_venn_env.sh

# 2. Activate environment
source venn_venn/bin/activate

# 3. Run tests to verify everything works
cd analysis && python test_venn_logic.py
```

## Generate Visualizations

### Option 1: Everything at Once (Recommended)

```bash
bash scripts/generate_all_venn_diagrams.sh
```

This generates:
- 27 individual Venn diagrams (one per model-transformation pair)
- 1 grid visualization showing all combinations
- Statistics JSON and CSV files

**Output:**
```
analysis/results/venn_diagrams/
├── yolov5_imgGrayscale_venn.png
├── yolov5_imgColorWheel_venn.png
├── ... (27 total)
├── all_models_grid.png
├── venn_statistics.json
└── venn_statistics.csv
```

### Option 2: Individual Steps

```bash
# Individual diagrams
python3 analysis/visualisation/plot_detection_venn.py

# Grid view
python3 analysis/visualisation/plot_venn_grid.py

# Export CSV
python3 analysis/utils/venn_statistics_analysis.py --summary
```

### Option 3: Specific Model or Transformation

```bash
# Only YOLOv8
python3 analysis/visualisation/plot_detection_venn.py --model yolov8

# Only imgGrayscale across all models
python3 analysis/visualisation/plot_detection_venn.py --transformation imgGrayscale

# Stricter IoU threshold
python3 analysis/visualisation/plot_detection_venn.py --iou_threshold 0.7
```

## Understanding the Venn Diagrams

Each diagram has 7 regions:

| Region | Meaning |
|--------|---------|
| **Original ∩ Transformed ∩ GT** | True positives maintained (good!) |
| **Original ∩ GT** | True positives LOST (critical failure!) |
| **Transformed ∩ GT** | True positives gained (recovery) |
| **Original ∩ Transformed** | Consistent false positives |
| **Original only** | FPs that disappeared |
| **Transformed only** | Spurious FPs introduced |
| **GT only** | Missed detections (baseline issue) |

### Key Metrics to Look For

**TP Retention Rate** (in diagram title):
- >90%: Robust transformation
- 70-90%: Moderate robustness
- <70%: Significant degradation

**Detection Loss Rate**:
- Proportion of original detections that disappeared
- High loss = transformation is disruptive

**Large "Original ∩ GT" region**:
- Critical issue: Real objects lost during transformation
- Prioritize fixing these

## Analyzing Results

### CSV Analysis

Import `venn_statistics.csv` into your favorite tool:

```python
import pandas as pd
df = pd.read_csv('analysis/results/venn_diagrams/venn_statistics.csv')

# Best model per transformation
best = df.loc[df.groupby('transformation')['tp_retention_rate'].idxmax()]
print(best[['transformation', 'model', 'tp_retention_rate']])

# Worst transformations per model
worst = df.groupby('model')['tp_retention_rate'].min()
print(worst)
```

### Grid View Analysis

Use `all_models_grid.png` to visually compare:
- Rows = Models
- Columns = Transformations
- Color = Model (Blue=v5, Purple=v8, Green=v12)

Look for patterns:
- Which transformations are hardest across all models?
- Which model has largest "Original ∩ GT" regions?
- Are there model-specific vulnerabilities?

## Common Questions

**Q: Why three circles instead of two?**
A: Adding ground truth distinguishes true positives from false positives, revealing which detections are real objects vs. spurious.

**Q: What if I don't have results/ directory?**
A: The worktree doesn't include the results directory (it's in .gitignore). Merge this branch back to your main working copy where you have the detection files.

**Q: Can I change the IoU threshold?**
A: Yes! Use `--iou_threshold 0.3` for looser matching or `--iou_threshold 0.7` for stricter matching.

**Q: How do I merge this to my working branch?**
A: From your main working directory:
```bash
git checkout sample  # or your working branch
git merge worktree-venn-diagram-viz
```

## Next Steps

1. ✓ Run visualization pipeline
2. Analyze CSV to identify robustness patterns
3. Answer RQ1.a: Compare metamorphic vs traditional transformation failures
4. Answer RQ1.b: Establish model selection guidelines based on transformation patterns
5. Integrate Venn diagrams into `analysis/dashboard.html`

## Troubleshooting

**"Module not found":**
```bash
source venv_venn/bin/activate
```

**"File not found" errors:**
- Make sure you're in the correct working directory with results/ and data/ folders
- Check paths with: `ls results/yolov8/imgGrayscale/detections_coco.json`

**Empty Venn diagrams:**
- Lower IoU threshold: `--iou_threshold 0.3`
- Verify COCO JSON format

## Documentation

- Full details: `VENN_IMPLEMENTATION.md`
- User guide: `analysis/visualisation/README_VENN.md`
- Tests: `analysis/test_venn_logic.py`
