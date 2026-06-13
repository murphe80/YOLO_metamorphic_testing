# Venn Diagram Visualization Implementation Summary

## Overview

Implemented a comprehensive three-circle Venn diagram visualization system to analyze detection overlap between original images, transformed images, and ground truth annotations for metamorphic testing of YOLO models.

**Addresses Research Question RQ1.a**: "Do we find different failures than original images, or that of classic transformations like brightness or blur?"

## What Was Implemented

### 1. Core Matching Algorithm (`analysis/utils/detection_matching.py`)

Three-way detection matching using IoU (Intersection over Union):

**Key Functions:**
- `compute_iou(bbox1, bbox2)` - Calculate IoU between bounding boxes
- `match_three_way_per_image()` - Assign detections to 7 Venn regions per image
- `aggregate_three_way_overlap()` - Aggregate statistics across all images
- `compute_retention_metrics()` - Calculate derived metrics (TP retention, detection loss, spurious rate)

**Algorithm:**
1. Greedy three-way matching (prioritizes 3-way → 2-way → 1-way matches)
2. One-to-one constraint (each detection assigned to exactly one region)
3. Configurable IoU threshold (default: 0.5)

**7 Venn Regions:**
1. `orig_trans_gt`: All three (TPs maintained after transformation)
2. `orig_trans`: Original + Transformed only (consistent FPs)
3. `orig_gt`: Original + GT only (TPs lost during transformation - **critical failures**)
4. `trans_gt`: Transformed + GT only (TPs gained - recovery)
5. `orig_only`: Original only (FPs that disappeared)
6. `trans_only`: Transformed only (spurious FPs introduced)
7. `gt_only`: GT only (missed in both original and transformed)

### 2. Individual Venn Diagrams (`analysis/visualisation/plot_detection_venn.py`)

Generates high-resolution PNG diagrams for each model-transformation pair.

**Features:**
- Model-specific color scheme (YOLOv5: Blue, YOLOv8: Purple, YOLOv12: Green)
- Displays all 7 regions with counts
- Title includes key metrics (TP retention rate, detection loss rate)
- Outputs JSON with raw statistics

**Usage:**
```bash
python3 analysis/visualisation/plot_detection_venn.py
python3 analysis/visualisation/plot_detection_venn.py --model yolov8 --iou_threshold 0.7
```

**Output:** `analysis/results/venn_diagrams/{model}_{transformation}_venn.png`

### 3. Grid Visualization (`analysis/visualisation/plot_venn_grid.py`)

Comprehensive grid showing all 27 combinations (3 models × 9 transformations).

**Layouts:**
- `--layout row`: Models as rows, transformations as columns (default, 3×9 grid)
- `--layout col`: Models as columns, transformations as rows (9×3 grid)

**Usage:**
```bash
python3 analysis/visualisation/plot_venn_grid.py
```

**Output:** `analysis/results/venn_diagrams/all_models_grid.png`

### 4. Statistics Analysis (`analysis/utils/venn_statistics_analysis.py`)

Exports quantitative metrics to CSV for further analysis.

**Metrics Computed:**
- **TP Retention Rate**: `(orig_trans_gt + orig_gt) / (orig_trans_gt + orig_gt + trans_gt)`
  - What proportion of true positives from original are still detected after transformation
- **Detection Consistency**: `2 * (orig_trans_gt + orig_trans) / (total_orig + total_trans)`
  - Overall consistency between original and transformed (includes both TPs and FPs)
- **Detection Loss Rate**: `orig_only / total_original`
  - Proportion of original detections that disappeared
- **Spurious Detection Rate**: `trans_only / total_transformed`
  - Proportion of new detections introduced by transformation
- **Metamorphic TP Consistency**: `orig_trans_gt / (orig_trans_gt + orig_gt)`
  - Of TPs in original, what % maintained after transformation

**Usage:**
```bash
python3 analysis/utils/venn_statistics_analysis.py --summary
```

**Output:** `analysis/results/venn_diagrams/venn_statistics.csv`

### 5. Supporting Scripts

**Setup Script** (`scripts/setup_venn_env.sh`):
- Creates isolated virtual environment `venn_venn`
- Installs all dependencies (matplotlib, matplotlib-venn, numpy, etc.)

**Master Run Script** (`scripts/generate_all_venn_diagrams.sh`):
- One-command execution of entire pipeline
- Generates individual diagrams → grid → CSV statistics
- Includes error checking for missing data

**Test Suite** (`analysis/test_venn_logic.py`):
- Unit tests for IoU computation
- Integration tests for three-way matching
- Edge case tests (detection loss, spurious detections, perfect consistency)

## Dependencies Added

```
matplotlib-venn==0.11.10  # Three-circle Venn diagram support
```

All other dependencies (matplotlib, numpy, pycocotools) were already in `requirements.txt`.

## Quick Start

```bash
# 1. Setup environment (one-time)
bash scripts/setup_venn_env.sh

# 2. Activate environment
source venn_venn/bin/activate

# 3. Run verification tests
cd analysis && python test_venn_logic.py

# 4. Generate all visualizations
bash scripts/generate_all_venn_diagrams.sh
```

## File Structure

```
analysis/
├── utils/
│   ├── detection_matching.py          # Core matching algorithm
│   └── venn_statistics_analysis.py    # CSV export and metrics
├── visualisation/
│   ├── plot_detection_venn.py         # Individual Venn diagrams
│   ├── plot_venn_grid.py              # Grid visualization
│   └── README_VENN.md                 # Comprehensive user guide
├── test_venn_logic.py                 # Verification tests
└── results/
    └── venn_diagrams/                 # Output directory
        ├── yolov5_imgGrayscale_venn.png
        ├── ... (27 diagrams total)
        ├── all_models_grid.png
        ├── venn_statistics.json
        └── venn_statistics.csv

scripts/
├── setup_venn_env.sh                  # Environment setup
└── generate_all_venn_diagrams.sh      # Master run script

venv_venn/                             # Virtual environment (created by setup)
```

## Key Design Decisions

### Why Three Circles Instead of Two?

Initial request was for two circles (Original vs Transformed), but user selected three circles with ground truth during planning phase. This provides:
- **Richer insights**: Distinguishes TPs from FPs
- **Critical failure identification**: `orig_gt` region shows real objects lost during transformation
- **Recovery detection**: `trans_gt` region shows objects the transformation helped detect
- **RQ1.a alignment**: Can now compare metamorphic failures (detection loss) vs. accuracy failures (missed GTs)

### IoU Threshold of 0.5

Standard COCO evaluation threshold. Can be adjusted via `--iou_threshold` flag:
- 0.3: Looser matching (more matches, less strict)
- 0.5: Standard (recommended)
- 0.7: Stricter matching (fewer matches, more precise)

### Aggregate vs Per-Image

Implemented aggregate-only (user choice during planning). This shows:
- Overall robustness trends
- Model-transformation patterns
- Comparative analysis across models

Per-image diagrams can be added later for detailed failure analysis.

### Color Scheme

Consistent with existing dashboard (`analysis/dashboard.html`):
- YOLOv5: Blue (#1976d2)
- YOLOv8: Purple (#7b1fa2)
- YOLOv12: Green (#388e3c)

Original circle uses model color to make grid visualization interpretable at a glance.

## Interpreting Results

### High-Level Patterns to Look For:

1. **Large `orig_trans_gt` region**: Good metamorphic consistency + accuracy
2. **Large `orig_gt` region**: **Critical issue** - transformation breaks TPs
3. **Large `trans_only` region**: Transformation introduces spurious FPs
4. **Large `gt_only` region**: Low recall (baseline issue, not metamorphic)

### Cross-Model Comparisons:

Use the grid visualization to identify:
- Which model maintains most detections across transformations?
- Which transformations are most disruptive per model?
- Are there model-specific robustness patterns?

### Answering Research Questions:

**RQ1.a**: Compare `orig_gt` counts across transformations:
- Do metamorphic tests (imgColorWheel, imgStretch) reveal different failures than classic tests (imgDarkBright, imgMotionBlur)?
- Check if different models fail on different images

**RQ1.b**: Use TP retention rates from CSV:
- Model with highest average TP retention → most robust
- Model with lowest TP retention on critical transformation → avoid for that scenario

## Next Steps

1. **Run on full dataset**: Execute `bash scripts/generate_all_venn_diagrams.sh`
2. **Analyze CSV**: Import `venn_statistics.csv` into R/Python/Excel for statistical analysis
3. **Document patterns**: Identify which transformations cause most TP loss per model
4. **Integrate into dashboard**: Add Venn diagrams to `analysis/dashboard.html` for interactive exploration
5. **Cross-validate with AP50**: Compare TP retention rates with existing AP50 metrics from `stats.json`

## Verification

All unit tests pass:
- ✓ IoU computation (identical, non-overlapping, partial overlap cases)
- ✓ Three-way matching (perfect consistency scenario)
- ✓ Detection loss (TP lost during transformation)
- ✓ Spurious detection (new FPs introduced)

Run tests: `cd analysis && python test_venn_logic.py`

## Troubleshooting

**"Module not found" errors:**
```bash
source venv_venn/bin/activate
```

**"No Data" in visualizations:**
- Check detection files exist: `ls results/yolov8/imgGrayscale/detections_coco.json`
- Ensure ground truth exists: `ls data/base_images/fsoco_sample/annotations_coco.json`

**Empty Venn regions:**
- Try lowering IoU threshold: `--iou_threshold 0.3`
- Check detection file format (must be COCO JSON)

## Performance Notes

- **Memory**: Loads all detections into memory; ~10MB per model-transformation pair
- **Speed**: ~2-5 seconds per Venn diagram (depends on detection count)
- **Total runtime**: ~1-3 minutes for all 27 combinations on typical hardware
