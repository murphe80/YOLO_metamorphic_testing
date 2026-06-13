# Venn Diagram Visualization for Detection Overlap

This module generates three-circle Venn diagrams showing the overlap between:
- Original image detections
- Transformed image detections  
- Ground truth annotations

## Purpose

Addresses **RQ1.a**: "Do we find different failures than original images, or that of classic transformations like brightness or blur?"

The Venn diagrams visualize 7 distinct regions:
1. **Original ∩ Transformed ∩ GT**: True positives maintained after transformation
2. **Original ∩ Transformed**: Consistent false positives
3. **Original ∩ GT**: True positives lost during transformation (critical failures)
4. **Transformed ∩ GT**: True positives gained in transformation (recovery)
5. **Original only**: False positives in original that disappeared
6. **Transformed only**: False positives introduced by transformation (spurious)
7. **GT only**: Missed detections in both original and transformed

## Setup

```bash
# Create virtual environment and install dependencies
bash scripts/setup_venn_env.sh

# Activate environment
source venv_venn/bin/activate
```

## Usage

### 1. Generate Individual Venn Diagrams

Generate Venn diagrams for all model-transformation combinations:

```bash
python3 analysis/visualisation/plot_detection_venn.py
```

Output: `analysis/results/venn_diagrams/[model]_[transformation]_venn.png`

Options:
- `--iou_threshold 0.7`: Change IoU threshold (default: 0.5)
- `--model yolov8`: Generate for specific model only
- `--transformation imgGrayscale`: Generate for specific transformation only
- `--results_dir results`: Custom results directory
- `--gt_path data/.../annotations_coco.json`: Custom ground truth path

Example:
```bash
# Only YOLOv8 with stricter IoU threshold
python3 analysis/visualisation/plot_detection_venn.py --model yolov8 --iou_threshold 0.7

# Single transformation across all models
python3 analysis/visualisation/plot_detection_venn.py --transformation imgGrayscale
```

### 2. Generate Grid Visualization

Create a comprehensive grid showing all combinations:

```bash
python3 analysis/visualisation/plot_venn_grid.py
```

Output: `analysis/results/venn_diagrams/all_models_grid.png`

Options:
- `--layout row`: Models as rows, transformations as columns (default)
- `--layout col`: Models as columns, transformations as rows

### 3. Export Statistics to CSV

Export quantitative analysis:

```bash
python3 analysis/utils/venn_statistics_analysis.py --summary
```

Output: `analysis/results/venn_diagrams/venn_statistics.csv`

The CSV includes:
- Raw region counts (7 Venn regions)
- TP retention rate: How many true positives are maintained
- Detection loss rate: Proportion of original detections that disappeared
- Spurious detection rate: Proportion of new spurious detections
- Precision/recall for original and transformed images
- Metamorphic consistency metrics

## Data Requirements

The scripts expect the following directory structure:

```
results/
├── yolov5/
│   ├── original/
│   │   └── detections_original_coco.json
│   ├── imgGrayscale/
│   │   └── detections_coco.json
│   └── [other transformations]/
├── yolov8/
└── yolov12/

data/base_images/fsoco_sample/
└── annotations_coco.json  # Ground truth
```

Detection files must be in COCO JSON format:
```json
[
  {
    "id": 1,
    "image_id": 1,
    "category_id": 1,
    "bbox": [x, y, width, height],
    "score": 0.903625
  },
  ...
]
```

## Interpretation Guide

### High TP Retention (>90%)
- Transformation is **robust** for this model
- Most true positive detections are maintained
- Model is metamorphically consistent

### High Detection Loss (>20%)
- Transformation causes **significant degradation**
- Many original detections disappear
- Model is sensitive to this transformation

### High Spurious Rate (>10%)
- Transformation introduces **false positives**
- New detections appear that weren't in original
- May indicate overfitting or instability

### Large "Original ∩ GT" Region
- **Critical failures**: True positives lost during transformation
- These are real objects the model detected originally but missed after transformation
- High-priority robustness issue

### Large "Transformed ∩ GT" Region
- **Recovery**: True positives gained in transformation
- Transformation helps model detect objects it missed originally
- Indicates complementary strengths

## Metrics Explained

### TP Retention Rate
```
(orig_trans_gt + orig_gt) / (orig_trans_gt + orig_gt + trans_gt)
```
Measures how many of the true positives from the original image are still detected after transformation.

### Detection Consistency
```
2 * (orig_trans_gt + orig_trans) / (total_orig + total_trans)
```
Measures overall consistency between original and transformed detections (includes both TPs and FPs).

### Metamorphic TP Consistency
```
orig_trans_gt / (orig_trans_gt + orig_gt)
```
Of the true positives detected in original, what proportion are still detected after transformation?

## Troubleshooting

**"No Data" in grid cells:**
- Detection files don't exist for that model-transformation pair
- Check that you've run inference on the transformed images

**Low counts across all regions:**
- Check IoU threshold - try lowering to 0.3 for more matches
- Verify detection files have correct COCO format
- Ensure image_id fields match across files

**Module import errors:**
- Activate the virtual environment: `source venn_venn/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

## Next Steps

After generating visualizations:

1. **Compare across models**: Which model has highest TP retention for each transformation?
2. **Identify weak transformations**: Which transformations cause most detection loss?
3. **Answer RQ1.b**: Establish patterns on when to use one model over another
4. **Integrate into dashboard**: Add Venn diagrams to `analysis/dashboard.html`
