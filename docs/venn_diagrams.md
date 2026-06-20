# Venn Diagram Visualization

This generates three-circle Venn diagrams showing the overlap between:

- Original image detections
- Transformed image detections  
- Ground truth annotations

## Purpose

The Venn diagrams visualize 7 distinct regions:

1. **Original ∩ Transformed ∩ GT**: True positives maintained after transformation
2. **Original ∩ Transformed**: Consistent false positives
3. **Original ∩ GT**: True positives lost during transformation (critical failures)
4. **Transformed ∩ GT**: True positives gained in transformation (recovery)
5. **Original only**: False positives in original that disappeared
6. **Transformed only**: False positives introduced by transformation (spurious)
7. **GT only**: Missed detections in both original and transformed

## How Detection Matching Works

### IoU-Based Three-Way Matching

Cone detections are matched across Original, Transformed, and Ground Truth using **Intersection over Union (IoU)** as the similarity metric.

**Matching Rule**: Two detections match if their IoU ≥ threshold of 0.5

- IoU = 1.0: Perfect match (identical boxes)
- IoU = 0.5: Moderate overlap (standard COCO threshold)
- IoU = 0.0: No overlap (different objects)

#### Three-Way Matching Algorithm

The algorithm assigns each detection to exactly one of the 7 Venn regions:

**Step 1: Find Three-Way Matches (Original ∩ Transformed ∩ GT)**

- For each original detection, find best matching transformed detection (highest IoU ≥ threshold)
- Check if this pair also matches a ground truth detection
- If all three match: assign to `orig_trans_gt` region
- Mark all three as "matched" to prevent double-counting

**Step 2: Find Two-Way Matches**

- Among remaining unmatched detections, find pairs:
  - Original ∩ Transformed → `orig_trans` region
  - Original ∩ GT → `orig_gt` region
  - Transformed ∩ GT → `trans_gt` region

**Step 3: Assign Single-Source Detections**

- Any still unmatched detections go to single-source regions:
  - Original only → `orig_only` region
  - Transformed only → `trans_only` region
  - GT only → `gt_only` region

**Key Properties:**

- **Greedy assignment**: Prioritises strongest matches first (3-way → 2-way → 1-way)
- **One-to-one constraint**: Each detection appears in exactly one region
- **Per-image matching**: Matches computed separately per image, then aggregated

### Comparison to Pycocotools Evaluation

This implementation differs from the standard `pycocotools` COCO evaluation in several ways:


| Aspect                 | Venn Diagram Matching                       | Pycocotools COCOeval                  |
| ---------------------- | ------------------------------------------- | ------------------------------------- |
| **Purpose**            | Metamorphic consistency analysis            | Model accuracy evaluation             |
| **Comparison**         | Original ↔ Transformed ↔ GT (3-way)         | Predictions ↔ GT only (2-way)         |
| **Matching Algorithm** |  COCO-style: confidence-sorted greedy       |  Confidence-sorted greedy            |
| **Sorting**            |  By confidence (highest first)              |  By confidence (highest first)       |
| **Assignment**         |  Best IoU match per detection               |  Best IoU match per detection        |
| **Regions**            | 7 Venn regions                              | 3 categories (TP/FP/FN)               |
| **IoU Thresholds**     | Single threshold (default: 0.5)             | Multiple thresholds [0.5:0.95]        |
| **Metrics**            | TP retention, detection loss, spurious rate | AP, AP50, AP75, AR                    |
| **Confidence Use**     | Match priority only                         | Match priority + P-R curves           |
| **Category**           | Ignored (all cones treated equally)         | Per-category evaluation               |
| **Output**             | Venn region counts, CSV statistics          | AP/AR scores, precision-recall curves |



#### Implementation

The matching logic is in `analysis/utils/detection_matching.py`:

- `compute_iou()`: COCO-format IoU (identical to pycocotools formula)
- `match_three_way_per_image()`: Greedy three-way assignment
- `aggregate_three_way_overlap()`: Aggregates across all images

**Algorithm Complexity**: O(n × m × k) per image, where n, m, k are detection counts in each source. Typically ~10-50 detections per image, so matching is fast (<1ms per image).


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
4. **Integrate into dashboard**: Add Venn diagrams to `dashboard/dashboard.html`

