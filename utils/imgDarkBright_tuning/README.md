# Brightness Tuning Utilities (RQ2)

Search-based algorithm to find brightness thresholds where YOLO model accuracy degrades to a target level.

## Purpose

Addresses **RQ2**: *Can a search-based algorithm efficiently find the brightness adjustment parameter for which YOLO detection accuracy degrades beyond an acceptable level, thus producing an operational range for which YOLO can be used?*

## How It Works

For each image in the dataset:

1. **Calculate brightness metrics:**
   - RMS contrast
   - BT.709 luminosity
   - Mean shift segmentation variance
   - Low/high pixel saturation ratio

2. **Search algorithm:**
   - Iteratively apply brightness transformations
   - Run YOLO inference on each transformed image
   - Calculate AP (Average Precision) for blue and yellow cones
   - Find optimal brightness multiplier where AP reaches target threshold

3. **Save results:**
   - Optimal brightness configuration
   - Original vs tuned brightness scores
   - AP metrics for blue and yellow cones
   - Confidence levels

## Usage

### Quick Start (Recommended)

```bash
# From project root
bash scripts/run_brightness_tuning.sh --model yolov5

# With custom target AP
bash scripts/run_brightness_tuning.sh --model yolov8 --target-ap 0.7

# On test dataset
bash scripts/run_brightness_tuning.sh --dataset test --model yolov12
```

### Direct Python Usage

```bash
cd utils/imgDarkBright_tuning

# Sample dataset
python tune_brightness.py --dataset fsoco_sample --model yolov5 --target_ap 0.8

# Test dataset
python tune_brightness.py --dataset test --model yolov8 --target_ap 0.7

# Custom paths
python tune_brightness.py \
    --images_path ../../data/base_images/custom/images \
    --original_detections_path ../../results/yolov5/original \
    --gt_detections_filepath ../../data/base_images/custom/labels \
    --model yolov5 \
    --target_ap 0.75 \
    --output_dir ../../data/transformed_images/custom_tuning
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (fsoco_sample, test, or custom) | `fsoco_sample` |
| `--model` | YOLO model (yolov5, yolov8, yolov12) | Required |
| `--target_ap` | Target AP threshold to search for | `0.8` |
| `--images_path` | Path to input images | Auto-configured from dataset |
| `--original_detections_path` | Path to model detections | Auto-configured: `results/<model>/original` |
| `--gt_detections_filepath` | Path to ground truth labels | Auto-configured from dataset |
| `--output_dir` | Output directory for results | `data/transformed_images/<dataset>_tuned_brightness` |

## Prerequisites

1. **Original detections must exist:**
   ```bash
   # Run inference first
   bash scripts/run_pipeline.sh --model v5
   # Or individual inference
   python models/yolov5/infer_v5_kfold.py --weights models/yolov5/yolov5_param_kfold.pt --image <image_path>
   ```

2. **Ground truth labels required:**
   - Sample dataset: `data/base_images/fsoco_sample/bounding_boxes/`
   - Test dataset: `data/base_images/test/labels/`

## Output

Results saved to: `data/transformed_images/<dataset>_tuned_brightness/results.json`

### JSON Structure

```json
[
  {
    "image_path": "amz_00016.jpg",
    "brightness_scores": {
      "rms": 0.234,
      "low_high_regions_ratio": 0.012,
      "mean_shift_variance": 0.456,
      "luminosity_709": 0.567
    },
    "tuned_brightness_scores": {
      "rms": 0.189,
      "low_high_regions_ratio": 0.023,
      "mean_shift_variance": 0.389,
      "luminosity_709": 0.445
    },
    "optimal_config": {
      "brightness_multiplier": 0.65,
      "gamma": 1.0,
      "method": "gamma"
    },
    "ap_blue": 0.792,
    "ap_yellow": 0.815,
    "original_confidence_level": 0.892,
    "updated_confidence_level": 0.751
  }
]
```

## Files

| File | Purpose |
|------|---------|
| `tune_brightness.py` | Main entry point - orchestrates tuning workflow |
| `search_brightness.py` | Search algorithm for finding optimal brightness |
| `mean_shift.py` | Mean shift segmentation for brightness variance |
| `improved_confidence.py` | Confidence calibration utilities |
| `split_test_data.py` | Utilities for train/test splits |

## Research Applications

### RQ2: Operational Brightness Ranges

Use the optimal brightness configurations to define safe operational ranges:

```python
import json

# Load results
with open('data/transformed_images/fsoco_sample_tuned_brightness/results.json') as f:
    results = json.load(f)

# Analyze brightness ranges
for result in results:
    multiplier = result['optimal_config']['brightness_multiplier']
    ap_blue = result['ap_blue']
    ap_yellow = result['ap_yellow']
    print(f"{result['image_path']}: brightness={multiplier:.2f}, AP_blue={ap_blue:.3f}, AP_yellow={ap_yellow:.3f}")
```

### RQ2.a: Brightness Scoring Methods

Compare different brightness scoring methods:
- RMS contrast
- Low/high saturation ratio
- Mean shift variance
- BT.709 luminosity

Determine which metric best predicts the degradation threshold found by the search algorithm.

### RQ3: Confidence Calibration

Use confidence levels to analyze miscalibration:

```python
for result in results:
    original_conf = result['original_confidence_level']
    updated_conf = result['updated_confidence_level']
    ap_drop = result['ap_blue']  # or ap_yellow
    
    # Does confidence drop match AP drop?
    print(f"Confidence drop: {original_conf - updated_conf:.3f}, AP: {ap_drop:.3f}")
```

## Example Workflow

```bash
# 1. Run main pipeline to get detections
bash scripts/run_pipeline.sh --model v5

# 2. Run brightness tuning
bash scripts/run_brightness_tuning.sh --model yolov5 --target-ap 0.8

# 3. Analyze results
python -c "
import json
with open('data/transformed_images/fsoco_sample_tuned_brightness/results.json') as f:
    results = json.load(f)
    
print(f'Processed {len(results)} images')
avg_multiplier = sum(r['optimal_config']['brightness_multiplier'] for r in results) / len(results)
print(f'Average optimal brightness multiplier: {avg_multiplier:.3f}')
"
```

## Troubleshooting

**Error: "No detections found"**
- Run inference first: `bash scripts/run_pipeline.sh --model v5`

**Error: "Ground truth labels not found"**
- Ensure labels exist in `data/base_images/<dataset>/labels/` or `bounding_boxes/`

**Search takes too long:**
- Reduce target AP (e.g., `--target-ap 0.7` instead of `0.8`)
- Use fewer images for initial testing
- Check search algorithm parameters in `search_brightness.py`

**Low AP values:**
- Verify ground truth labels are correct
- Check that detections format matches expected format
- Ensure image and label filenames match
