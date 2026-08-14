# RQ2.a Usage Guide: Gamma Search Scripts

## Overview

Two scripts implement different search strategies to find operational gamma thresholds:

1. **Brute Force Search** (`rq2a_brute_force_search.py`)
   - Linear sweep through gamma values
   - Step size: 0.05
   - ~90 iterations per image

2. **Golden-Section Search** (`rq2a_golden_section_search.py`)
   - Adaptive search algorithm
   - ~15-20 iterations per image (expected)
   - 3-4x faster than brute force

Both scripts find **left** and **right** gamma crossing points where mAP = target_ap.

---

## Quick Start

### Test with 1 Image

```bash
# Brute force (takes ~5-10 minutes per image)
python scripts/rq2a_brute_force_search.py \
    --model yolov8 \
    --num-images 1 \
    --target-ap 0.7

# Golden-section (takes ~2-3 minutes per image)
python scripts/rq2a_golden_section_search.py \
    --model yolov8 \
    --num-images 1 \
    --target-ap 0.7
```

### Test with 10 Images

```bash
# Brute force
python scripts/rq2a_brute_force_search.py \
    --model yolov8 \
    --num-images 10 \
    --target-ap 0.7

# Golden-section
python scripts/rq2a_golden_section_search.py \
    --model yolov8 \
    --num-images 10 \
    --target-ap 0.7
```

### Run on All Test Images

```bash
# Brute force (WARNING: may take hours!)
python scripts/rq2a_brute_force_search.py \
    --model yolov8 \
    --target-ap 0.7

# Golden-section
python scripts/rq2a_golden_section_search.py \
    --model yolov8 \
    --target-ap 0.7
```

---

## Command-Line Options

### Common Options (both scripts)

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | From `config/models.json` | Any enabled model defined in `config/models.json` |
| `--dataset` | `test` | Dataset under data/base_images/ |
| `--num-images` | All | Number of images to process |
| `--target-ap` | `0.7` | Target mAP threshold |
| `--device` | `cuda:0` | Torch device (cuda:0, cpu) |
| `--output-dir` | Auto | Output directory path |

### Examples

```bash
# Different model
python scripts/rq2a_brute_force_search.py --model yolov5

# Different target AP
python scripts/rq2a_golden_section_search.py --target-ap 0.6

# CPU only
python scripts/rq2a_brute_force_search.py --device cpu

# Custom output directory
python scripts/rq2a_golden_section_search.py \
    --output-dir my_experiments/test1
```

---

## Output Structure

Both scripts produce identical JSON output format for easy comparison.

### Directory Structure

```
analysis/results/
├── rq2a_brute_force_yolov8/
│   ├── results.json                    # Main results file
│   ├── amz_00811/                      # Per-image results
│   │   ├── original_detections/
│   │   ├── gamma_0.50/
│   │   ├── gamma_0.55/
│   │   └── ...
│   └── amz_00832/
│       └── ...
│
└── rq2a_golden_section_yolov8/
    ├── results.json
    ├── amz_00811/
    │   ├── original_detections/
    │   ├── gamma_2.145_peak_m1_init/
    │   ├── gamma_3.090_peak_m2_init/
    │   └── ...
    └── ...
```

### JSON Format

```json
{
  "method": "brute_force" | "golden_section",
  "model": "yolov8",
  "dataset": "test",
  "target_ap": 0.7,
  "total_images": 10,
  "successful_images": 9,
  "failed_images": 1,
  
  "results": [
    {
      "image_name": "amz_00811.png",
      "original_ap": 0.92,
      "peak_gamma": 1.85,
      "peak_ap": 0.95,
      "gamma_left": 1.20,
      "ap_left": 0.698,
      "gamma_right": 2.65,
      "ap_right": 0.702,
      "iterations": 45,
      "inference_count": 45,
      "time_seconds": 312.4
    }
  ],
  
  "failures": [
    {
      "image": "amz_00999.png",
      "error": "No detections found"
    }
  ],
  
  "summary": {
    "avg_gamma_left": 1.18,
    "avg_gamma_right": 2.73,
    "avg_peak_gamma": 1.87,
    "avg_iterations": 43.2,
    "avg_inference_count": 43.2,
    "total_time_seconds": 2811.6,
    "avg_time_per_image": 312.4
  }
}
```

---

## Understanding the Results

### Key Metrics

1. **gamma_left**: Lower gamma threshold (darker image)
   - Defines darkest acceptable condition
   - Below this, AP drops below target

2. **gamma_right**: Upper gamma threshold (brighter image)
   - Defines brightest acceptable condition
   - Above this, AP drops below target

3. **peak_gamma**: Optimal gamma (peak AP)
   - Best brightness setting
   - Typically between left and right crossings

4. **Operational range**: `[gamma_left, gamma_right]`
   - Safe operating window for this image
   - Model maintains AP ≥ target_ap in this range

### Example Interpretation

```json
{
  "gamma_left": 1.20,
  "gamma_right": 2.65,
  "peak_gamma": 1.85,
  "target_ap": 0.7
}
```

**Interpretation:**
- Model works acceptably in gamma range [1.20, 2.65]
- Optimal performance at gamma=1.85
- Below gamma=1.20 (too dark): AP < 0.7
- Above gamma=2.65 (too bright): AP < 0.7

---

## Comparing Methods

After running both scripts, compare efficiency:

```python
import json

# Load results
with open('analysis/results/rq2a_brute_force_yolov8/results.json') as f:
    bf_results = json.load(f)

with open('analysis/results/rq2a_golden_section_yolov8/results.json') as f:
    gs_results = json.load(f)

# Compare
print(f"Brute Force:")
print(f"  Avg iterations: {bf_results['summary']['avg_iterations']:.0f}")
print(f"  Avg time: {bf_results['summary']['avg_time_per_image']:.1f}s")

print(f"\nGolden-Section:")
print(f"  Avg iterations: {gs_results['summary']['avg_iterations']:.0f}")
print(f"  Avg time: {gs_results['summary']['avg_time_per_image']:.1f}s")

print(f"\nSpeedup: {bf_results['summary']['avg_time_per_image'] / gs_results['summary']['avg_time_per_image']:.2f}x")
```

---

## Expected Runtime

Per image estimates (GPU):

| Method | Iterations | Time per Image | 10 Images | 100 Images |
|--------|-----------|----------------|-----------|------------|
| Brute Force | ~90 | 5-10 min | 50-100 min | 8-16 hours |
| Golden-Section | ~15-20 | 2-3 min | 20-30 min | 3-5 hours |

**Note:** CPU execution will be 5-10x slower.

---

## Troubleshooting

### Error: Model weights not found

```
FileNotFoundError: models/yolov8_param_kfold.pt
```

**Solution:** Ensure model weights exist at specified path:
```bash
ls models/yolov8_param_kfold.pt
```

### Error: Images directory not found

```
Error: Images directory not found: data/base_images/test/images
```

**Solution:** Check dataset exists:
```bash
ls data/base_images/test/images/
```

### Error: COCO evaluation failed

```
Warning: COCO evaluation failed: ...
```

**Causes:**
- No detections in transformed image
- Ground truth labels missing
- Format mismatch

**Solution:** Check ground truth format:
```bash
# Check labels exist
ls data/base_images/test/labels/

# Verify format (should be: class conf x1 y1 x2 y2)
head data/base_images/test/labels/amz_00811.txt
```

### Target AP unreachable

```
⚠ Target AP 0.7 is unreachable (peak=0.65)
```

**Meaning:** Even at optimal gamma, AP doesn't reach target.

**Solutions:**
1. Lower target AP: `--target-ap 0.6`
2. Check if original image has low AP (poor quality)
3. Widen gamma range (modify `GAMMA_RANGE` in script)

### Script hangs or very slow

**Causes:**
- Running on CPU
- Large images
- Many images

**Solutions:**
```bash
# Use GPU
python script.py --device cuda:0

# Test with fewer images first
python script.py --num-images 3

# Use golden-section instead of brute force
python scripts/rq2a_golden_section_search.py ...
```

---

## Next Steps

### 1. Run Initial Tests

Start with 1 image to validate:
```bash
python scripts/rq2a_golden_section_search.py --num-images 1
```

### 2. Scale to 10 Images

```bash
python scripts/rq2a_golden_section_search.py --num-images 10
```

### 3. Run Full Experiment

```bash
# Golden-section on all images
python scripts/rq2a_golden_section_search.py

# Brute force on subset for comparison
python scripts/rq2a_brute_force_search.py --num-images 10
```

### 4. Analyze Results

Compare efficiency metrics:
- Iteration count
- Time per image
- Gamma accuracy (do both methods find similar crossings?)

### 5. Visualize (TODO)

Create plots:
- Gamma vs AP curves
- Operational range distributions
- Method comparison

---

## Configuration

To modify search parameters, edit the constants at the top of each script:

### Brute Force (`rq2a_brute_force_search.py`)

```python
GAMMA_RANGE = (0.5, 5.0)  # Search range
GAMMA_STEP = 0.05         # Step size (smaller = more accurate, slower)
AP_TOLERANCE = 0.02       # Tolerance for target matching
```

### Golden-Section (`rq2a_golden_section_search.py`)

```python
GAMMA_LOW = 0.5           # Lower bound
GAMMA_HIGH = 5.0          # Upper bound
TOLERANCE_GAMMA = 0.01    # Convergence threshold
AP_TOLERANCE = 0.02       # Tolerance for target matching
MAX_ITERATIONS = 30       # Safety limit
```

---

## Research Questions Addressed

These scripts answer **RQ2.a**:

> Can golden-section search replace brute force for finding operational gamma thresholds?

**Analysis points:**
1. **Efficiency:** Iterations and time comparison
2. **Accuracy:** Do both methods find similar gamma values?
3. **Robustness:** Success rate across different images
4. **Practical value:** Is speedup worth potential accuracy loss?

---

## Citation

If using these scripts in research:

```bibtex
@software{rq2a_gamma_search,
  title={RQ2.a Gamma Search Scripts},
  author={Murphy, Emma},
  year={2026},
  note={YOLO Metamorphic Testing - Brightness Operational Range}
}
```
