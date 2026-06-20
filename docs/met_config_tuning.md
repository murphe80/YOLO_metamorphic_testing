# Configuration Tuning Framework

Automated brute-force search for optimal metamorphic transformation configurations that maximize model performance disparities between YOLOv5, v8, and v12.

## Overview

The framework systematically explores parameter spaces for 6 metamorphic transformations, running 10 configuration variants per transformation across CPU-only Daytona sandboxes. It identifies which configurations best reveal robustness disparities between YOLO models.

### Key Features

- **10 variants per transformation** - Linear parameter sweeps across meaningful ranges
- **Parallel execution** - 10 Daytona sandboxes process variants simultaneously
- **CPU-based inference** - Each sandbox runs YOLO inference on CPU (no GPU required)
- **Disparity-based ranking** - Quantifies model performance differences
- **Visual analysis** - Automatic generation of disparity curves and reports

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Main Orchestrator (tune_metamorphic_configs.py)                │
│  - Generates configs → Provisions sandboxes → Runs experiments  │
│  - Aggregates results → Ranks by disparity → Exports reports   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                   ↓                   ↓
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Sandbox 0    │   │  Sandbox 1    │   │  Sandbox 9    │
│  CPU-only     │   │  CPU-only     │   │  CPU-only     │
├───────────────┤   ├───────────────┤   ├───────────────┤
│ Transform     │   │ Transform     │   │ Transform     │
│ Infer v5/8/12 │   │ Infer v5/8/12 │   │ Infer v5/8/12 │
│ Analyze       │   │ Analyze       │   │ Analyze       │
│ Return AP50   │   │ Return AP50   │   │ Return AP50   │
└───────────────┘   └───────────────┘   └───────────────┘
```

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key:**
   ```bash
   # Add to .env file
   echo "DAYTONA_API_KEY=your_key_here" >> .env
   ```

3. **Test framework:**
   ```bash
   python scripts/test_framework.py
   ```

## Usage

### Quick Start

```bash
# Run with defaults: all 6 transformations, 10 sandboxes, fsoco_sample dataset
python scripts/tune_metamorphic_configs.py
```

### Custom Configuration

```bash
# Select specific transformations
python scripts/tune_metamorphic_configs.py \
    --transformations imgResolution imgColorWheel imgDarkBright

# Use full test dataset
python scripts/tune_metamorphic_configs.py \
    --dataset fsoco_test

# Specify number of sandboxes
python scripts/tune_metamorphic_configs.py \
    --num-sandboxes 5

# Custom output directory
python scripts/tune_metamorphic_configs.py \
    --output-dir analysis/results/my_experiment
```

### All Options

```
--num-sandboxes       Number of Daytona sandboxes (default: 10)
--dataset             Dataset: fsoco_sample or fsoco_test (default: fsoco_sample)
--transformations     Space-separated list (default: all 6)
--output-dir          Output directory (default: analysis/results/config_tuning)
--num-sample-images   Sample images per config for logging (default: 3)
```

## Transformation Parameters

| Transformation | Parameter Varied | Range | Total Variants |
|----------------|------------------|-------|----------------|
| `imgGrayscale` | (none) | N/A | 10 (identical) |
| `imgColorWheel` | `hue_rotation` | 0 to 180° | 10 |
| `imgResolution` | `size_ratio` | 0.1 to 1.0 | 10 |
| `imgDarkBright` | `gamma` | 0.5 to 5.0 | 10 |
| `imgMotionBlur` | `kernel_size` | 3 to 21 (odd) | 10 |
| `imgStretch` | `width_ratio` | 1.0 to 2.8 | 10 |

**Total:** 60 configurations (6 transformations × 10 variants)

## Disparity Metric

The framework ranks configurations by their **disparity score**:

```python
disparity = max(
    |yolov5_AP50 - yolov8_AP50|,
    |yolov5_AP50 - yolov12_AP50|,
    |yolov8_AP50 - yolov12_AP50|
)
```

**Higher disparity = better at revealing model differences**

- Range: [0, 1]
- 0 = all models perform identically
- 1 = maximum divergence (one model perfect, another fails)

## Output Files

### Directory Structure

```
analysis/results/config_tuning/
├── ranked_configs.csv                    # All variants ranked by disparity
├── experiment_results.json               # Raw AP50 scores for all experiments
├── summary_report.md                     # Markdown report with top configs
└── disparity_curves/                     # Visualizations
    ├── imgResolution.png
    ├── imgColorWheel.png
    ├── imgDarkBright.png
    ├── imgMotionBlur.png
    ├── imgStretch.png
    └── imgGrayscale.png

config/
└── optimal_transformations.json          # Top 3 configs per transformation

logs/config_samples/                      # Sample transformed images (NEW)
├── imgResolution_0/
│   ├── sample_000_img1.jpg
│   ├── sample_001_img2.jpg
│   └── sample_002_img3.jpg
├── imgResolution_1/
│   └── ...
├── ... (60 total variant directories)
└── INDEX.md                              # Catalog of all sample images
```

### ranked_configs.csv

CSV with all 60 configurations ranked by disparity:

```csv
variant_id,transformation,disparity,yolov5_ap50,yolov8_ap50,yolov12_ap50,rank,config
imgResolution_2,imgResolution,0.18,0.85,0.67,0.82,1,"{\"size_ratio\": 0.3}"
imgDarkBright_7,imgDarkBright,0.15,0.78,0.63,0.75,2,"{\"gamma\": 4.0, ...}"
...
```

### optimal_transformations.json

Top-ranked configurations for use in future experiments:

```json
{
  "imgResolution": {
    "rank_1": {
      "config": {"size_ratio": 0.3},
      "disparity": 0.18,
      "scores": {"yolov5": 0.85, "yolov8": 0.67, "yolov12": 0.82}
    },
    "rank_2": {...},
    "rank_3": {...}
  },
  ...
}
```

### Disparity Curves

Each transformation gets 2 plots:
1. **Disparity vs Parameter** - Shows which parameter values maximize model differences
2. **AP50 vs Parameter** - Individual model performance curves

![Example Disparity Curve](../test_output/disparity_curves/imgResolution.png)

### Sample Images (Transparency Log)

For transparency and verification, the framework automatically downloads sample transformed images from each configuration:

- **3 sample images per configuration** (default, configurable)
- **Organized by variant:** `logs/config_samples/<variant_id>/`
- **Indexed catalog:** `logs/config_samples/INDEX.md`

**Purpose:**
- Visual verification of transformations
- Quality assurance
- Debugging configuration issues
- Research transparency and reproducibility

**Example INDEX.md:**
```markdown
## imgResolution

### imgResolution_0
**Configuration:** `{"size_ratio": 0.1, "variant_id": "imgResolution_0"}`
**Sample images:** 3

- `logs/config_samples/imgResolution_0/sample_000_img1.jpg`
- `logs/config_samples/imgResolution_0/sample_001_img2.jpg`
- `logs/config_samples/imgResolution_0/sample_002_img3.jpg`
```

**Customize sample count:**
```bash
python scripts/tune_metamorphic_configs.py --num-sample-images 5
```

## Expected Runtime

| Configuration | Time per Transformation | Total Time |
|---------------|-------------------------|------------|
| 10 sandboxes, fsoco_sample | ~20-30 min | ~2-3 hours (all 6) |
| 10 sandboxes, fsoco_test | ~45-60 min | ~5-6 hours (all 6) |
| 5 sandboxes, fsoco_sample | ~40-50 min | ~4-5 hours (all 6) |

**Note:** Times are estimated for CPU-only sandboxes with model weights already downloaded.

## Resource Requirements

### Per Sandbox
- **CPU:** 4 cores
- **Memory:** 8 GiB (max: 8 GiB)
- **Disk:** 10 GiB (max: 10 GiB)
- **GPU:** 0 (CPU-only)

### Total (10 sandboxes)
- **CPU:** 40 cores
- **Memory:** 80 GiB
- **Disk:** 100 GiB
- **GPU:** 0

## Troubleshooting

### Sandbox Provisioning Fails

**Issue:** `RuntimeError: Failed to provision sandbox after 3 attempts`

**Solutions:**
- Check Daytona API key is valid: `echo $DAYTONA_API_KEY`
- Verify account has sufficient quota for CPU resources
- Reduce `--num-sandboxes` to lower resource requirements
- Check Daytona service status

### Experiment Timeout

**Issue:** Experiment exceeds 30-minute timeout

**Solutions:**
- Use `fsoco_sample` instead of `fsoco_test` for testing
- Reduce number of images per experiment (modify `ExperimentRunner._generate_transformed_images`)
- Increase timeout in `sandbox_manager.py` (default: 1800000ms)

### Missing Model Weights

**Issue:** `FileNotFoundError: models/yolov5/yolov5_param_kfold.pt`

**Solutions:**
- Ensure model weights exist locally before running
- Update `setup_sandbox()` to download weights from remote storage
- Manually upload weights to sandboxes

### Low Disparity Scores

**Issue:** All configurations have similar disparity (~0.02)

**Solutions:**
- Models may be equally robust to this transformation
- Try different parameter ranges in `config_generator.py`
- Use `fsoco_test` instead of `fsoco_sample` (more diverse images)
- Consider alternative disparity metrics (std, range, cv)

## Advanced Usage

### Custom Parameter Ranges

Edit `utils/config_generator.py` to modify parameter ranges:

```python
def _generate_resolution_configs() -> List[Dict]:
    # Change from [0.1, 0.2, ..., 1.0]
    size_ratios = np.linspace(0.1, 1.0, 10)
    
    # To [0.2, 0.3, ..., 1.1] for higher resolution focus
    size_ratios = np.linspace(0.2, 1.1, 10)
    
    return [{"size_ratio": float(ratio)} for ratio in size_ratios]
```

### Custom Disparity Metrics

Modify `analysis/utils/config_analysis.py`:

```python
def compute_disparity_metric(ap_scores: Dict[str, float]) -> float:
    # Use standard deviation instead of max difference
    scores = list(ap_scores.values())
    return float(np.std(scores))
```

### Reusing Sandboxes

For multiple experiments, provision once and reuse:

```python
from utils.sandbox_manager import SandboxManager

manager = SandboxManager(api_key=API_KEY, num_sandboxes=10)
manager.provision_sandboxes()
manager.setup_all_sandboxes()

# Run multiple experiments without re-provisioning
# ...

# Cleanup when done
manager.cleanup_sandboxes()
```

## Code Structure

```
utils/
├── config_generator.py          # Generate 10 configs per transformation
└── sandbox_manager.py           # Daytona provisioning & experiment execution

analysis/utils/
└── config_analysis.py           # Disparity computation, ranking, visualization

scripts/
├── tune_metamorphic_configs.py  # Main orchestrator
└── test_framework.py            # Test suite (no Daytona required)
```

## Testing

Run the test suite to verify framework without needing Daytona:

```bash
python scripts/test_framework.py
```

**Tests:**
1. Configuration generation (60 configs)
2. Disparity metric computation
3. Configuration ranking
4. Optimal config export
5. Disparity curve plotting
6. Summary report generation
7. Full workflow integration

All tests should pass with output: `✓ ALL TESTS PASSED`

## Related Documentation

- [Daytona SDK Reference](daytona_sdk.md) - Sandbox API documentation
- [CLAUDE.md](../CLAUDE.md) - Research questions and metamorphic tests
- [Plan](../.claude/plans/lexical-swinging-charm.md) - Implementation plan

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{yolo_config_tuning,
  title={Configuration Tuning Framework for Metamorphic Testing},
  author={Murphy, Emma},
  year={2026},
  howpublished={YOLO Metamorphic Testing Repository}
}
```

## Support

For issues or questions:
1. Check this documentation
2. Run `python scripts/test_framework.py` to verify setup
3. Review logs in experiment output directory
4. Open an issue on GitHub (if applicable)
