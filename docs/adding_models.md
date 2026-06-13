# Adding New Models to the Framework

This guide explains how to add new YOLO models (e.g., YOLOv26) to the metamorphic testing framework.

## Overview

The framework uses a **centralized model registry** (`config/models.json`) as the single source of truth for all model configurations. This makes adding new models straightforward and eliminates the need to update multiple files across the codebase.

## Quick Start: Adding YOLOv26

### Step 1: Create Model Directory

Create a new directory for your model under `/models`:

```bash
mkdir models/yolov26
```

### Step 2: Add Inference Script

Copy an existing inference script and modify it:

```bash
cp models/yolov12/infer_v12.py models/yolov26/infer_v26.py
```

Update the script:
- Change all references from `yolov12` to `yolov26`  
- Change all references from `YOLOv12` to `YOLOv26`
- Update the results directory path from `results/yolov12` to `results/yolov26`

```python
# Key changes in infer_v26.py:
def load_model(weights, device="cuda:0"):
    """Load YOLOv26 model."""  # Update comment
    ...

results_dir = os.path.join("./results/yolov26", args.output_subdir)  # Update path
```

### Step 3: Add Model Weights

Place your trained model weights in the model directory:

```bash
# Example
cp /path/to/your/weights.pt models/yolov26/yolov26_param_kfold.pt
```

### Step 4: Update Model Registry

**This is the only configuration file you need to update!**

Edit `config/models.json` and add your new model entry:

```json
{
  "models": [
    {
      "name": "yolov5",
      "display_name": "YOLOv5",
      "version": "v5",
      "weights": "models/yolov5/yolov5_param_kfold.pt",
      "inference_script": "models/yolov5/infer_v5_kfold.py",
      "results_dir": "results/yolov5",
      "enabled": true,
      "color": {
        "rgb": "rgb(25, 118, 210)",
        "hex": "#1976D2"
      }
    },
    {
      "name": "yolov26",
      "display_name": "YOLOv26",
      "version": "v26",
      "weights": "models/yolov26/yolov26_param_kfold.pt",
      "inference_script": "models/yolov26/infer_v26.py",
      "results_dir": "results/yolov26",
      "enabled": true,
      "color": {
        "rgb": "rgb(255, 87, 34)",
        "hex": "#FF5722"
      }
    }
  ],
  "default_inference_params": {
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "device": "cuda:0"
  }
}
```

### Step 5: Test Your Model

Verify the model is registered correctly:

```bash
# List all models
python utils/model_registry.py --list

# Get model configuration
python utils/model_registry.py --get yolov26

# Validate model files exist
python utils/model_registry.py --validate yolov26
```

### Step 6: Run the Pipeline

Your new model is now automatically included in all pipelines:

```bash
# Run only YOLOv26
bash scripts/run_pipeline.sh --model v26

# Run all models (including YOLOv26)
bash scripts/run_pipeline.sh --model all
```

## Configuration Fields

### Required Fields

- **`name`** (string): Internal model identifier (e.g., `"yolov26"`)
  - Used in code and as directory names
  - Should be lowercase, no spaces
  
- **`display_name`** (string): Human-readable name (e.g., `"YOLOv26"`)
  - Used in console output and visualizations
  
- **`version`** (string): Short version identifier (e.g., `"v26"`)
  - Used with `--model` CLI argument
  - Should match pattern: `v{number}`
  
- **`weights`** (string): Path to model weights file
  - Relative to repository root
  - Example: `"models/yolov26/yolov26_param_kfold.pt"`
  
- **`inference_script`** (string): Path to inference script
  - Relative to repository root
  - Example: `"models/yolov26/infer_v26.py"`
  
- **`results_dir`** (string): Where to save detection results
  - Relative to repository root
  - Example: `"results/yolov26"`

### Optional Fields

- **`enabled`** (boolean, default: `true`): Whether to include this model
  - Set to `false` to temporarily disable without deleting the entry
  
- **`color`** (object): Color scheme for visualizations
  - `rgb`: CSS RGB color string (e.g., `"rgb(255, 87, 34)"`)
  - `hex`: Hex color code (e.g., `"#FF5722"`)

## Automatic Integration

Once you've updated `config/models.json`, your new model is **automatically integrated** into:

- ✅ **Pipeline scripts** (`scripts/run_pipeline.sh`)
- ✅ **Analysis scripts** (`analysis/scripts/run_all_analysis_matias_meta.sh`)
- ✅ **Visualization tools** (`analysis/visualisation/visualise_ap.py`)
- ✅ **Dashboard** (via `/api/models` endpoint)
- ✅ **All cone color transformations**
- ✅ **Label transformation pipelines**

No need to manually update any of these scripts!

## Model Registry API

### Python API

```python
from utils.model_registry import (
    get_models, 
    get_model_by_name, 
    get_model_names,
    validate_model_files
)

# Get all enabled models
models = get_models()

# Get specific model
model = get_model_by_name("yolov26")
print(model["weights"])  # models/yolov26/yolov26_param_kfold.pt

# Get list of model names
names = get_model_names()  # ['yolov5', 'yolov8', 'yolov12', 'yolov26']

# Validate model files exist
result = validate_model_files("yolov26")
if result["weights"] and result["script"]:
    print("Model ready!")
```

### Shell API

```bash
# Source the helper functions
source utils/model_registry.sh

# Get all model names (space-separated)
MODEL_NAMES=$(get_model_names)

# Get specific model field
WEIGHTS=$(get_model_field "yolov26" "weights")
SCRIPT=$(get_model_field "yolov26" "inference_script")

# Iterate over all models
for model in $(get_model_names); do
    echo "Processing $model..."
    weights=$(get_model_field "$model" "weights")
    script=$(get_model_field "$model" "inference_script")
    # ... your code here
done

# Validate version argument
if validate_model_version "$USER_INPUT"; then
    MODEL_NAME=$(get_model_by_version "$USER_INPUT")
fi
```

### CLI Tool

```bash
# List all models
python utils/model_registry.py --list

# List model versions
python utils/model_registry.py --list-versions

# Get model config as JSON
python utils/model_registry.py --get yolov26

# Validate model files
python utils/model_registry.py --validate yolov26

# Include disabled models
python utils/model_registry.py --list --include-disabled
```

## Disabling Models Temporarily

To temporarily disable a model without deleting it:

```json
{
  "name": "yolov26",
  "enabled": false,
  ...
}
```

Disabled models are excluded from `get_models()` and won't run in pipelines.

## Advanced: Custom Inference Parameters

Override default inference parameters per model (future enhancement):

```json
{
  "name": "yolov26",
  "inference_params": {
    "conf_thres": 0.30,
    "iou_thres": 0.50
  },
  ...
}
```

## Troubleshooting

### Model not showing up in pipeline

1. Check spelling in `config/models.json`:
```bash
python utils/model_registry.py --list
```

2. Verify `enabled: true`

3. Check file paths exist:
```bash
python utils/model_registry.py --validate yolov26
```

### Inference script errors

1. Ensure script is executable:
```bash
chmod +x models/yolov26/infer_v26.py
```

2. Test script directly:
```bash
python models/yolov26/infer_v26.py \
  --weights models/yolov26/yolov26_param_kfold.pt \
  --image data/base_images/fsoco_sample/images/test_image.jpg
```

3. Check script paths match `results_dir` in config

### "Model not found" errors

Make sure:
- Model name in config matches exactly (case-sensitive)
- Version string matches (e.g., `v26` not `V26`)

## Migration Guide (Old Setup → New Setup)

If you have scripts using the old hardcoded approach, here's how to migrate:

**Old way:**
```bash
V26_WEIGHTS="models/yolov26/yolov26_param_kfold.pt"
V26_SCRIPT="models/yolov26/infer_v26.py"
run_model "$V26_SCRIPT" "$V26_WEIGHTS" "YOLOv26"
```

**New way:**
```bash
source utils/model_registry.sh
for model in $(get_model_names); do
    script=$(get_model_field "$model" "inference_script")
    weights=$(get_model_field "$model" "weights")
    display=$(get_model_field "$model" "display_name")
    run_model "$script" "$weights" "$display"
done
```

## Benefits of the Registry Pattern

1. **Single source of truth** - One file to update instead of 10+
2. **Automatic integration** - New models work everywhere immediately  
3. **Easy enabling/disabling** - Toggle `enabled` flag
4. **Consistency** - Model info (colors, names) stays synchronized
5. **Validation** - Built-in checks for missing files
6. **Extensibility** - Easy to add new fields (e.g., model metadata)

## See Also

- [Model Registry API Reference](../utils/model_registry.py)
- [Shell Helper Functions](../utils/model_registry.sh)
- [Configuration Format](../config/models.json)
