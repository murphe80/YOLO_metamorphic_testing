# Matias Metamorphic Transformations

This module provides image transformations for metamorphic testing of YOLO models.

## Overview

The transformation pipeline has been refactored to:
1. **Centralize configuration** in the bash script for easy modification
2. **Generate `configs.json`** files documenting the exact parameters used
3. **Simplify the Python script** to accept parameters via command line

## Files

### `utils/matias_metamorphic.py`

Python script that applies a single transformation rule to an image.

**Usage:**
```bash
python utils/matias_metamorphic.py \
    --rule imgDarkBright \
    --config '{"brightness_constant": -20, "brightness_multiplier": 0.2}' \
    --base_image path/to/image.jpg \
    --output_path output/directory \
    --save-config
```

**Arguments:**
- `--rule`: Name of the metamorphic rule (e.g., `imgDarkBright`)
- `--config`: JSON string with rule parameters
- `--base_image`: Path to the original image
- `--output_path`: Directory to save transformed images
- `--save-config`: (Optional) Save configuration to `configs.json`

### `scripts/run_all_matias_transformations.sh`

Batch script that applies all transformations to all test images.

**Configuration Section:**
```bash
# Rule configurations (JSON format)
declare -A RULE_CONFIGS
RULE_CONFIGS[imgDarkBright]='{"brightness_constant": -20, "brightness_multiplier": 0.2, "gamma": 1, "method": "gamma"}'
RULE_CONFIGS[imgResolution]='{"size_ratio": 0.6}'
RULE_CONFIGS[imgMotionBlur]='{"kernel_size": 12, "orientation": 45}'
RULE_CONFIGS[imgStretch]='{"height_ratio": 1.5, "width_ratio": 1.5}'
RULE_CONFIGS[imgGrayscale]='{}'
RULE_CONFIGS[imgColorWheel]='{"hue_rotation": 15}'
```

**Usage:**
```bash
./scripts/run_all_matias_transformations.sh
```

## Available Transformations

### 1. imgDarkBright - Brightness/Darkness Adjustment
Adjusts image brightness using gamma correction or linear scaling.

**Parameters:**
- `brightness_constant`: Constant to add to pixel values (-20)
- `brightness_multiplier`: Multiplier for pixel values (0.2)
- `gamma`: Gamma correction value (1)
- `method`: Method to use ("gamma")

### 2. imgResolution - Resolution Scaling
Reduces or increases image resolution.

**Parameters:**
- `size_ratio`: Scaling factor for image dimensions (0.6 = 60% of original)

### 3. imgMotionBlur - Motion Blur Effect
Simulates motion blur with directional kernel.

**Parameters:**
- `kernel_size`: Size of motion blur kernel (12)
- `orientation`: Angle of motion in degrees (45)

### 4. imgStretch - Aspect Ratio Stretching
Stretches image by different factors in height/width.

**Parameters:**
- `height_ratio`: Vertical scaling factor (1.5)
- `width_ratio`: Horizontal scaling factor (1.5)

### 5. imgGrayscale - Grayscale Conversion
Converts image to grayscale.

**Parameters:** None (empty object)

### 6. imgColorWheel - Hue Rotation
Rotates hue values in HSV color space.

**Parameters:**
- `hue_rotation`: Degrees to rotate hue (15)

## Output Structure

```
data/transformed_images/fsoco_test/
├── imgDarkBright/
│   ├── configs.json
│   ├── image1_imgDarkBright.jpg
│   └── image2_imgDarkBright.jpg
├── imgResolution/
│   ├── configs.json
│   └── ...
├── imgMotionBlur/
│   ├── configs.json
│   └── ...
├── imgStretch/
│   ├── configs.json
│   └── ...
├── imgGrayscale/
│   ├── configs.json
│   └── ...
└── imgColorWheel/
    ├── configs.json
    └── ...
```

## configs.json Format

Each output directory contains a `configs.json` file documenting the parameters used:

```json
{
  "imgDarkBright": {
    "brightness_constant": -20,
    "brightness_multiplier": 0.2,
    "gamma": 1,
    "method": "gamma"
  }
}
```

This ensures **reproducibility** and **traceability** of transformations.

## Modifying Configurations

To change transformation parameters:

1. Edit the `RULE_CONFIGS` section in `scripts/run_all_matias_transformations.sh`
2. Run the script - new `configs.json` will be generated with updated parameters
3. Compare results with different parameter sets

**Example - Make images darker:**
```bash
RULE_CONFIGS[imgDarkBright]='{"brightness_constant": -40, "brightness_multiplier": 0.1, "gamma": 1, "method": "gamma"}'
```

**Example - Stronger motion blur:**
```bash
RULE_CONFIGS[imgMotionBlur]='{"kernel_size": 20, "orientation": 90}'
```

## Benefits of New Design

✅ **Single Source of Truth**: All configs in one place (bash script)
✅ **Documentation**: `configs.json` records exact parameters used
✅ **Flexibility**: Easy to modify and experiment with parameters
✅ **Reproducibility**: Anyone can recreate transformations from `configs.json`
✅ **Version Control**: Track parameter changes in git
✅ **Simplified Python**: Script focused on transformation logic only

## Workflow

1. **Configure**: Edit parameters in `run_all_matias_transformations.sh`
2. **Generate**: Run script to create transformed images
3. **Document**: `configs.json` files automatically created
4. **Test**: Run YOLO models on transformed images
5. **Analyze**: Compare detection performance
6. **Iterate**: Modify configs and repeat

