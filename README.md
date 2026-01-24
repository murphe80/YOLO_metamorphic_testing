# YOLO Metamorphic Testing Framework

A framework for testing YOLO object detection models using metamorphic testing. This repository allows for an evaluation of model robustness through various image transformations, specifically designed for the context of Formula Trinity's autonomous team.

## Overview

Metamorphic testing is a software testing technique that validates the behavior of machine learning models by applying transformations to input data and checking if the model's outputs follow expected relationships. This framework applies this concept to object detection models by:

- **Transforming images** using various metamorphic rules (blur, rotation, brightness changes, etc.)
- **Running inference** on both original and transformed images
- **Comparing results** to identify robustness issues and performance degradation
- **Generating visualizations** to analyze detection quality across transformations

## Features

### Available Metamorphic Transformations

- **Geometric Transformations**: stretching
- **Visual Quality**: Motion blur, resolution reduction
- **Color/Contrast**: Brightness adjustment, grayscale conversion, color channel manipulation

### Supported Models

- YOLOv5 
- YOLOv8
- YOLOv12 

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/murphe80/YOLO_metamorphic_testing.git
cd YOLO_metamorphic_testing

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

#### Single Image Testing
```bash
# Test a single image with default transformations
python models/yolov5/infer_v5.py \
    --weights models/yolov5/yolov5_parameters.pt \
    --image data/base_images/fsoco_sample/images/amz_00016.jpg \
    --rules imgBlur imgRotation imgDarkBright
```

### 3. Available Transformation Rules

| Rule | Description | Parameters |
|------|-------------|------------|
| `imgBlur` | Apply bilateral blur filter | `blur_strength`, `filter_size` |
| `imgDarkBright` | Adjust brightness/contrast | `brightness_constant`, `brightness_multiplier`, `method` |
| `imgResolution` | Reduce image resolution | `size_ratio` |
| `imgGrayscale` | Convert to grayscale | None |
| `imgMotionBlur` | Apply motion blur effect | `kernel_size`, `orientation` |

### 4. Output

The framework generates:
- **Detection results** printed to console
- **Visualization images** saved to `results/` directory


## Project Structure

```
YOLO_metamorphic_testing/
├── models/
│   └── yolov5/
│       ├── infer_v5.py          # Basic inference script
│       ├── metamorphic_test.py  # Metamorphic testing script
│       └── yolov5_parameters.pt # Model weights
├── utils/
│   ├── metamorphic_rules/       # Transformation rules
│   │   ├── image_rules.py       # Image transformation implementations
│   │   ├── instantiate_rules.py # Rule factory functions
│   │   └── metamorphic_rules.py # Base rule classes
│   └── visualise.py             # Visualization utilities
├── data/
│   └── fsoco_sample/            # Sample dataset
├── results/                     # Output directory
└── requirements.txt             # Dependencies
```



## Troubleshooting

### NumPy Compatibility Issues

If you encounter the error:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 as it may crash.
```

**Solution:**
1. Activate your virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Downgrade NumPy to a compatible version:
   ```bash
   pip install "numpy<2.0" --force-reinstall
   ```

3. Reinstall OpenCV to ensure compatibility:
   ```bash
   pip uninstall opencv-python opencv-python-headless opencv-contrib-python-headless -y
   pip install opencv-python==4.8.1.78
   ```

### PyTorch Model Loading Issues

If you encounter the error:
```
WeightsUnpickler error: Unsupported global: GLOBAL models.yolo.DetectionModel was not an allowed global by default.
```

**Solution:**
This is due to PyTorch 2.6+ security changes. You need to modify the YOLOv5 experimental.py file:

1. Navigate to the YOLOv5 installation in your virtual environment:
   ```bash
   find venv -name "experimental.py" -path "*/yolov5/models/*"
   ```

2. Edit the file and find the `torch.load` call (around line 82):
   ```python
   # Change this line:
   ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
   
   # To this:
   ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)  # load
   ```

3. Save the file and try running your script again.

```
