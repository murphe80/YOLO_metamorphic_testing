# Metamorphic Testing Dashboard

Dashboard for visualizing and comparing metamorphic testing results across YOLO models 

## Quick Start

### 1. Generate the Manifest

First, generate the manifest file that indexes all available models and tests:

```bash
cd dashboard
python3 generate_manifest.py
```

This scans the `../analysis/results` directory and creates `manifest.json` with available models and tests, including brute-force variant directories such as `imgResolution_3` when analysis stats exist for them.

### 2. Start the Dashboard

Launch the local web server:

```bash
python3 serve.py
```

The dashboard will automatically open in your browser at `http://localhost:8000/dashboard/dashboard.html`

**Note:** The server runs from the project root directory to access both dashboard files and analysis results.

### Alternative: Manual Server

If you prefer to use Python's built-in HTTP server directly:

```bash
cd ..  # Go to project root
python3 -m http.server 8000
```

Then open `http://localhost:8000/dashboard/dashboard.html` in your browser.

## Dashboard Features

### Metric Comparison
- Compare multiple metrics (AP, AP50, AP75, etc.) across models
- Interactive charts showing performance across different comparison types
- Dropdown filters for model, test, and comparison type selection

### Comparison Types
- **Original vs Transformed**: Compare model performance on original images vs metamorphic transformations
- **Transformed vs Ground Truth**: Evaluate transformed images against ground truth annotations
- **Original vs Ground Truth**: Baseline performance of original images against ground truth (no test selection needed)

### Supported Metamorphic Tests
- **General transformations**: Grayscale, Motion Blur, Stretch, Resolution, Color Wheel, Dark/Bright, including brute-force variants such as `imgStretch_4`
- **Cone color transformations**: Yellow→Blue, Blue→Yellow, Blue→Orange, Yellow→Orange
- **Label transformations**: Label removal with different backgrounds (avg, black, white)

