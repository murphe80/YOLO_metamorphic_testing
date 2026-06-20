# YOLO Metamorphic Testing

A metamorphic testing framework for evaluating the robustness of YOLO-based cone detectors. It systematically applies image transformations (blur, brightness, resolution changes, colour shifts, etc.) and checks whether the model's detections remain consistent.

**Supported models:** YOLOv5 · YOLOv8 · YOLOv12

---

## Quick start

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the full pipeline on the bundled sample images (10 images, all 3 models)
bash scripts/run_pipeline.sh
```

Results land in `analysis/results/<model>/<rule>/original_vs_transformed/stats.json`.

---

## How it works

```
data/base_images/fsoco_sample/          ← sample images (bundled)
          │
          ▼  Stage 1: generate transforms
data/transformed_images/fsoco_sample/
          │
          ▼  Stage 2: run YOLO inference
results/<model>/original/               ← detections on original images
results/<model>/<rule>/                 ← detections on transformed images
          │
          ▼  Stage 3: COCO-style evaluation
analysis/results/<model>/<rule>/
    original_vs_transformed/stats.json  ← metamorphic comparison (AP, AR)
    transformed_vs_gt/stats.json        ← vs ground truth (if annotations available)
```

The key metamorphic relation is: a robust detector should produce the same detections on an image regardless of perceptually minor transformations. AP drop between original and transformed detections quantifies robustness.

---

## Running the pipeline

### Full pipeline (one command)

```bash
# Run on bundled sample data (default)
bash scripts/run_pipeline.sh

# Run on held-out test data (requires data/base_images/test/)
bash scripts/run_pipeline.sh --dataset test

# Run specific model only
bash scripts/run_pipeline.sh --model v8

# Skip transformation stage (re-use existing transformed images)
bash scripts/run_pipeline.sh --skip-transform
```

**Options:**

| Flag | Description |
|------|-------------|
| `--dataset <name>` | Dataset to use: `fsoco_sample` (default), `test`, or custom path |
| `--model <v5\|v8\|v12\|all>` | Run inference for one model only (default: `all`) |
| `--skip-transform` | Skip Stage 1 (re-use existing transformed images) |

### Running stages individually

**Stage 1 — Generate transformed images**

```bash
# General metamorphic transformations (blur, brightness, resolution, etc.)
bash scripts/run_all_matias_transformations.sh --dataset fsoco_sample

# Cone color transformations (yellow↔blue, etc.) - requires labels
bash scripts/run_all_cone_color_transformations.sh --dataset fsoco_sample

# Cone label removal - requires segmented images and labels
bash scripts/run_cone_label_transformations.sh --dataset fsoco_sample
```

Or apply a single rule to a single image:

```bash
python utils/matias_metamorphic.py \
    --rule imgDarkBright \
    --config '{"brightness_multiplier": 0.2, "gamma": 1, "method": "gamma"}' \
    --base_image data/base_images/fsoco_sample/segmented_images/amz_00016.jpg \
    --output_path data/transformed_images/fsoco_sample/imgDarkBright
```

**Stage 2 — Run inference**

```bash
# YOLOv5
python models/yolov5/infer_v5_kfold.py \
    --weights models/yolov5/yolov5_param_kfold.pt \
    --image <path/to/image.jpg> \
    --output-subdir original

# YOLOv8
python models/yolov8/infer_v8.py \
    --weights models/yolov8/yolov8_param_kfold.pt \
    --image <path/to/image.jpg> \
    --output-subdir original

# YOLOv12
python models/yolov12/infer_v12.py \
    --weights models/yolov12/yolov12_param_kfold.pt \
    --image <path/to/image.jpg> \
    --output-subdir original
```

**Stage 3 — Evaluate**

```bash
bash analysis/scripts/run_all_analysis_matias_meta.sh --dataset fsoco_sample
```

Output `stats.json` files contain COCO-standard AP and AR metrics at multiple IoU thresholds.

---

## Metamorphic tests

### 1. General metamorphic transformations

| Rule | What it does | Key parameters |
|------|-------------|----------------|
| `imgDarkBright` | Adjusts brightness via gamma correction | `brightness_multiplier`, `gamma`, `method` |
| `imgResolution` | Downscales then upscales (simulates low-res camera) | `size_ratio` (0–1) |
| `imgMotionBlur` | Applies directional motion blur | `kernel_size`, `orientation` (degrees) |
| `imgStretch` | Scales image dimensions | `height_ratio`, `width_ratio` |
| `imgGrayscale` | Converts to greyscale | — |
| `imgColorWheel` | Rotates hues uniformly | `hue_rotation` (degrees) |

Default configurations used by the pipeline are defined in `scripts/run_pipeline.sh`.

### 2. Cone colour transformation tests

Requires: Ground truth labels + segmentation masks

Transformations:
- Yellow → Blue
- Blue → Yellow
- Yellow → Orange
- Blue → Orange

```bash
bash scripts/run_all_cone_color_transformations.sh --dataset fsoco_sample
```

### 3. Cone label removal tests

Requires: Segmented images (from histogram backprojection) + ground truth labels

Modes:
- `avg`: Replace labels with average background color
- `white`: Replace labels with white
- `black`: Replace labels with black

```bash
bash scripts/run_cone_label_transformations.sh --dataset fsoco_sample
```

---

## Analysis

### Per-Class COCO Evaluation

Analyze performance broken down by cone class (yellow, blue, orange):

```bash
python analysis/coco_eval_per_class.py \
    --gt data/base_images/fsoco_sample/annotations_coco.json \
    --dt results/yolov5/original/detections_coco.json \
    --stats-out analysis/results/yolov5/per_class_stats.json
```

## Brightness Threshold Search (RQ2)

Find the optimal brightness level where model accuracy degrades to a target threshold:

```bash
# Quick start (sample dataset)
bash scripts/run_brightness_tuning.sh --model yolov5

# Test dataset with custom target AP
bash scripts/run_brightness_tuning.sh --dataset test --model yolov8 --target-ap 0.7

# Direct Python usage (more options)
python utils/imgDarkBright_tuning/tune_brightness.py \
    --dataset fsoco_sample \
    --model yolov5 \
    --target_ap 0.8
```

**What it does:**
- Calculates brightness metrics (RMS contrast, luminosity, mean shift variance)
- Uses search algorithm to find brightness multiplier where AP reaches target
- Saves optimal configuration and brightness scores to JSON

**Requirements:**
- Original model detections must exist: `results/<model>/original/`
- Ground truth labels required
- Run main pipeline first: `bash scripts/run_pipeline.sh --model v5`

**Output:** `data/transformed_images/<dataset>_tuned_brightness/results.json`

**Documentation:** See `utils/imgDarkBright_tuning/README.md` for detailed usage

---

## Project structure

```
├── data/
│   └── base_images/fsoco_sample/   ← bundled sample images
├── models/
│   ├── yolov5/                     ← infer_v5_kfold.py + weights
│   ├── yolov8/                     ← infer_v8.py + weights
│   └── yolov12/                    ← infer_v12.py + weights
├── utils/
│   ├── matias_metamorphic.py       ← general transformation entry point
│   ├── metamorphic_rules/          ← rule implementations
│   ├── cone_colour_class_transformation.py  ← cone color transformations
│   ├── cone_label_transformation_bb.py      ← label removal transformations
│   ├── extract_cone_regions_gt.py  ← histogram backprojection segmentation
│   └── imgDarkBright_tuning/       ← brightness threshold search (RQ2)
│       ├── tune_brightness.py      ← main brightness tuning script
│       ├── search_brightness.py    ← search algorithm
│       └── mean_shift.py           ← brightness variance calculation
├── analysis/
│   ├── coco_eval.py                ← COCO metric evaluation
│   ├── coco_eval_per_class.py      ← per-class COCO evaluation
│   ├── utils/format_detections.py  ← formats raw .txt detections to COCO JSON
│   └── scripts/
│       ├── run_all_analysis_matias_meta.sh  ← metamorphic analysis
│       └── run_analysis_cone_color.sh       ← cone color analysis
└── scripts/
    ├── run_pipeline.sh                      ← main end-to-end pipeline (start here)
    ├── run_all_matias_transformations.sh    ← general transformations
    ├── run_all_cone_color_transformations.sh ← cone color transformations
    ├── run_cone_label_transformations.sh    ← label removal transformations
    ├── run_gt_histogram_backprojection.sh   ← create segmentation masks
    └── run_brightness_tuning.sh             ← brightness threshold search (RQ2)
```

---

## Requirements

- Python 3.9+
- CUDA-capable GPU recommended (CPU inference works but is slow)
- See `requirements.txt` for full dependency list

Notable dependencies: `torch`, `ultralytics`, `yolov5`, `opencv-python`, `pycocotools`

---

## Using your own dataset

To run experiments on your own dataset:

1. **Prepare your dataset structure:**
   ```
   data/base_images/<your_dataset>/
   ├── images/              ← raw images
   ├── labels/              ← ground truth labels (optional, for cone color/label tests)
   └── annotations_coco.json  ← COCO format annotations (optional)
   ```

2. **Run the pipeline:**
   ```bash
   bash scripts/run_pipeline.sh --dataset <your_dataset>
   ```

3. **For cone color/label transformations, first create segmentation masks:**
   ```bash
   bash scripts/run_gt_histogram_backprojection.sh --dataset <your_dataset>
   bash scripts/run_all_cone_color_transformations.sh --dataset <your_dataset>
   bash scripts/run_cone_label_transformations.sh --dataset <your_dataset>
   ```

---

## Research questions

This framework supports the following research questions:

RQ1: Do metamorphic transformations reveal model-specific robustness disparities between YOLOv5, v8, and v12 that are not reflected by their rankings on standard benchmarks like COCO? 

RQ1.a: If we look at a venn diagram of images that fail, do we find different failures than original images, or that of classic transformations like brightness or blur?

RQ1.b: Can we establish a clear pattern on when to use one model over another? 

RQ1.c: Do robustness disparities identified via metamorphic testing on held-out test data also show when the same transformations are applied to the model's training data, and if so, to what extent?

RQ2: Can we optimise tuning the brightness metamorphic test for the strongest signal?

RQ2.a: Can a search algorithm find an operational range for models with things like brightness compared to a brute force approach?

Note: Brute force approach involves linearly testing brightness values until threshold is met. 

RQ2.b: Among different methods of brightness scoring, which most reliably predicts the degradation threshold identified in RQ2, and does the method remain consistent across all models?

RQ2.c: How does the accuracy distribution of the original image versus the transformed brightness images compare? 

RQ3: How can we minimize the amount of test runs?

RQ3.a: If we run the metamorphic tests on 10% of the test data, do we gain sufficient insight in comparison to running on the entire dataset? 

RQ3.b: Do any of the metamorphic tests subsume each other? 


---

