# YOLO Metamorphic Testing

A metamorphic testing framework for evaluating the robustness of YOLO-based cone detectors. It systematically applies image transformations (blur, brightness, resolution changes, colour shifts, etc.) and checks whether the model's detections remain consistent.

**Supported models:** YOLOv5 · YOLOv8 · YOLOv12

---

## Quick start

```bash
git clone https://github.com/murphe80/YOLO_metamorphic_testing_sample.git
cd YOLO_metamorphic_testing_sample

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the full pipeline on the bundled sample images (10 images, all 3 models)
bash scripts/run_sample_pipeline.sh
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

### Full sample pipeline (one command)

```bash
bash scripts/run_sample_pipeline.sh
```

Options:

| Flag | Description |
|------|-------------|
| `--model v5\|v8\|v12\|all` | Run inference for one model only (default: `all`) |
| `--skip-transform` | Skip Stage 1 (re-use existing transformed images) |

### Running stages individually

**Stage 1 — Generate transformed images**

```bash
bash scripts/run_all_matias_transformations.sh
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
bash analysis/scripts/run_all_analysis_matias_meta.sh
```

Output `stats.json` files contain COCO-standard AP and AR metrics at multiple IoU thresholds.

---

## Transformation rules

| Rule | What it does | Key parameters |
|------|-------------|----------------|
| `imgDarkBright` | Adjusts brightness via gamma correction | `brightness_multiplier`, `gamma`, `method` |
| `imgResolution` | Downscales then upscales (simulates low-res camera) | `size_ratio` (0–1) |
| `imgMotionBlur` | Applies directional motion blur | `kernel_size`, `orientation` (degrees) |
| `imgStretch` | Scales image dimensions | `height_ratio`, `width_ratio` |
| `imgGrayscale` | Converts to greyscale | — |
| `imgColorWheel` | Rotates hues uniformly | `hue_rotation` (degrees) |

Default configurations used by the sample pipeline are defined at the top of `scripts/run_sample_pipeline.sh`.

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
│   ├── matias_metamorphic.py       ← transformation entry point
│   └── metamorphic_rules/          ← rule implementations
├── analysis/
│   ├── coco_eval.py                ← COCO metric evaluation
│   ├── utils/format_detections.py  ← formats raw .txt detections to COCO JSON
│   └── scripts/
│       └── run_all_analysis_matias_meta.sh
└── scripts/
    ├── run_sample_pipeline.sh      ← end-to-end sample pipeline (start here)
    └── run_all_matias_transformations.sh
```

---

## Requirements

- Python 3.9+
- CUDA-capable GPU recommended (CPU inference works but is slow)
- See `requirements.txt` for full dependency list

Notable dependencies: `torch`, `ultralytics`, `yolov5`, `opencv-python`, `pycocotools`

---

## Using your own images

Point any stage at a different image directory by editing the `IMAGES_DIR` variable at the top of the relevant script, or by passing `--base_image` / `--image` arguments directly to the Python scripts.
