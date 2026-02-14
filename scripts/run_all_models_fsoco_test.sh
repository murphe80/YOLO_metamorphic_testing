#!/bin/bash
# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# ============================================
# Configuration Parameters
# ============================================

# Number of images per (model, transform) to save visualizations for validation
NUM_VIZ=20

# Input directories
BASE_IMAGES_DIR="data/base_images/test/images"
TRANSFORMED_IMAGES_DIR="data/transformed_images/fsoco_test"

# Model weights
YOLOV5_WEIGHTS="./models/yolov5/yolov5_param_kfold.pt"
YOLOV8_WEIGHTS="./models/yolov8/yolov8_param_kfold.pt"
YOLOV12_WEIGHTS="./models/yolov12/yolov12_param_kfold.pt"

# Model inference scripts
YOLOV5_SCRIPT="models/yolov5/infer_v5_kfold.py"
YOLOV8_SCRIPT="models/yolov8/infer_v8.py"
YOLOV12_SCRIPT="models/yolov12/infer_v12.py"

# Results directories
YOLOV5_RESULTS_DIR="results/yolov5"
YOLOV8_RESULTS_DIR="results/yolov8"
YOLOV12_RESULTS_DIR="results/yolov12"

# ============================================
# **Assumes all metamorphic test images have been generated**
# ============================================

# Run original images on v5
if [ -d "$YOLOV5_RESULTS_DIR/original" ] && [ -n "$(ls -A "$YOLOV5_RESULTS_DIR/original" 2>/dev/null)" ]; then
    echo "✓ YOLOv5 original results already exist, skipping..."
    RESULT_COUNT=$(find "$YOLOV5_RESULTS_DIR/original" -type f 2>/dev/null | wc -l)
    echo "  Found $RESULT_COUNT result files"
else
    echo "Running YOLOv5 on original images..."
    img_count=0
    for image in "$BASE_IMAGES_DIR"/*; do
        [ ! -f "$image" ] && continue
        save_viz=""
        [ $img_count -lt $NUM_VIZ ] && save_viz="--save-viz"
        python "$YOLOV5_SCRIPT" \
            --weights "$YOLOV5_WEIGHTS" \
            --image "$image" \
            --output-subdir original \
            $save_viz
        ((img_count++)) || true
    done
fi

# Run transformed images on v5
for dir in "$TRANSFORMED_IMAGES_DIR"/*; do
    [ ! -d "$dir" ] && continue
    subdir=$(basename "$dir")
    img_count=0
    for image in "$dir"/*; do
        [ ! -f "$image" ] && continue
        save_viz=""
        [ $img_count -lt $NUM_VIZ ] && save_viz="--save-viz"
        python "$YOLOV5_SCRIPT" \
            --weights "$YOLOV5_WEIGHTS" \
            --image "$image" \
            --output-subdir "$subdir" \
            $save_viz
        ((img_count++)) || true
    done
done

# Run original images on v8
if [ -d "$YOLOV8_RESULTS_DIR/original" ] && [ -n "$(ls -A "$YOLOV8_RESULTS_DIR/original" 2>/dev/null)" ]; then
    echo "✓ YOLOv8 original results already exist, skipping..."
    RESULT_COUNT=$(find "$YOLOV8_RESULTS_DIR/original" -type f 2>/dev/null | wc -l)
    echo "  Found $RESULT_COUNT result files"
else
    echo "Running YOLOv8 on original images..."
    img_count=0
    for image in "$BASE_IMAGES_DIR"/*; do
        [ ! -f "$image" ] && continue
        save_viz=""
        [ $img_count -lt $NUM_VIZ ] && save_viz="--save-viz"
        python "$YOLOV8_SCRIPT" \
            --weights "$YOLOV8_WEIGHTS" \
            --image "$image" \
            --output-subdir original \
            $save_viz
        ((img_count++)) || true
    done
fi

# Run transformed images on v8
for dir in "$TRANSFORMED_IMAGES_DIR"/*; do
    [ ! -d "$dir" ] && continue
    subdir=$(basename "$dir")
    img_count=0
    for image in "$dir"/*; do
        [ ! -f "$image" ] && continue
        save_viz=""
        [ $img_count -lt $NUM_VIZ ] && save_viz="--save-viz"
        python "$YOLOV8_SCRIPT" \
            --weights "$YOLOV8_WEIGHTS" \
            --image "$image" \
            --output-subdir "$subdir" \
            $save_viz
        ((img_count++)) || true
    done
done

# Run original images on v12
if [ -d "$YOLOV12_RESULTS_DIR/original" ] && [ -n "$(ls -A "$YOLOV12_RESULTS_DIR/original" 2>/dev/null)" ]; then
    echo "✓ YOLOv12 original results already exist, skipping..."
    RESULT_COUNT=$(find "$YOLOV12_RESULTS_DIR/original" -type f 2>/dev/null | wc -l)
    echo "  Found $RESULT_COUNT result files"
else
    echo "Running YOLOv12 on original images..."
    img_count=0
    for image in "$BASE_IMAGES_DIR"/*; do
        [ ! -f "$image" ] && continue
        save_viz=""
        [ $img_count -lt $NUM_VIZ ] && save_viz="--save-viz"
        python "$YOLOV12_SCRIPT" \
            --weights "$YOLOV12_WEIGHTS" \
            --image "$image" \
            --output-subdir original \
            $save_viz
        ((img_count++)) || true
    done
fi

# Run transformed images on v12
for dir in "$TRANSFORMED_IMAGES_DIR"/*; do
    [ ! -d "$dir" ] && continue
    subdir=$(basename "$dir")
    img_count=0
    for image in "$dir"/*; do
        [ ! -f "$image" ] && continue
        save_viz=""
        [ $img_count -lt $NUM_VIZ ] && save_viz="--save-viz"
        python "$YOLOV12_SCRIPT" \
            --weights "$YOLOV12_WEIGHTS" \
            --image "$image" \
            --output-subdir "$subdir" \
            $save_viz
        ((img_count++)) || true
    done
done
