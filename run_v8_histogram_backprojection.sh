#!/bin/bash

# YOLOv8 detection followed by histogram backprojection pipeline

set -e

echo "=== YOLOv8 Histogram Backprojection Pipeline ==="
echo

# Configuration
WEIGHTS="models/yolov8/yolov8_param_kfold.pt"
INPUT_IMAGES_DIR="data/base_images/test/images"
DETECTIONS_OUTPUT_DIR="results/yolov8"
HIST_SAMPLES_DIR="utils/hist_backprojection"
SEGMENTATION_OUTPUT_DIR="data/base_images/test_segmented_yolov8"

# Check if weights file exists
if [ ! -f "$WEIGHTS" ]; then
    echo "Error: YOLOv8 weights file not found: $WEIGHTS"
    echo "Available YOLOv8 weights files:"
    find models/yolov8/ -name "*.pt" 2>/dev/null || echo "No .pt files found"
    exit 1
fi

# Check if input directory exists
if [ ! -d "$INPUT_IMAGES_DIR" ]; then
    echo "Error: Input images directory not found: $INPUT_IMAGES_DIR"
    echo "Available base image directories:"
    find data/base_images/ -type d -maxdepth 1 2>/dev/null | head -10
    exit 1
fi

# Create output directories
mkdir -p "$DETECTIONS_OUTPUT_DIR"
mkdir -p "$SEGMENTATION_OUTPUT_DIR"

echo "Step 1: Running YOLOv8 detection on test images..."
echo "Using weights: $WEIGHTS"
echo "Input directory: $INPUT_IMAGES_DIR"
echo

# Get first few images for testing
TEST_IMAGES=$(find "$INPUT_IMAGES_DIR" -name "*.jpg" -o -name "*.png" | head -3)

if [ -z "$TEST_IMAGES" ]; then
    echo "Error: No images found in $INPUT_IMAGES_DIR"
    exit 1
fi

echo "Processing images:"
echo "$TEST_IMAGES"
echo

for image in $TEST_IMAGES; do
    echo "Processing: $(basename $image)"
    python models/yolov8/infer_v8.py \
        --weights "$WEIGHTS" \
        --image "$image" \
        --conf_thres 0.25 \
        --iou_thres 0.45 \
        --device cuda:0
    echo
done

echo "Step 2: Running histogram backprojection on detected regions..."
echo

# Check if any detection files were created
DETECTION_FILES=$(find "$DETECTIONS_OUTPUT_DIR" -name "*.txt" 2>/dev/null | head -3)
if [ -z "$DETECTION_FILES" ]; then
    echo "Warning: No detection .txt files found in $DETECTIONS_OUTPUT_DIR"
    echo "The YOLOv8 detection step may have failed or produced no detections."
    exit 1
fi

echo "Found detection files:"
echo "$DETECTION_FILES"
echo

# Run histogram backprojection
python utils/histogram_backproj_yolo.py \
    --images_dir "$INPUT_IMAGES_DIR" \
    --detections_dir "$DETECTIONS_OUTPUT_DIR" \
    --hist_samples_dir "$HIST_SAMPLES_DIR" \
    --output_dir "$SEGMENTATION_OUTPUT_DIR"

echo
echo "=== Pipeline Complete ==="
echo "Detection results: $DETECTIONS_OUTPUT_DIR/*.txt"
echo "Segmented images: $SEGMENTATION_OUTPUT_DIR/segmented_*.jpg/*.png"
echo
echo "Output contains only the segmented regions (cones extracted via histogram backprojection)"
echo
echo "To view results:"
echo "  ls -la $SEGMENTATION_OUTPUT_DIR/"