#!/bin/bash

# Ground Truth based Histogram Backprojection Pipeline
# Uses existing JSON annotation files instead of running YOLO detection

set -e

echo "=== Ground Truth Histogram Backprojection Pipeline ==="
echo

# Configuration
INPUT_IMAGES_DIR="data/base_images/test/images"
LABELS_DIR="data/base_images/test/labels"
HIST_SAMPLES_DIR="utils/hist_backprojection"
OUTPUT_DIR="data/base_images/segmented_fsoco_test_gt"

# Check if input directories exist
if [ ! -d "$INPUT_IMAGES_DIR" ]; then
    echo "Error: Input images directory not found: $INPUT_IMAGES_DIR"
    exit 1
fi

if [ ! -d "$LABELS_DIR" ]; then
    echo "Error: Labels directory not found: $LABELS_DIR"
    exit 1
fi

if [ ! -d "$HIST_SAMPLES_DIR" ]; then
    echo "Error: Histogram samples directory not found: $HIST_SAMPLES_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Images directory: $INPUT_IMAGES_DIR"
echo "  Labels directory: $LABELS_DIR"
echo "  Histogram samples: $HIST_SAMPLES_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo

# Count available files
IMAGE_COUNT=$(find "$INPUT_IMAGES_DIR" -name "*.jpg" -o -name "*.png" | wc -l)
LABEL_COUNT=$(find "$LABELS_DIR" -name "*.json" | wc -l)

echo "Available files:"
echo "  Images: $IMAGE_COUNT"
echo "  Labels: $LABEL_COUNT"
echo

if [ $LABEL_COUNT -eq 0 ]; then
    echo "Error: No JSON label files found in $LABELS_DIR"
    exit 1
fi

# Check for histogram sample files
echo "Checking histogram sample files..."
BLUE_SAMPLES=$(find "$HIST_SAMPLES_DIR" -name "*blue_cones_samples*" | head -1)
YELLOW_SAMPLES=$(find "$HIST_SAMPLES_DIR" -name "*yellow_cones_samples*" | head -1)
ORANGE_SAMPLES=$(find "$HIST_SAMPLES_DIR" -name "*orange_cones_samples*" | head -1)

if [ -z "$BLUE_SAMPLES" ]; then
    echo "Warning: No blue cone samples found"
else
    echo "  Blue samples: $(basename "$BLUE_SAMPLES")"
fi

if [ -z "$YELLOW_SAMPLES" ]; then
    echo "Warning: No yellow cone samples found"
else
    echo "  Yellow samples: $(basename "$YELLOW_SAMPLES")"
fi

if [ -z "$ORANGE_SAMPLES" ]; then
    echo "Warning: No orange cone samples found"
else
    echo "  Orange samples: $(basename "$ORANGE_SAMPLES")"
fi

echo

# Run the ground truth histogram backprojection
echo "Running ground truth histogram backprojection..."
echo "This will process all images that have corresponding JSON label files."
echo

python utils/extract_cone_regions_gt.py \
    --images_dir "$INPUT_IMAGES_DIR" \
    --labels_dir "$LABELS_DIR" \
    --hist_samples_dir "$HIST_SAMPLES_DIR" \
    --output_dir "$OUTPUT_DIR"


