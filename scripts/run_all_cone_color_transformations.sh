#!/bin/bash

# Run all combinations of cone color transformations with segmentation masks
# This script transforms cone colors based on bounding box labels and segmentation masks

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Configuration
FSOCO_DIR="data/base_images/fsoco_test"
IMAGES_DIR="$FSOCO_DIR/images"
LABELS_DIR="$FSOCO_DIR/labels"
SEGMENTED_DIR="$FSOCO_DIR/segmented_images"
OUTPUT_BASE_DIR="data/transformed_images/cone_color_transformations"

# Histogram backprojection configuration
HIST_SAMPLES_DIR="utils/hist_backprojection"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "======================================"
echo "Cone Color Transformation Pipeline"
echo "======================================"
echo "Images: $IMAGES_DIR"
echo "Labels: $LABELS_DIR"
echo "Segmentation: $SEGMENTED_DIR"
echo "Output base: $OUTPUT_BASE_DIR"
echo ""

# Check if segmentation directory exists and has files
if [ ! -d "$SEGMENTED_DIR" ] || [ -z "$(ls -A "$SEGMENTED_DIR" 2>/dev/null)" ]; then
    echo "======================================"
    echo "Segmentation directory not found or empty"
    echo "Running ground truth histogram backprojection first..."
    echo "======================================"
    echo ""
    
    # Create segmentation directory
    mkdir -p "$SEGMENTED_DIR"
    
    # Run ground truth histogram backprojection
    echo "Running: extract_cone_regions_gt.py"
    python utils/extract_cone_regions_gt.py \
        --images_dir "$IMAGES_DIR" \
        --labels_dir "$LABELS_DIR" \
        --hist_samples_dir "$HIST_SAMPLES_DIR" \
        --output_dir "$SEGMENTED_DIR"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Segmentation completed successfully"
        echo ""
    else
        echo ""
        echo "✗ Segmentation failed"
        exit 1
    fi
else
    echo "✓ Segmentation directory exists with files"
    SEGMENTED_COUNT=$(find "$SEGMENTED_DIR" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
    echo "  Found $SEGMENTED_COUNT segmented images"
    echo ""
fi

# Array of all transformation types
TRANSFORMATIONS=(
    "yellow_to_blue"
    "blue_to_yellow"
    "yellow_to_orange"
    "orange_to_blue"
    "blue_to_orange"
    "orange_to_yellow"
)

echo "======================================"
echo "Starting Color Transformations"
echo "======================================"
echo ""

# Run each transformation
for transformation in "${TRANSFORMATIONS[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE_DIR/$transformation"
    
    echo "======================================"
    echo "Running transformation: $transformation"
    echo "Output directory: $OUTPUT_DIR"
    echo "======================================"
    
    python utils/cone_colour_class_transformation.py \
        --images_dir "$IMAGES_DIR" \
        --labels_dir "$LABELS_DIR" \
        --segmented_dir "$SEGMENTED_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --transformation "$transformation"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $transformation"
    else
        echo "✗ Failed: $transformation"
        exit 1
    fi
    echo ""
done

echo "======================================"
echo "All transformations completed!"
echo "======================================"
echo ""
echo "Summary of output directories:"
for transformation in "${TRANSFORMATIONS[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE_DIR/$transformation"
    NUM_FILES=$(find "$OUTPUT_DIR" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
    echo "  - $transformation: $NUM_FILES files in $OUTPUT_DIR"
done


