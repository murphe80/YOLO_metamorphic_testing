#!/bin/bash
# ============================================
# run_all_cone_color_transformations.sh
#
# Apply cone color metamorphic transformations (yellow↔blue, etc.)
# Requires ground truth labels and segmentation masks.
#
# Usage:
#   bash scripts/run_all_cone_color_transformations.sh [--dataset fsoco_sample|test]
#
# Outputs: data/transformed_images/<dataset>_cone_color/<transformation>/
# ============================================

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Load model registry
source utils/model_registry.sh

# ============================================
# Parse arguments
# ============================================
DATASET="fsoco_sample"  # Default to sample dataset

for arg in "$@"; do
    case $arg in
        --dataset) DATASET="${2:-fsoco_sample}"; shift ;;
        --dataset=*) DATASET="${arg#*=}" ;;
    esac
done

# ============================================
# Configuration
# ============================================

# Set paths based on dataset
if [ "$DATASET" = "fsoco_sample" ]; then
    FSOCO_DIR="data/base_images/fsoco_sample"
    IMAGES_DIR="$FSOCO_DIR/segmented_images"  # Already segmented for sample
    LABELS_DIR="$FSOCO_DIR/bounding_boxes"
    SEGMENTED_DIR="$FSOCO_DIR/segmented_images"
    OUTPUT_BASE_DIR="data/transformed_images/fsoco_sample_cone_color"
elif [ "$DATASET" = "test" ]; then
    FSOCO_DIR="data/base_images/test"
    IMAGES_DIR="$FSOCO_DIR/images"
    LABELS_DIR="$FSOCO_DIR/labels"
    SEGMENTED_DIR="$FSOCO_DIR/segmented_images/segmented_images_good"
    OUTPUT_BASE_DIR="data/transformed_images/fsoco_test_cone_color"
else
    FSOCO_DIR="data/base_images/$DATASET"
    IMAGES_DIR="$FSOCO_DIR/images"
    LABELS_DIR="$FSOCO_DIR/labels"
    SEGMENTED_DIR="$FSOCO_DIR/segmented_images/segmented_images_good"
    OUTPUT_BASE_DIR="data/transformed_images/${DATASET}_cone_color"
fi

# Histogram backprojection configuration
HIST_SAMPLES_DIR="utils/hist_backprojection"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "======================================"
echo "Cone Color Transformation Pipeline"
echo "======================================"
echo "Dataset: $DATASET"
echo "Images: $IMAGES_DIR"
echo "Labels: $LABELS_DIR"
echo "Segmentation: $SEGMENTED_DIR"
echo "Output base: $OUTPUT_BASE_DIR"
echo ""

# Validate required directories
if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found: $IMAGES_DIR"
    exit 1
fi

if [ ! -d "$LABELS_DIR" ]; then
    echo "Error: Labels directory not found: $LABELS_DIR"
    echo "Cone color transformations require ground truth labels."
    exit 1
fi

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
    "blue_to_orange"
    "swap_sides"
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
echo ""

# COCO eval: run analysis on cone color detection results if any exist
RESULTS_BASE="results"
CONE_COLOR_RESULTS=()
for model in $(get_model_names); do
    if [ -d "$RESULTS_BASE/$model/cone_color" ] && [ -d "$RESULTS_BASE/$model/cone_color/original" ]; then
        CONE_COLOR_RESULTS+=("$RESULTS_BASE/$model/cone_color")
    fi
done
if [ ${#CONE_COLOR_RESULTS[@]} -gt 0 ]; then
    echo "======================================"
    echo "Running COCO evaluation on cone color results"
    echo "======================================"
    source "$(dirname "$0")/../analysis/scripts/run_analysis_cone_color.sh"
    echo ""
    echo "COCO metrics saved under analysis/results/<model>/cone_color/"
else
    echo "No cone color detection results found under $RESULTS_BASE/<model>/cone_color."
    echo "Run YOLO inference on the transformed images, then run COCO eval with:"
    echo "  bash analysis/scripts/run_analysis_cone_color.sh"
    echo ""
fi


