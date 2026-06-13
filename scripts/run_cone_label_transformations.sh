#!/bin/bash
# ============================================
# run_cone_label_transformations.sh
#
# Apply cone label removal metamorphic transformations.
# Requires segmented images (from histogram backprojection) and ground truth labels.
#
# Usage:
#   bash scripts/run_cone_label_transformations.sh [--dataset fsoco_sample|test]
#
# Outputs: data/transformed_images/<dataset>_label_transforms/<mode>/
# ============================================

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

set -e  # fail fast
set -u  # undefined vars are errors

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
    STACKED_IMAGES_DIR="data/base_images/fsoco_sample/segmented_images"
    LABELS_DIR="data/base_images/fsoco_sample/bounding_boxes"
    OUTPUT_DIR="data/transformed_images/fsoco_sample_label_transforms"
elif [ "$DATASET" = "test" ]; then
    STACKED_IMAGES_DIR="data/base_images/test/segmented_images"
    LABELS_DIR="data/base_images/test/labels"
    OUTPUT_DIR="data/transformed_images/fsoco_test_label_transforms"
else
    STACKED_IMAGES_DIR="data/base_images/$DATASET/segmented_images"
    LABELS_DIR="data/base_images/$DATASET/labels"
    OUTPUT_DIR="data/transformed_images/${DATASET}_label_transforms"
fi

SCRIPT_PATH="utils/cone_label_transformation_bb.py"

# ============================================
# Validate inputs
# ============================================

if [[ ! -d "$STACKED_IMAGES_DIR" ]]; then
    echo "ERROR: Stacked images directory not found: $STACKED_IMAGES_DIR"
    echo "Run histogram backprojection first:"
    echo "  bash scripts/run_gt_histogram_backprojection.sh --dataset $DATASET"
    exit 1
fi

if [[ ! -d "$LABELS_DIR" ]]; then
    echo "ERROR: Labels directory not found: $LABELS_DIR"
    echo "Label transformations require ground truth labels."
    exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "ERROR: Script not found: $SCRIPT_PATH"
    exit 1
fi

# ============================================
# Run label transformations
# ============================================

echo "======================================"
echo "Cone Label Removal Transformations"
echo "======================================"
echo "Dataset: $DATASET"
echo "Stacked images: $STACKED_IMAGES_DIR"
echo "Labels: $LABELS_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Run for all transformation modes
for MODE in avg white black; do
    MODE_OUTPUT_DIR="$OUTPUT_DIR/$MODE"
    mkdir -p "$MODE_OUTPUT_DIR"

    echo "────────────────────────────────────────"
    echo "Running label removal: $MODE mode"
    echo "────────────────────────────────────────"

    python3 "$SCRIPT_PATH" \
        --stacked_images_dir "$STACKED_IMAGES_DIR" \
        --labels_dir "$LABELS_DIR" \
        --output_dir "$MODE_OUTPUT_DIR" \
        --transform_mode "$MODE"

    echo "✓ Completed $MODE mode"
    echo ""
done

echo "======================================"
echo "All label transformations complete"
echo "======================================"
echo "Results saved under: $OUTPUT_DIR"
