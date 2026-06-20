#!/bin/bash
# =============================================================================
# run_label_pipeline.sh
#
# End-to-end metamorphic testing pipeline for YOLO models using label transformations.
# Tests whether YOLO models rely on cone labels vs. cone body shape by removing/changing labels.
#
# Runs all three stages in sequence:
#   1. Generate label-transformed images (remove cone labels: avg, white, black)
#   2. Run inference with YOLOv5, YOLOv8, and YOLOv12
#   3. Evaluate: compare original detections vs transformed detections
#
# Prerequisites:
#   - Segmented images must exist (run scripts/run_gt_histogram_backprojection.sh first)
#   - Ground truth labels must exist in bounding_boxes directory
#
# Usage:
#   bash scripts/run_label_pipeline.sh [OPTIONS]
#
# Options:
#   --dataset <name>       Dataset to use (default: fsoco_sample)
#                          Options: fsoco_sample, test, or custom path
#   --model <v5|v8|v12|all> Which model(s) to run (default: all)
#   --skip-transform       Skip Stage 1 (re-use existing transformed images)
#   --mode <avg|white|black|all> Transformation mode (default: all)
#
# Examples:
#   bash scripts/run_label_pipeline.sh
#   bash scripts/run_label_pipeline.sh --dataset test --model v8
#   bash scripts/run_label_pipeline.sh --dataset fsoco_sample --skip-transform
#   bash scripts/run_label_pipeline.sh --mode avg --model v5
#
# Outputs are written to:
#   data/transformed_images/<dataset>_label_transforms/<mode>/
#   results/<model>/original/
#   results/<model>/labelRemove_<mode>/
#   analysis/results/<model>/labelRemove_<mode>/
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

# Load model registry
source utils/model_registry.sh

# ─── Parse arguments ──────────────────────────────────────────────────────────
DATASET="fsoco_sample"  # Default to sample dataset
MODEL_ARG="all"
SKIP_TRANSFORM=false
MODE_ARG="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --dataset=*) DATASET="${1#*=}"; shift ;;
        --model) MODEL_ARG="$2"; shift 2 ;;
        --model=*) MODEL_ARG="${1#*=}"; shift ;;
        --mode) MODE_ARG="$2"; shift 2 ;;
        --mode=*) MODE_ARG="${1#*=}"; shift ;;
        --skip-transform) SKIP_TRANSFORM=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Configure paths based on dataset ─────────────────────────────────────────
if [ "$DATASET" = "fsoco_sample" ]; then
    IMAGES_DIR="data/base_images/fsoco_sample/images"
    SEGMENTED_DIR="data/base_images/fsoco_sample/segmented_images"
    LABELS_DIR="data/base_images/fsoco_sample/bounding_boxes"
    TRANSFORMED_BASE="data/transformed_images/fsoco_sample_label_transforms"
elif [ "$DATASET" = "test" ]; then
    IMAGES_DIR="data/base_images/test/images"
    SEGMENTED_DIR="data/base_images/test/segmented_images"
    LABELS_DIR="data/base_images/test/labels"
    TRANSFORMED_BASE="data/transformed_images/fsoco_test_label_transforms"
else
    IMAGES_DIR="data/base_images/$DATASET/images"
    SEGMENTED_DIR="data/base_images/$DATASET/segmented_images"
    LABELS_DIR="data/base_images/$DATASET/labels"
    TRANSFORMED_BASE="data/transformed_images/${DATASET}_label_transforms"
fi

RESULTS_BASE="results"

# ─── Label transformation modes ───────────────────────────────────────────────
# avg: Fill labels with average cone color (preserves appearance)
# white: Fill labels with white color (simulates white labels)
# black: Fill labels with black color (simulates black labels)

if [ "$MODE_ARG" = "all" ]; then
    TRANSFORM_MODES=(avg white black)
else
    TRANSFORM_MODES=("$MODE_ARG")
fi

LABEL_TRANSFORM_SCRIPT="utils/cone_label_transformation_bb.py"

# ─── Helper: run inference for one model on one image directory ───────────────
run_inference_dir() {
    local script="$1"
    local weights="$2"
    local image_dir="$3"
    local output_subdir="$4"

    for image in "$image_dir"/*; do
        [ -f "$image" ] || continue
        python "$script" \
            --weights "$weights" \
            --image "$image" \
            --output-subdir "$output_subdir"
    done
}

# ─── Validate input directories ───────────────────────────────────────────────
if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found: $IMAGES_DIR"
    echo "Please ensure the dataset exists."
    exit 1
fi

if [ ! -d "$SEGMENTED_DIR" ]; then
    echo "Error: Segmented images directory not found: $SEGMENTED_DIR"
    echo "Label transformations require segmented images from histogram backprojection."
    echo "Please run: bash scripts/run_gt_histogram_backprojection.sh --dataset $DATASET"
    exit 1
fi

if [ ! -d "$LABELS_DIR" ]; then
    echo "Error: Labels directory not found: $LABELS_DIR"
    echo "Label transformations require ground truth labels."
    exit 1
fi

if [ ! -f "$LABEL_TRANSFORM_SCRIPT" ]; then
    echo "Error: Label transformation script not found: $LABEL_TRANSFORM_SCRIPT"
    exit 1
fi

# =============================================================================
# Stage 1 — Generate label-transformed images
# =============================================================================
if [ "$SKIP_TRANSFORM" = false ]; then
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Stage 1: Generating label-transformed images║"
    echo "╚══════════════════════════════════════════════╝"
    echo "Dataset: $DATASET"
    echo "Segmented images: $SEGMENTED_DIR"
    echo "Ground truth labels: $LABELS_DIR"
    echo ""

    for mode in "${TRANSFORM_MODES[@]}"; do
        output_dir="$TRANSFORMED_BASE/$mode"
        mkdir -p "$output_dir"

        echo ""
        echo "── Label removal: $mode mode ──"
        python3 "$LABEL_TRANSFORM_SCRIPT" \
            --stacked_images_dir "$SEGMENTED_DIR" \
            --labels_dir "$LABELS_DIR" \
            --output_dir "$output_dir" \
            --transform_mode "$mode"

        img_count=$(find "$output_dir" -type f | wc -l | xargs)
        echo "  ✓ $img_count images → $output_dir"
    done
else
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Stage 1: Skipped (--skip-transform)         ║"
    echo "╚══════════════════════════════════════════════╝"
fi

# =============================================================================
# Stage 2 — Run inference
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Stage 2: Running inference                  ║"
echo "╚══════════════════════════════════════════════╝"

run_model() {
    local model_name="$1"

    local script=$(get_model_field "$model_name" "inference_script")
    local weights=$(get_model_field "$model_name" "weights")
    local display_name=$(get_model_field "$model_name" "display_name")

    echo ""
    echo "── $display_name — original images ──"
    run_inference_dir "$script" "$weights" "$IMAGES_DIR" "original"

    for mode in "${TRANSFORM_MODES[@]}"; do
        transformed_dir="$TRANSFORMED_BASE/$mode"
        if [ -d "$transformed_dir" ]; then
            # Use naming convention: labelRemove_<mode> to match analysis expectations
            rule_name="labelRemove_$mode"
            echo "── $display_name — $rule_name ──"
            run_inference_dir "$script" "$weights" "$transformed_dir" "$rule_name"
        else
            echo "Warning: Transformed directory not found: $transformed_dir"
        fi
    done
}

# Determine which models to run
if [ "$MODEL_ARG" = "all" ]; then
    # Run all enabled models from registry
    MODEL_LIST=$(get_model_names)
else
    # Map version to model name (e.g., v5 → yolov5)
    MODEL_NAME=$(get_model_by_version "$MODEL_ARG")
    if [ -z "$MODEL_NAME" ]; then
        echo "Error: Invalid model version '$MODEL_ARG'"
        echo "Valid versions: $(get_model_versions)"
        exit 1
    fi
    MODEL_LIST="$MODEL_NAME"
fi

# Run inference for each model
for model in $MODEL_LIST; do
    run_model "$model"
done

# =============================================================================
# Stage 3 — Evaluate (metamorphic comparison: original vs label-transformed)
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Stage 3: Running metamorphic analysis       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Run the standard analysis script which handles all transformations
bash analysis/scripts/run_all_analysis_matias_meta.sh --dataset "$DATASET"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Label transformation pipeline complete      ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Dataset: $DATASET"
echo "Transformation modes: ${TRANSFORM_MODES[*]}"
echo ""
echo "Results locations:"
for mode in "${TRANSFORM_MODES[@]}"; do
    echo "  - labelRemove_$mode:"
    echo "    • analysis/results/<model>/labelRemove_$mode/original_vs_transformed/stats.json"
    if [ -f "data/base_images/$DATASET/annotations_coco.json" ]; then
        echo "    • analysis/results/<model>/labelRemove_$mode/transformed_vs_gt/stats.json"
    fi
done
