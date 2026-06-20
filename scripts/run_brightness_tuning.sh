#!/bin/bash
# ============================================
# run_brightness_tuning.sh
#
# Run brightness threshold search for YOLO models (RQ2).
# Finds the brightness multiplier where model accuracy degrades to target threshold.
#
# Usage:
#   bash scripts/run_brightness_tuning.sh [OPTIONS]
#
# Options:
#   --dataset <name>      Dataset to use (default: fsoco_sample)
#   --model <v5|v8|v12>   YOLO model (required)
#   --target-ap <float>   Target AP threshold (default: 0.8)
#
# Examples:
#   bash scripts/run_brightness_tuning.sh --model yolov5
#   bash scripts/run_brightness_tuning.sh --dataset test --model yolov8 --target-ap 0.7
#
# Requirements:
#   - Original detections must exist: results/<model>/original/
#   - Ground truth labels required
#   - Run inference first if detections don't exist
# ============================================

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Load model registry
source utils/model_registry.sh

# ─── Parse arguments ──────────────────────────────────────────────────────────
DATASET="fsoco_sample"
MODEL=""
TARGET_AP="0.8"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --dataset=*) DATASET="${1#*=}"; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --model=*) MODEL="${1#*=}"; shift ;;
        --target-ap) TARGET_AP="$2"; shift 2 ;;
        --target-ap=*) TARGET_AP="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Validate arguments ───────────────────────────────────────────────────────
if [ -z "$MODEL" ]; then
    echo "Error: --model is required"
    VALID_VERSIONS=$(get_model_versions)
    echo "Usage: bash scripts/run_brightness_tuning.sh --model <$VALID_VERSIONS>"
    exit 1
fi

# Validate model version
if ! validate_model_version "$MODEL"; then
    echo "Error: Invalid model: $MODEL"
    echo "Choose from: $(get_model_versions)"
    exit 1
fi

# Convert version to model name (e.g., v5 → yolov5)
MODEL=$(get_model_by_version "$MODEL")

# ─── Check prerequisites ──────────────────────────────────────────────────────
DETECTIONS_DIR="results/$MODEL/original"

if [ ! -d "$DETECTIONS_DIR" ] || [ -z "$(ls -A "$DETECTIONS_DIR" 2>/dev/null)" ]; then
    echo "Error: No detections found in $DETECTIONS_DIR"
    echo ""
    echo "Please run inference first:"
    if [ "$DATASET" = "fsoco_sample" ]; then
        echo "  bash scripts/run_pipeline.sh --model ${MODEL#yolo} --skip-transform"
    else
        echo "  bash scripts/run_pipeline.sh --dataset $DATASET --model ${MODEL#yolo}"
    fi
    exit 1
fi

# ─── Run brightness tuning ────────────────────────────────────────────────────
echo "======================================"
echo "Brightness Threshold Search (RQ2)"
echo "======================================"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Target AP: $TARGET_AP"
echo ""

python utils/imgDarkBright_tuning/tune_brightness.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --target_ap "$TARGET_AP"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Brightness tuning complete!"
    echo "======================================"
    echo "Results: data/transformed_images/${DATASET}_tuned_brightness/results.json"
else
    echo ""
    echo "Error: Brightness tuning failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
