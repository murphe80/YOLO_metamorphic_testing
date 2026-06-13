#!/bin/bash
# =============================================================================
# run_pipeline.sh
#
# End-to-end metamorphic testing pipeline for YOLO models.
# Runs all three stages in sequence:
#   1. Generate transformed images (6 metamorphic rules)
#   2. Run inference with YOLOv5, YOLOv8, and YOLOv12
#   3. Evaluate: compare original detections vs transformed detections
#
# Usage:
#   bash scripts/run_pipeline.sh [OPTIONS]
#
# Options:
#   --dataset <name>       Dataset to use (default: fsoco_sample)
#                          Options: fsoco_sample, test, or custom path
#   --model <v5|v8|v12|all> Which model(s) to run (default: all)
#   --skip-transform       Skip Stage 1 (re-use existing transformed images)
#
# Examples:
#   bash scripts/run_pipeline.sh
#   bash scripts/run_pipeline.sh --dataset test --model v8
#   bash scripts/run_pipeline.sh --dataset fsoco_sample --skip-transform
#
# Outputs are written to:
#   data/transformed_images/<dataset>/<rule>/
#   results/<model>/original/
#   results/<model>/<rule>/
#   analysis/results/<model>/<rule>/
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

# ─── Parse arguments ──────────────────────────────────────────────────────────
DATASET="fsoco_sample"  # Default to sample dataset
MODEL_ARG="all"
SKIP_TRANSFORM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --dataset=*) DATASET="${1#*=}"; shift ;;
        --model) MODEL_ARG="$2"; shift 2 ;;
        --model=*) MODEL_ARG="${1#*=}"; shift ;;
        --skip-transform) SKIP_TRANSFORM=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Configure paths based on dataset ─────────────────────────────────────────
if [ "$DATASET" = "fsoco_sample" ]; then
    IMAGES_DIR="data/base_images/fsoco_sample/images"
    TRANSFORMED_DIR="data/transformed_images/fsoco_sample"
elif [ "$DATASET" = "test" ]; then
    IMAGES_DIR="data/base_images/test/images"
    TRANSFORMED_DIR="data/transformed_images/fsoco_test"
else
    IMAGES_DIR="data/base_images/$DATASET/images"
    TRANSFORMED_DIR="data/transformed_images/$DATASET"
fi

RESULTS_BASE="results"

# ─── Transformation rules ─────────────────────────────────────────────────────
# Load transformation configurations from centralized config file
CONFIG_FILE="config/transformations.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Transformation config file not found at $CONFIG_FILE"
    exit 1
fi

# Parse transformation configs using Python
read -r -d '' PYTHON_PARSE_CONFIG << 'EOF' || true
import json
import sys
with open(sys.argv[1]) as f:
    config = json.load(f)
    for rule_name in sys.argv[2:]:
        if rule_name in config:
            print(json.dumps(config[rule_name]))
        else:
            print("{}")
EOF

# Parallel arrays (bash 3.2 on macOS does not support declare -A).
RULE_NAMES=(
    imgDarkBright
    imgResolution
    imgMotionBlur
    imgStretch
    imgGrayscale
    imgColorWheel
)

# Read configurations from JSON file (while-read: portable on bash 3.2 / macOS)
RULE_CONFIGS=()
while IFS= read -r line; do
    RULE_CONFIGS+=("$line")
done < <(python3 -c "$PYTHON_PARSE_CONFIG" "$CONFIG_FILE" "${RULE_NAMES[@]}")

# ─── Load model registry ──────────────────────────────────────────────────────
source utils/model_registry.sh

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

# ─── Validate input directory ─────────────────────────────────────────────────
if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found: $IMAGES_DIR"
    echo "Please ensure the dataset exists."
    exit 1
fi

# =============================================================================
# Stage 1 — Generate metamorphic images
# =============================================================================
if [ "$SKIP_TRANSFORM" = false ]; then
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Stage 1: Generating metamorphic images      ║"
    echo "╚══════════════════════════════════════════════╝"
    echo "Dataset: $DATASET"
    echo ""

    for i in "${!RULE_NAMES[@]}"; do
        rule="${RULE_NAMES[$i]}"
        config="${RULE_CONFIGS[$i]}"
        output_dir="$TRANSFORMED_DIR/$rule"
        mkdir -p "$output_dir"

        echo ""
        echo "── $rule ──"
        img_count=0
        for image in "$IMAGES_DIR"/*; do
            [ -f "$image" ] || continue
            python utils/matias_metamorphic.py \
                --rule "$rule" \
                --config "$config" \
                --base_image "$image" \
                --output_path "$output_dir" \
                --save-config
            ((img_count++)) || true
        done
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

    for rule_dir in "$TRANSFORMED_DIR"/*/; do
        [ -d "$rule_dir" ] || continue
        rule_name=$(basename "$rule_dir")
        echo "── $display_name — $rule_name ──"
        run_inference_dir "$script" "$weights" "$rule_dir" "$rule_name"
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
# Stage 3 — Evaluate (metamorphic comparison: original vs transformed)
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Stage 3: Running metamorphic analysis       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

bash analysis/scripts/run_all_analysis_matias_meta.sh --dataset "$DATASET"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Pipeline complete                           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Dataset: $DATASET"
echo "Analysis results: analysis/results/<model>/<rule>/original_vs_transformed/stats.json"
