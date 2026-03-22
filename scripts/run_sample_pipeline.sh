#!/bin/bash
# =============================================================================
# run_sample_pipeline.sh
#
# End-to-end metamorphic testing pipeline on the bundled FSOCO sample images.
# Runs all three stages in sequence:
#   1. Generate transformed images (6 metamorphic rules)
#   2. Run inference with YOLOv5, YOLOv8, and YOLOv12
#   3. Evaluate: compare original detections vs transformed detections
#
# Usage:
#   bash scripts/run_sample_pipeline.sh [--model v5|v8|v12|all] [--skip-transform]
#
# Outputs are written to:
#   data/transformed_images/fsoco_sample/<rule>/
#   results/<model>/original/
#   results/<model>/<rule>/
#   analysis/results/<model>/<rule>/
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

# ─── Configurable parameters ──────────────────────────────────────────────────
IMAGES_DIR="data/base_images/fsoco_sample/segmented_images"
TRANSFORMED_DIR="data/transformed_images/fsoco_sample"
RESULTS_BASE="results"

MODEL_ARG="all"
SKIP_TRANSFORM=false

for arg in "$@"; do
    case $arg in
        --model) MODEL_ARG="${2:-all}"; shift ;;
        --model=*) MODEL_ARG="${arg#*=}" ;;
        --skip-transform) SKIP_TRANSFORM=true ;;
    esac
done

# ─── Transformation rules ─────────────────────────────────────────────────────
# Each rule name maps to a JSON config string.
declare -A RULE_CONFIGS
RULE_CONFIGS[imgDarkBright]='{"brightness_constant": -20, "brightness_multiplier": 0.2, "gamma": 1, "method": "gamma"}'
RULE_CONFIGS[imgResolution]='{"size_ratio": 0.6}'
RULE_CONFIGS[imgMotionBlur]='{"kernel_size": 12, "orientation": 45}'
RULE_CONFIGS[imgStretch]='{"height_ratio": 1.5, "width_ratio": 1.5}'
RULE_CONFIGS[imgGrayscale]='{}'
RULE_CONFIGS[imgColorWheel]='{"hue_rotation": 15}'

# ─── Model weights and inference scripts ──────────────────────────────────────
V5_WEIGHTS="models/yolov5/yolov5_param_kfold.pt"
V8_WEIGHTS="models/yolov8/yolov8_param_kfold.pt"
V12_WEIGHTS="models/yolov12/yolov12_param_kfold.pt"

V5_SCRIPT="models/yolov5/infer_v5_kfold.py"
V8_SCRIPT="models/yolov8/infer_v8.py"
V12_SCRIPT="models/yolov12/infer_v12.py"

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

# =============================================================================
# Stage 1 — Generate metamorphic images
# =============================================================================
if [ "$SKIP_TRANSFORM" = false ]; then
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Stage 1: Generating metamorphic images      ║"
    echo "╚══════════════════════════════════════════════╝"

    for rule in "${!RULE_CONFIGS[@]}"; do
        config="${RULE_CONFIGS[$rule]}"
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
    echo "Skipping Stage 1 (--skip-transform passed)."
fi

# =============================================================================
# Stage 2 — Run inference
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Stage 2: Running inference                  ║"
echo "╚══════════════════════════════════════════════╝"

run_model() {
    local script="$1"
    local weights="$2"
    local model_label="$3"

    echo ""
    echo "── $model_label — original images ──"
    run_inference_dir "$script" "$weights" "$IMAGES_DIR" "original"

    for rule_dir in "$TRANSFORMED_DIR"/*/; do
        [ -d "$rule_dir" ] || continue
        rule_name=$(basename "$rule_dir")
        echo "── $model_label — $rule_name ──"
        run_inference_dir "$script" "$weights" "$rule_dir" "$rule_name"
    done
}

if [[ "$MODEL_ARG" == "v5" || "$MODEL_ARG" == "all" ]]; then
    run_model "$V5_SCRIPT" "$V5_WEIGHTS" "YOLOv5"
fi
if [[ "$MODEL_ARG" == "v8" || "$MODEL_ARG" == "all" ]]; then
    run_model "$V8_SCRIPT" "$V8_WEIGHTS" "YOLOv8"
fi
if [[ "$MODEL_ARG" == "v12" || "$MODEL_ARG" == "all" ]]; then
    run_model "$V12_SCRIPT" "$V12_WEIGHTS" "YOLOv12"
fi

# =============================================================================
# Stage 3 — Evaluate (metamorphic comparison: original vs transformed)
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Stage 3: Running metamorphic analysis       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Note: No ground-truth annotations are included with the sample data,"
echo "      so only the metamorphic comparison (original vs transformed) runs."
echo ""

bash analysis/scripts/run_all_analysis_matias_meta.sh

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Pipeline complete                           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Analysis results: analysis/results/<model>/<rule>/original_vs_transformed/stats.json"
