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
#   --dataset <name[,name...]>  Dataset(s) to use (default: test). May be
#                          repeated or supplied as a comma-separated list.
#                          Examples: test, train_samples, or
#                          --dataset test --dataset train_samples.
#   --model <v5|v8|v12|all> Which model(s) to run (default: all)
#   --config <path>        Transformation config file
#                          (default: config/transformations.json)
#                          Supports either one config object per rule or
#                          a list of config variants per rule.
#   --viz-samples <n>      Save detection visualizations for the first n images
#                          per model/configuration (default: 5; 0 disables)
#   --skip-transform       Skip Stage 1 (re-use existing transformed images)
#
# Examples:
#   bash scripts/run_pipeline.sh
#   bash scripts/run_pipeline.sh --dataset test --model v8
#   bash scripts/run_pipeline.sh --dataset train_samples
#   bash scripts/run_pipeline.sh --dataset test,train_samples
#   bash scripts/run_pipeline.sh --dataset fsoco_sample --skip-transform
#
# Outputs are written to:
#   data/transformed_images/<dataset>/<rule>/
#   results/<dataset>/<model>/original/
#   results/<dataset>/<model>/<rule>/
#   analysis/results/<dataset>/<model>/<rule>/
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

# All project scripts target Python 3.  Some environments do not provide a
# `python` alias, which previously made Stage 2 stop at the first model with
# only a shell error on stderr.
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python interpreter '$PYTHON_BIN' was not found. Set PYTHON_BIN to a Python 3 executable."
    exit 1
fi

# Keep Ultralytics' settings/cache inside the project.  This also supports
# read-only home directories used by CI and containerized runs.
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$PWD/.ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"

# ─── Parse arguments ──────────────────────────────────────────────────────────
DATASETS=()
MODEL_ARG="all"
CONFIG_FILE="config/brute_force_transformations.json"
VIZ_SAMPLES=5
SKIP_TRANSFORM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            IFS=',' read -r -a selected_datasets <<< "$2"
            DATASETS+=("${selected_datasets[@]}")
            shift 2
            ;;
        --dataset=*)
            IFS=',' read -r -a selected_datasets <<< "${1#*=}"
            DATASETS+=("${selected_datasets[@]}")
            shift
            ;;
        --model) MODEL_ARG="$2"; shift 2 ;;
        --model=*) MODEL_ARG="${1#*=}"; shift ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --config=*) CONFIG_FILE="${1#*=}"; shift ;;
        --viz-samples) VIZ_SAMPLES="$2"; shift 2 ;;
        --viz-samples=*) VIZ_SAMPLES="${1#*=}"; shift ;;
        --skip-transform) SKIP_TRANSFORM=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "${#DATASETS[@]}" -eq 0 ]; then
    DATASETS=("test")
fi

# Run each requested dataset independently so transformations, detections, and
# analyses cannot overwrite one another. Re-invoking this script keeps the
# existing single-dataset pipeline unchanged.
if [ "${#DATASETS[@]}" -gt 1 ]; then
    for dataset in "${DATASETS[@]}"; do
        if [ -z "$dataset" ]; then
            echo "Error: Dataset names cannot be empty"
            exit 1
        fi
        args=(--dataset "$dataset" --model "$MODEL_ARG" --config "$CONFIG_FILE" --viz-samples "$VIZ_SAMPLES")
        if [ "$SKIP_TRANSFORM" = true ]; then
            args+=(--skip-transform)
        fi
        bash "$0" "${args[@]}"
    done
    exit 0
fi

DATASET="${DATASETS[0]}"
if [ -z "$DATASET" ]; then
    echo "Error: Dataset names cannot be empty"
    exit 1
fi
if [[ ! "$DATASET" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Error: Dataset names may contain only letters, numbers, '.', '_', and '-': $DATASET"
    exit 1
fi

# ─── Configure paths based on dataset ─────────────────────────────────────────
if [ "$DATASET" = "fsoco_sample" ]; then
    IMAGES_DIR="data/base_images/fsoco_sample/images"
    TRANSFORMED_DIR="data/transformed_images/fsoco_sample"
elif [ "$DATASET" = "test" ]; then
    IMAGES_DIR="data/base_images/test/images"
    TRANSFORMED_DIR="data/transformed_images/fsoco_test"
elif [ "$DATASET" = "val" ]; then
    IMAGES_DIR="data/base_images/val/images"
    TRANSFORMED_DIR="data/transformed_images/fsoco_val"
elif [ "$DATASET" = "train_samples" ]; then
    IMAGES_DIR="data/base_images/train_samples/images"
    TRANSFORMED_DIR="data/transformed_images/train_samples"
else
    IMAGES_DIR="data/base_images/$DATASET/images"
    TRANSFORMED_DIR="data/transformed_images/$DATASET"
fi

RESULTS_BASE="results/$DATASET"

# ─── Transformation rules ─────────────────────────────────────────────────────
# Load transformation configurations from centralized config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Transformation config file not found at $CONFIG_FILE"
    exit 1
fi

# Parse transformation configs using Python. The legacy format stores one
# object per rule. Brute-force configs may store a list of variants per rule.
read -r -d '' PYTHON_PARSE_CONFIG << 'EOF' || true
import json
import sys
with open(sys.argv[1]) as f:
    config = json.load(f)
    for rule_name in sys.argv[2:]:
        rule_config = config.get(rule_name, {})
        variants = rule_config if isinstance(rule_config, list) else [rule_config]
        for idx, variant in enumerate(variants):
            if variant is None:
                variant = {}
            if not isinstance(variant, dict):
                raise TypeError(f"{rule_name} variant {idx} must be a JSON object")
            output_name = variant.get("variant_id", rule_name if len(variants) == 1 else f"{rule_name}_{idx}")
            print(f"{rule_name}\t{output_name}\t{json.dumps(variant, separators=(',', ':'))}")
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
RUN_RULE_NAMES=()
RUN_OUTPUT_NAMES=()
RULE_CONFIGS=()
while IFS= read -r line; do
    IFS=$'\t' read -r rule_name output_name config_json <<< "$line"
    RUN_RULE_NAMES+=("$rule_name")
    RUN_OUTPUT_NAMES+=("$output_name")
    RULE_CONFIGS+=("$config_json")
done < <(python3 -c "$PYTHON_PARSE_CONFIG" "$CONFIG_FILE" "${RULE_NAMES[@]}")

for output_name in "${RUN_OUTPUT_NAMES[@]}"; do
    if [[ ! "$output_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "Error: Rule output names may contain only letters, numbers, '.', '_', and '-': $output_name"
        exit 1
    fi
done

# ─── Load model registry ──────────────────────────────────────────────────────
source utils/model_registry.sh

# ─── Helper: run inference for one model on one image directory ───────────────
is_supported_image() {
    case "$1" in
        *.png|*.PNG|*.jpg|*.JPG|*.jpeg|*.JPEG) return 0 ;;
        *) return 1 ;;
    esac
}

run_inference_dir() {
    local script="$1"
    local weights="$2"
    local image_dir="$3"
    local output_subdir="$4"

    for image in "$image_dir"/*; do
        [ -f "$image" ] || continue
        is_supported_image "$image" || continue
        "$PYTHON_BIN" "$script" \
            --weights "$weights" \
            --image "$image" \
            --output-subdir "$output_subdir"
    done
}

clear_yolov26_v2_run() {
    local results_dir="$1"
    local expected_results_dir="results/$DATASET/yolov26_v2"
    local analysis_dir="analysis/results/$DATASET/yolov26_v2"
    local output_name

    if [ "$results_dir" != "$expected_results_dir" ]; then
        echo "Error: Refusing to clear unexpected YOLOv26_V2 results directory: $results_dir"
        return 1
    fi

    echo "Clearing stale YOLOv26_V2 outputs for this pipeline run..."
    rm -rf -- "$results_dir/original" "$analysis_dir/original_vs_gt"
    for output_name in "${RUN_OUTPUT_NAMES[@]}"; do
        rm -rf -- "$results_dir/$output_name" "$analysis_dir/$output_name"
    done
}

visualize_detection_samples() {
    local model_version="$1"
    local image_dir="$2"
    local results_dir="$3"
    local output_subdir="$4"

    if [ "$VIZ_SAMPLES" -le 0 ]; then
        return
    fi

    local detections_dir="$results_dir/$output_subdir"
    local visualizations_dir="$detections_dir/visualizations"
    local count=0

    [ -d "$detections_dir" ] || return
    mkdir -p "$visualizations_dir"

    for detections_file in "$detections_dir"/*.txt; do
        [ -f "$detections_file" ] || continue

        local stem
        local image_path=""
        stem=$(basename "$detections_file" .txt)

        for ext in jpg jpeg png JPG JPEG PNG; do
            if [ -f "$image_dir/$stem.$ext" ]; then
                image_path="$image_dir/$stem.$ext"
                break
            fi
        done

        if [ -z "$image_path" ]; then
            echo "  ! No source image found for $detections_file"
            continue
        fi

        python3 utils/visualise.py \
            --image "$image_path" \
            --detections "$detections_file" \
            --output "$visualizations_dir/$stem.jpg" \
            --model "$model_version" \
            --conf_threshold 0.0 \
            --no_show

        ((count++)) || true
        if [ "$count" -ge "$VIZ_SAMPLES" ]; then
            break
        fi
    done

    echo "  ✓ $count visualization samples → $visualizations_dir"
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
    echo "Config: $CONFIG_FILE"
    echo ""

    for i in "${!RUN_RULE_NAMES[@]}"; do
        rule="${RUN_RULE_NAMES[$i]}"
        output_name="${RUN_OUTPUT_NAMES[$i]}"
        config="${RULE_CONFIGS[$i]}"
        output_dir="$TRANSFORMED_DIR/$output_name"
        mkdir -p "$output_dir"

        echo ""
        echo "── $output_name ($rule) ──"
        echo "Configuration: $config"
        img_count=0
        for image in "$IMAGES_DIR"/*; do
            [ -f "$image" ] || continue
            is_supported_image "$image" || continue
            "$PYTHON_BIN" utils/matias_metamorphic.py \
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
    local model_version=$(get_model_field "$model_name" "version")
    local results_dir="$RESULTS_BASE/$model_name"

    if [ -z "$script" ] || [ ! -s "$script" ]; then
        echo "Error: Inference script for $display_name is missing or empty: ${script:-<unset>}"
        return 1
    fi
    if [ -z "$weights" ] || [ ! -f "$weights" ]; then
        echo "Error: Weights for $display_name were not found: ${weights:-<unset>}"
        return 1
    fi

    if [ "$model_name" = "yolov26_v2" ]; then
        clear_yolov26_v2_run "$results_dir"
    fi

    echo ""
    echo "── $display_name — original images ──"
    # Inference scripts store outputs below results/<model>/ by default. The
    # relative subdirectory below resolves to results/<dataset>/<model>/.
    run_inference_dir "$script" "$weights" "$IMAGES_DIR" "../$DATASET/$model_name/original"
    visualize_detection_samples "$model_version" "$IMAGES_DIR" "$results_dir" "original"

    for rule_name in "${RUN_OUTPUT_NAMES[@]}"; do
        rule_dir="$TRANSFORMED_DIR/$rule_name"
        [ -d "$rule_dir" ] || continue
        echo "── $display_name — $rule_name ──"
        run_inference_dir "$script" "$weights" "$rule_dir" "../$DATASET/$model_name/$rule_name"
        visualize_detection_samples "$model_version" "$rule_dir" "$results_dir" "$rule_name"
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

RULE_FILTER=$(IFS=,; echo "${RUN_OUTPUT_NAMES[*]}")
bash analysis/scripts/run_all_analysis_matias_meta.sh \
    --dataset "$DATASET" \
    --detections-base "$RESULTS_BASE" \
    --rules "$RULE_FILTER"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Pipeline complete                           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Dataset: $DATASET"
echo "Config: $CONFIG_FILE"
echo "Visualization samples per model/configuration: $VIZ_SAMPLES"
echo "Detections: results/$DATASET/<model>/<rule>/"
echo "Analysis results: analysis/results/$DATASET/<model>/<rule>/original_vs_transformed/stats.json"
