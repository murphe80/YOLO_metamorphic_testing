#!/usr/bin/env bash
# =============================================================================
# Run both metamorphic experiments on the flat, unannotated competition data.
#
# The general image transformations are delegated to run_pipeline.sh. For the
# cone-colour experiment, each model's detections on the original images are
# used as pseudo-labels to locate and classify cones. Those pseudo-labels are
# used only to generate transformed images; evaluation compares original model
# detections with transformed-image detections and never treats them as GT.
#
# Usage:
#   bash scripts/run_competition_pipeline.sh [OPTIONS]
#
# Options:
#   --images-dir <path>  Flat competition image directory
#   --config <path>      General transformation config
#   --viz-samples <n>    Visualisations per model/transformation (default: 5)
#   --skip-transform     Reuse previously generated transformed images
#   --resume-cone-inference
#                        Finish the pending general analysis without rerunning
#                        inference, then resume the cone-colour experiment
#   --help               Show this help
#
# Outputs:
#   results/competition/<model>/...
#   analysis/results/competition/<model>/...
#   data/transformed_images/competition/...
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_PIPELINE_SCRIPT="$PROJECT_ROOT/scripts/run_pipeline.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="competition"
IMAGES_DIR="data/base_images/comp_data/filtered_comp_data"
CONFIG_FILE="config/brute_force_transformations.json"
VIZ_SAMPLES=5
SKIP_TRANSFORM=false
RESUME_CONE_INFERENCE=false

TRANSFORMATIONS=(
    yellow_to_blue
    blue_to_yellow
    yellow_to_orange
    blue_to_orange
)

usage() {
    sed -n '3,28p' "$0"
}

require_value() {
    local option="$1"
    local value="${2:-}"

    if [ -z "$value" ] || [[ "$value" == --* ]]; then
        echo "Error: $option requires a value."
        exit 1
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --images-dir)
            require_value "$1" "${2:-}"
            IMAGES_DIR="$2"
            shift 2
            ;;
        --images-dir=*)
            IMAGES_DIR="${1#*=}"
            shift
            ;;
        --config)
            require_value "$1" "${2:-}"
            CONFIG_FILE="$2"
            shift 2
            ;;
        --config=*)
            CONFIG_FILE="${1#*=}"
            shift
            ;;
        --viz-samples)
            require_value "$1" "${2:-}"
            VIZ_SAMPLES="$2"
            shift 2
            ;;
        --viz-samples=*)
            VIZ_SAMPLES="${1#*=}"
            shift
            ;;
        --skip-transform)
            SKIP_TRANSFORM=true
            shift
            ;;
        --resume-cone-inference)
            RESUME_CONE_INFERENCE=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python interpreter '$PYTHON_BIN' was not found."
    exit 1
fi
if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Competition image directory not found: $IMAGES_DIR"
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Transformation config not found: $CONFIG_FILE"
    exit 1
fi
if ! bash -n "$RUN_PIPELINE_SCRIPT"; then
    echo "Error: General pipeline script has invalid shell syntax: $RUN_PIPELINE_SCRIPT"
    exit 1
fi
if ! [[ "$VIZ_SAMPLES" =~ ^[0-9]+$ ]]; then
    echo "Error: --viz-samples must be a non-negative integer."
    exit 1
fi

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$PROJECT_ROOT/.ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"

source utils/model_registry.sh
read -r -a MODELS <<< "$(get_model_names)"

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "Error: No enabled models were found in config/models.json."
    exit 1
fi

directory_has_images() {
    local directory="$1"

    [ -d "$directory" ] || return 1
    compgen -G "$directory/*.jpg" >/dev/null ||
        compgen -G "$directory/*.jpeg" >/dev/null ||
        compgen -G "$directory/*.png" >/dev/null ||
        compgen -G "$directory/*.JPG" >/dev/null ||
        compgen -G "$directory/*.JPEG" >/dev/null ||
        compgen -G "$directory/*.PNG" >/dev/null
}

is_supported_image() {
    case "$1" in
        *.png|*.PNG|*.jpg|*.JPG|*.jpeg|*.JPEG) return 0 ;;
        *) return 1 ;;
    esac
}

if ! directory_has_images "$IMAGES_DIR"; then
    echo "Error: No supported images found in: $IMAGES_DIR"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Competition metamorphic experiments             ║"
echo "╚══════════════════════════════════════════════════╝"
echo "Images: $IMAGES_DIR"
echo "Models: ${MODELS[*]}"
echo "Ground truth: unavailable (original vs transformed only)"

# -----------------------------------------------------------------------------
# Experiment 1: general metamorphic transformations
# -----------------------------------------------------------------------------
general_args=(
    --dataset "$DATASET"
    --images-dir "$IMAGES_DIR"
    --model all
    --config "$CONFIG_FILE"
    --viz-samples "$VIZ_SAMPLES"
)
if [ "$SKIP_TRANSFORM" = true ]; then
    general_args+=(--skip-transform)
fi

if [ "$RESUME_CONE_INFERENCE" = false ]; then
    bash "$RUN_PIPELINE_SCRIPT" "${general_args[@]}"
else
    echo ""
    echo "Resuming with general analysis, then cone-colour inference."
    bash "$RUN_PIPELINE_SCRIPT" "${general_args[@]}" --analysis-only
fi

# -----------------------------------------------------------------------------
# Experiment 2: cone-colour transformations using per-model pseudo-labels
# -----------------------------------------------------------------------------
HIST_SAMPLES_DIR="utils/hist_backprojection"
SEGMENT_SCRIPT="utils/histogram_backproj_yolo.py"
TRANSFORM_SCRIPT="utils/cone_colour_class_transformation.py"
CONE_WORK_BASE="data/transformed_images/$DATASET/cone_color_by_model"
RESULTS_BASE="$PROJECT_ROOT/results/$DATASET"

for required_path in "$HIST_SAMPLES_DIR" "$SEGMENT_SCRIPT" "$TRANSFORM_SCRIPT"; do
    if [ ! -e "$required_path" ]; then
        echo "Error: Required cone-colour resource not found: $required_path"
        exit 1
    fi
done

make_pseudo_labels() {
    local detections_dir="$1"
    local labels_dir="$2"
    local detection_file
    local label_file

    mkdir -p "$labels_dir"
    for detection_file in "$detections_dir"/*.txt; do
        [ -f "$detection_file" ] || continue
        label_file="$labels_dir/$(basename "$detection_file")"
        # Detection format: class confidence x1 y1 x2 y2
        # Transformation label format: class x1 y1 x2 y2
        awk 'NF >= 6 { print $1, $3, $4, $5, $6 }' \
            "$detection_file" > "$label_file"
    done
}

reset_existing_transforms() {
    local transformed_dir="$1"
    local transformed_image
    local source_image

    # A cone that was detected on an earlier run may no longer be present in
    # the current pseudo-labels. Reset only previously materialised variants
    # to their source pixels before applying the current transformations.
    for transformed_image in "$transformed_dir"/*; do
        [ -f "$transformed_image" ] || continue
        is_supported_image "$transformed_image" || continue
        source_image="$IMAGES_DIR/$(basename "$transformed_image")"
        [ -f "$source_image" ] || continue
        cp -f -- "$source_image" "$transformed_image"
    done
}

run_cone_inference_dir() {
    local inference_script="$1"
    local weights="$2"
    local transformed_dir="$3"
    local output_dir="$4"
    local source_image
    local inference_image
    local image_name
    local image_stem
    local skipped=0

    mkdir -p "$output_dir"
    for source_image in "$IMAGES_DIR"/*; do
        [ -f "$source_image" ] || continue
        is_supported_image "$source_image" || continue

        inference_image="$transformed_dir/$(basename "$source_image")"
        if [ ! -f "$inference_image" ]; then
            inference_image="$source_image"
        fi

        image_name=$(basename "$source_image")
        image_stem="${image_name%.*}"
        if [ "$RESUME_CONE_INFERENCE" = true ] && [ -f "$output_dir/$image_stem.txt" ]; then
            ((skipped++)) || true
            continue
        fi

        "$PYTHON_BIN" "$inference_script" \
            --weights "$weights" \
            --image "$inference_image" \
            --output-subdir "$output_dir"
    done

    if [ "$RESUME_CONE_INFERENCE" = true ]; then
        echo "  ↪ Reused $skipped existing detection files"
    fi
}

visualize_detection_samples() {
    local model_version="$1"
    local image_dir="$2"
    local detections_dir="$3"
    local fallback_image_dir="${4:-}"
    local visualizations_dir="$detections_dir/visualizations"
    local detections_file
    local image_path
    local stem
    local ext
    local count=0

    if [ "$VIZ_SAMPLES" -eq 0 ] || [ ! -d "$detections_dir" ]; then
        return
    fi

    mkdir -p "$visualizations_dir"
    for detections_file in "$detections_dir"/*.txt; do
        [ -f "$detections_file" ] || continue
        stem=$(basename "$detections_file" .txt)
        image_path=""

        for ext in jpg jpeg png JPG JPEG PNG; do
            if [ -f "$image_dir/$stem.$ext" ]; then
                image_path="$image_dir/$stem.$ext"
                break
            fi
            if [ -n "$fallback_image_dir" ] && [ -f "$fallback_image_dir/$stem.$ext" ]; then
                image_path="$fallback_image_dir/$stem.$ext"
                break
            fi
        done

        [ -n "$image_path" ] || continue
        "$PYTHON_BIN" utils/visualise.py \
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

    echo "  ✓ $count visualisation samples → $visualizations_dir"
}

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Cone-colour experiment                          ║"
echo "╚══════════════════════════════════════════════════╝"

for model in "${MODELS[@]}"; do
    inference_script=$(get_model_field "$model" "inference_script")
    weights=$(get_model_field "$model" "weights")
    display_name=$(get_model_field "$model" "display_name")
    model_version=$(get_model_field "$model" "version")
    original_results="$RESULTS_BASE/$model/original"
    model_work="$CONE_WORK_BASE/$model"
    pseudo_labels="$model_work/pseudo_labels"
    segmented_dir="$model_work/segmented_images"
    transformed_base="$model_work/transformed_images"

    if [ ! -d "$original_results" ]; then
        echo "Error: Original detections missing for $model: $original_results"
        exit 1
    fi
    if [ ! -f "$inference_script" ] || [ ! -f "$weights" ]; then
        echo "Error: Inference resources missing for $model."
        exit 1
    fi

    echo ""
    echo "── $display_name: preparing cone-colour sources ──"

    if [ "$SKIP_TRANSFORM" = false ]; then
        make_pseudo_labels "$original_results" "$pseudo_labels"
        mkdir -p "$segmented_dir"
        "$PYTHON_BIN" "$SEGMENT_SCRIPT" \
            --images_dir "$IMAGES_DIR" \
            --detections_dir "$original_results" \
            --hist_samples_dir "$HIST_SAMPLES_DIR" \
            --output_dir "$segmented_dir"

        for transformation in "${TRANSFORMATIONS[@]}"; do
            transformed_dir="$transformed_base/$transformation"

            mkdir -p "$transformed_dir"
            reset_existing_transforms "$transformed_dir"
            "$PYTHON_BIN" "$TRANSFORM_SCRIPT" \
                --images_dir "$IMAGES_DIR" \
                --labels_dir "$pseudo_labels" \
                --segmented_dir "$segmented_dir" \
                --output_dir "$transformed_dir" \
                --transformation "$transformation"
        done
    fi

    for transformation in "${TRANSFORMATIONS[@]}"; do
        transformed_dir="$transformed_base/$transformation"
        transformed_results="$RESULTS_BASE/$model/$transformation"

        if [ ! -d "$transformed_dir" ]; then
            echo "Error: Transformed-image directory missing for $model/$transformation:"
            echo "  $transformed_dir"
            exit 1
        fi

        echo "── $display_name — $transformation ──"
        # The colour utility writes only images containing the source class.
        # For every other image, infer on the unchanged source so that the
        # original and follow-up detection sets cover the same source images.
        run_cone_inference_dir \
            "$inference_script" \
            "$weights" \
            "$transformed_dir" \
            "$transformed_results"
        visualize_detection_samples \
            "$model_version" \
            "$transformed_dir" \
            "$transformed_results" \
            "$IMAGES_DIR"
    done

    transformation_filter=$(IFS=,; echo "${TRANSFORMATIONS[*]}")
    bash analysis/scripts/run_analysis_cone_color.sh \
        --dataset "$DATASET" \
        --model "$model_version" \
        --images-dir "$IMAGES_DIR" \
        --transformed-base "$transformed_base" \
        --transformations "$transformation_filter"
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Competition experiments complete                ║"
echo "╚══════════════════════════════════════════════════╝"
echo "Detections: results/$DATASET/<model>/<transformation>/"
echo "Analysis: analysis/results/$DATASET/<model>/<transformation>/"
echo "Cone-colour sources: $CONE_WORK_BASE/<model>/transformed_images/"
echo "Ground-truth comparisons were not run."
