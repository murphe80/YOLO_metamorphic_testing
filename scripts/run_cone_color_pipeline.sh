#!/usr/bin/env bash
# =============================================================================
# run_cone_color_pipeline.sh
#
# End-to-end cone-colour metamorphic testing pipeline.
#
# Stages:
#   1. Generate segmentation masks when they do not already exist.
#   2. Generate cone-colour transformed images.
#   3. Run inference on the original and transformed images.
#   4. Compare the transformed detections with the colour-adjusted expectations.
#
# Usage:
#   bash scripts/run_cone_color_pipeline.sh [OPTIONS]
#
# Options:
#   --dataset <name[,name...]> Dataset(s) under data/base_images/ (default: test)
#   --datasets <name[,name...]> Alias for --dataset
#   --model <version|all>     Model version to run, for example v8 (default: all)
#   --transformations <value> all, one transformation, or a comma-separated list
#                             (default: all; --transformation is an alias)
#   --viz-samples <n>         Detection visualisations per result directory
#                             (default: 5; 0 disables)
#   --skip-transform          Reuse existing segmentation and transformed images
#   --help                    Show this help
#
# Default transformations:
#   yellow_to_blue, blue_to_yellow, yellow_to_orange, blue_to_orange
#
# Dataset layout:
#   data/base_images/<dataset>/images/
#   data/base_images/<dataset>/labels/             (YOLO TXT), or
#   data/base_images/<dataset>/bounding_boxes/     (Supervisely JSON)
#   data/base_images/<dataset>/segmented_images/
#
# Multiple datasets are processed sequentially to avoid concurrent GPU inference.
#
# Outputs:
#   data/transformed_images/<output-dataset>/<transformation>/
#   results/<dataset>/<model>/{original,<transformation>}/
#   analysis/results/<dataset>/<model>/<transformation>/
#
# The test and val output-dataset names follow run_pipeline.sh:
#   test -> fsoco_test
#   val  -> fsoco_val
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

DATASET="test"
MODEL_ARG="all"
TRANSFORMATION_ARG="all"
VIZ_SAMPLES=5
SKIP_TRANSFORM=false

usage() {
    sed -n '3,43p' "$0"
}

require_value() {
    local option="$1"
    local value="${2:-}"

    if [ -z "$value" ] || [[ "$value" == --* ]]; then
        echo "Error: $option requires a value."
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset|--datasets)
            require_value "$1" "${2:-}"
            DATASET="$2"
            shift 2
            ;;
        --dataset=*|--datasets=*)
            DATASET="${1#*=}"
            shift
            ;;
        --model)
            require_value "$1" "${2:-}"
            MODEL_ARG="$2"
            shift 2
            ;;
        --model=*)
            MODEL_ARG="${1#*=}"
            shift
            ;;
        --transformation|--transformations)
            require_value "$1" "${2:-}"
            TRANSFORMATION_ARG="$2"
            shift 2
            ;;
        --transformation=*|--transformations=*)
            TRANSFORMATION_ARG="${1#*=}"
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

if [ -z "$DATASET" ]; then
    echo "Error: --dataset cannot be empty."
    exit 1
fi

if [[ "$DATASET" == ,* || "$DATASET" == *, || "$DATASET" == *,,* ]]; then
    echo "Error: --dataset must be a comma-separated list of non-empty dataset names."
    exit 1
fi

IFS=',' read -r -a DATASETS <<< "$DATASET"

if [ "${#DATASETS[@]}" -gt 1 ]; then
    echo "Running cone-colour experiment for datasets: ${DATASETS[*]}"

    for dataset in "${DATASETS[@]}"; do
        echo ""
        echo "════════════════ Dataset: $dataset ════════════════"

        dataset_command=(
            bash "$0"
            --dataset "$dataset"
            --model "$MODEL_ARG"
            --transformations "$TRANSFORMATION_ARG"
            --viz-samples "$VIZ_SAMPLES"
        )
        if [ "$SKIP_TRANSFORM" = true ]; then
            dataset_command+=(--skip-transform)
        fi
        "${dataset_command[@]}"
    done

    echo ""
    echo "Cone-colour multi-dataset experiment complete."
    exit 0
fi

DATASET="${DATASETS[0]}"

if ! [[ "$VIZ_SAMPLES" =~ ^[0-9]+$ ]]; then
    echo "Error: --viz-samples must be a non-negative integer."
    exit 1
fi

source utils/model_registry.sh

BASE_DATASET_DIR="data/base_images/$DATASET"
IMAGES_DIR="$BASE_DATASET_DIR/images"

if [ -d "$BASE_DATASET_DIR/labels" ]; then
    LABELS_DIR="$BASE_DATASET_DIR/labels"
elif [ -d "$BASE_DATASET_DIR/bounding_boxes" ]; then
    LABELS_DIR="$BASE_DATASET_DIR/bounding_boxes"
else
    LABELS_DIR="$BASE_DATASET_DIR/labels"
fi

case "$DATASET" in
    test)
        TRANSFORMED_BASE="data/transformed_images/fsoco_test"
        ;;
    val)
        TRANSFORMED_BASE="data/transformed_images/fsoco_val"
        ;;
    *)
        TRANSFORMED_BASE="data/transformed_images/$DATASET"
        ;;
esac

RESULTS_BASE="$PWD/results/$DATASET"
HIST_SAMPLES_DIR="utils/hist_backprojection"
TRANSFORM_SCRIPT="utils/cone_colour_class_transformation.py"
SEGMENT_SCRIPT="utils/extract_cone_regions_gt.py"

DEFAULT_TRANSFORMATIONS=(
    yellow_to_blue
    blue_to_yellow
    yellow_to_orange
    blue_to_orange
)

SUPPORTED_TRANSFORMATIONS=(
    yellow_to_blue
    blue_to_yellow
    yellow_to_orange
    orange_to_blue
    blue_to_orange
    orange_to_yellow
    swap_sides
)

is_supported_transformation() {
    local candidate="$1"
    local supported

    for supported in "${SUPPORTED_TRANSFORMATIONS[@]}"; do
        if [ "$candidate" = "$supported" ]; then
            return 0
        fi
    done
    return 1
}

if [ "$TRANSFORMATION_ARG" = "all" ]; then
    TRANSFORMATIONS=("${DEFAULT_TRANSFORMATIONS[@]}")
else
    IFS=',' read -r -a TRANSFORMATIONS <<< "$TRANSFORMATION_ARG"
fi

if [ "${#TRANSFORMATIONS[@]}" -eq 0 ]; then
    echo "Error: At least one transformation must be selected."
    exit 1
fi

for transformation in "${TRANSFORMATIONS[@]}"; do
    if [ -z "$transformation" ] || ! is_supported_transformation "$transformation"; then
        echo "Error: Unsupported cone-colour transformation '$transformation'."
        echo "Supported transformations: ${SUPPORTED_TRANSFORMATIONS[*]}"
        exit 1
    fi
done

if [ "$MODEL_ARG" = "all" ]; then
    read -r -a MODELS <<< "$(get_model_names)"
else
    MODEL_NAME=$(get_model_by_version "$MODEL_ARG")
    if [ -z "$MODEL_NAME" ]; then
        echo "Error: Invalid model version '$MODEL_ARG'."
        echo "Valid versions: $(get_model_versions)"
        exit 1
    fi
    MODELS=("$MODEL_NAME")
fi

directory_has_images() {
    local directory="$1"

    [ -d "$directory" ] || return 1
    compgen -G "$directory/*.jpg" > /dev/null ||
        compgen -G "$directory/*.jpeg" > /dev/null ||
        compgen -G "$directory/*.png" > /dev/null ||
        compgen -G "$directory/*.JPG" > /dev/null ||
        compgen -G "$directory/*.JPEG" > /dev/null ||
        compgen -G "$directory/*.PNG" > /dev/null
}

if directory_has_images "$BASE_DATASET_DIR/segmented_images/segmented_images_good"; then
    SEGMENTED_DIR="$BASE_DATASET_DIR/segmented_images/segmented_images_good"
else
    SEGMENTED_DIR="$BASE_DATASET_DIR/segmented_images"
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found: $IMAGES_DIR"
    exit 1
fi

if [ ! -d "$LABELS_DIR" ]; then
    echo "Error: Labels directory not found."
    echo "Expected either:"
    echo "  $BASE_DATASET_DIR/labels"
    echo "  $BASE_DATASET_DIR/bounding_boxes"
    exit 1
fi

if [ ! -f "$TRANSFORM_SCRIPT" ] || [ ! -f "$SEGMENT_SCRIPT" ]; then
    echo "Error: Cone-colour transformation utilities are missing."
    exit 1
fi

if [ ! -d "$HIST_SAMPLES_DIR" ]; then
    echo "Error: Histogram samples directory not found: $HIST_SAMPLES_DIR"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Cone-colour metamorphic testing pipeline   ║"
echo "╚══════════════════════════════════════════════╝"
echo "Dataset: $DATASET"
echo "Images: $IMAGES_DIR"
echo "Labels: $LABELS_DIR"
echo "Transformed images: $TRANSFORMED_BASE"
echo "Transformations: ${TRANSFORMATIONS[*]}"
echo "Models: ${MODELS[*]}"

# =============================================================================
# Stage 1 — Generate segmentation masks and cone-colour transformations
# =============================================================================
if [ "$SKIP_TRANSFORM" = false ]; then
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Stage 1: Generating transformed images      ║"
    echo "╚══════════════════════════════════════════════╝"

    if ! directory_has_images "$SEGMENTED_DIR"; then
        SEGMENTED_DIR="$BASE_DATASET_DIR/segmented_images"
        mkdir -p "$SEGMENTED_DIR"

        echo "No segmentation masks found; generating them first."
        python3 "$SEGMENT_SCRIPT" \
            --images_dir "$IMAGES_DIR" \
            --labels_dir "$LABELS_DIR" \
            --hist_samples_dir "$HIST_SAMPLES_DIR" \
            --output_dir "$SEGMENTED_DIR"
    else
        echo "Reusing segmentation masks from: $SEGMENTED_DIR"
    fi

    if ! directory_has_images "$SEGMENTED_DIR"; then
        echo "Error: No segmentation masks were generated in $SEGMENTED_DIR"
        exit 1
    fi

    for transformation in "${TRANSFORMATIONS[@]}"; do
        output_dir="$TRANSFORMED_BASE/$transformation"
        mkdir -p "$output_dir"

        echo ""
        echo "── Generating $transformation ──"
        python3 "$TRANSFORM_SCRIPT" \
            --images_dir "$IMAGES_DIR" \
            --labels_dir "$LABELS_DIR" \
            --segmented_dir "$SEGMENTED_DIR" \
            --output_dir "$output_dir" \
            --transformation "$transformation"

        image_count=$(find "$output_dir" -maxdepth 1 -type f \
            \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
        image_count="${image_count//[[:space:]]/}"
        echo "  ✓ $image_count transformed images → $output_dir"
    done
else
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Stage 1: Skipped (--skip-transform)         ║"
    echo "╚══════════════════════════════════════════════╝"
fi

for transformation in "${TRANSFORMATIONS[@]}"; do
    transformed_dir="$TRANSFORMED_BASE/$transformation"
    if ! directory_has_images "$transformed_dir"; then
        echo "Error: No transformed images found for $transformation in:"
        echo "  $transformed_dir"
        exit 1
    fi
done

# =============================================================================
# Stage 2 — Run inference
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Stage 2: Running inference                  ║"
echo "╚══════════════════════════════════════════════╝"

run_inference_dir() {
    local inference_script="$1"
    local weights="$2"
    local image_dir="$3"
    local output_dir="$4"
    local image

    mkdir -p "$output_dir"
    for image in "$image_dir"/*; do
        [ -f "$image" ] || continue
        python3 "$inference_script" \
            --weights "$weights" \
            --image "$image" \
            --output-subdir "$output_dir"
    done
}

visualize_detection_samples() {
    local model_version="$1"
    local image_dir="$2"
    local detections_dir="$3"
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
        done

        if [ -z "$image_path" ]; then
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

    echo "  ✓ $count visualisation samples → $visualizations_dir"
}

for model in "${MODELS[@]}"; do
    inference_script=$(get_model_field "$model" "inference_script")
    weights=$(get_model_field "$model" "weights")
    display_name=$(get_model_field "$model" "display_name")
    model_version=$(get_model_field "$model" "version")
    model_results="$RESULTS_BASE/$model"

    if [ ! -f "$inference_script" ]; then
        echo "Error: Inference script not found for $model: $inference_script"
        exit 1
    fi
    if [ ! -f "$weights" ]; then
        echo "Error: Weights not found for $model: $weights"
        exit 1
    fi

    echo ""
    echo "── $display_name — original images ──"
    run_inference_dir "$inference_script" "$weights" "$IMAGES_DIR" "$model_results/original"
    visualize_detection_samples "$model_version" "$IMAGES_DIR" "$model_results/original"

    for transformation in "${TRANSFORMATIONS[@]}"; do
        transformed_dir="$TRANSFORMED_BASE/$transformation"
        transformation_results="$model_results/$transformation"

        echo "── $display_name — $transformation ──"
        run_inference_dir \
            "$inference_script" \
            "$weights" \
            "$transformed_dir" \
            "$transformation_results"
        visualize_detection_samples \
            "$model_version" \
            "$transformed_dir" \
            "$transformation_results"
    done
done

# =============================================================================
# Stage 3 — Evaluate cone-colour metamorphic relations
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Stage 3: Running metamorphic analysis       ║"
echo "╚══════════════════════════════════════════════╝"

TRANSFORMATION_FILTER=$(IFS=,; echo "${TRANSFORMATIONS[*]}")
bash analysis/scripts/run_analysis_cone_color.sh \
    --dataset "$DATASET" \
    --model "$MODEL_ARG" \
    --transformations "$TRANSFORMATION_FILTER"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Cone-colour pipeline complete               ║"
echo "╚══════════════════════════════════════════════╝"
echo "Transformed images: $TRANSFORMED_BASE/<transformation>/"
echo "Detections: results/$DATASET/<model>/<transformation>/"
echo "Analysis: analysis/results/$DATASET/<model>/<transformation>/"
