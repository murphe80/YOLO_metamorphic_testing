#!/usr/bin/env bash
# =============================================================================
# Dataset-aware analysis for cone-colour metamorphic tests.
#
# Usage:
#   bash analysis/scripts/run_analysis_cone_color.sh [OPTIONS]
#
# Options:
#   --dataset <name>              Dataset used for inference (default: test)
#   --model <version|all>         Model version to analyze (default: all)
#   --transformations <value>     Comma-separated transformation names
#
# Inputs:
#   results/<dataset>/<model>/original/
#   results/<dataset>/<model>/<transformation>/
#
# Outputs:
#   analysis/results/<dataset>/<model>/<transformation>/
#       original_vs_transformed/stats.json
#       transformed_vs_gt/stats.json
# =============================================================================

set -euo pipefail
ANALYSIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ANALYSIS_SCRIPT_DIR/../.." || exit 1

DATASET="test"
MODEL_ARG="all"
TRANSFORMATION_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --dataset=*)
            DATASET="${1#*=}"
            shift
            ;;
        --model)
            MODEL_ARG="$2"
            shift 2
            ;;
        --model=*)
            MODEL_ARG="${1#*=}"
            shift
            ;;
        --transformations|--rules)
            TRANSFORMATION_ARG="$2"
            shift 2
            ;;
        --transformations=*|--rules=*)
            TRANSFORMATION_ARG="${1#*=}"
            shift
            ;;
        --help|-h)
            sed -n '3,18p' "$0"
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            exit 1
            ;;
    esac
done

source utils/model_registry.sh

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

DEFAULT_TRANSFORMATIONS=(
    yellow_to_blue
    blue_to_yellow
    yellow_to_orange
    blue_to_orange
)

if [ -n "$TRANSFORMATION_ARG" ]; then
    IFS=',' read -r -a TRANSFORMATIONS <<< "$TRANSFORMATION_ARG"
else
    TRANSFORMATIONS=("${DEFAULT_TRANSFORMATIONS[@]}")
fi

BASE_DATASET_DIR="data/base_images/$DATASET"
IMAGES_DIR="$BASE_DATASET_DIR/images"
GT_COCO="$BASE_DATASET_DIR/annotations_coco.json"
RESULTS_BASE="results/$DATASET"
ANALYSIS_BASE="analysis/results/$DATASET"

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

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found: $IMAGES_DIR"
    exit 1
fi

# Generate a full COCO ground-truth file when the dataset has source labels but
# no pre-generated annotations_coco.json.
if [ ! -f "$GT_COCO" ]; then
    if [ -d "$BASE_DATASET_DIR/bounding_boxes" ]; then
        echo "Generating ground truth: $GT_COCO"
        python3 analysis/utils/format_gt.py \
            --input_dir "$BASE_DATASET_DIR/bounding_boxes" \
            --output_file "$GT_COCO" \
            --image_dir "$IMAGES_DIR"
    elif [ -d "$BASE_DATASET_DIR/labels" ]; then
        echo "Generating ground truth: $GT_COCO"
        python3 analysis/utils/format_detections.py \
            --detection_dir "$BASE_DATASET_DIR/labels" \
            --output_file "$GT_COCO" \
            --as_ground_truth \
            --yolo_format \
            --image_dir "$IMAGES_DIR"
    fi
fi

echo ""
echo "======================================"
echo "Cone-colour metamorphic analysis"
echo "======================================"
echo "Dataset: $DATASET"
echo "Models: ${MODELS[*]}"
echo "Transformations: ${TRANSFORMATIONS[*]}"
if [ -f "$GT_COCO" ]; then
    echo "Ground truth: $GT_COCO"
else
    echo "Ground truth: unavailable (metamorphic comparison only)"
fi
echo ""

for model in "${MODELS[@]}"; do
    model_results="$RESULTS_BASE/$model"
    original_results="$model_results/original"

    if [ ! -d "$original_results" ]; then
        echo "Skipping $model: original detections not found at $original_results"
        continue
    fi

    original_coco="$original_results/detections_original_coco.json"
    echo "── Formatting $model original detections ──"
    python3 analysis/utils/format_detections.py \
        --detection_dir "$original_results" \
        --output_file "$original_coco" \
        --as_ground_truth \
        --image_dir "$IMAGES_DIR"

    for transformation in "${TRANSFORMATIONS[@]}"; do
        transformed_results="$model_results/$transformation"
        transformed_images="$TRANSFORMED_BASE/$transformation"

        if [ ! -d "$transformed_results" ]; then
            echo "Skipping $model/$transformation: detections directory not found."
            continue
        fi
        if [ ! -d "$transformed_images" ]; then
            echo "Skipping $model/$transformation: transformed images not found."
            continue
        fi

        stats_base="$ANALYSIS_BASE/$model/$transformation"
        mkdir -p "$stats_base/original_vs_transformed"

        echo ""
        echo "── Analyzing $model — $transformation ──"

        expected_original="$transformed_results/expected_original_coco.json"
        transformed_coco="$transformed_results/detections_coco.json"

        python3 analysis/utils/cone_color_analysis.py \
            --ground_truth_file "$original_coco" \
            --transformation "$transformation" \
            --output_path "$expected_original"

        python3 analysis/utils/format_detections.py \
            --detection_dir "$transformed_results" \
            --output_file "$transformed_coco" \
            --gt_coco "$expected_original"

        echo "Comparing transformed detections with colour-adjusted original detections."
        python3 analysis/coco_eval.py \
            --gt "$expected_original" \
            --dt "$transformed_coco" \
            --image-dir "$transformed_images" \
            --stats-out "$stats_base/original_vs_transformed/stats.json"

        if [ -f "$GT_COCO" ]; then
            mkdir -p "$stats_base/transformed_vs_gt"

            expected_gt="$transformed_results/expected_ground_truth_coco.json"
            transformed_for_gt="$transformed_results/detections_for_gt.json"

            python3 analysis/utils/cone_color_analysis.py \
                --ground_truth_file "$GT_COCO" \
                --transformation "$transformation" \
                --output_path "$expected_gt"

            python3 analysis/utils/format_detections.py \
                --detection_dir "$transformed_results" \
                --output_file "$transformed_for_gt" \
                --gt_coco "$expected_gt"

            echo "Comparing transformed detections with colour-adjusted ground truth."
            python3 analysis/coco_eval.py \
                --gt "$expected_gt" \
                --dt "$transformed_for_gt" \
                --image-dir "$transformed_images" \
                --stats-out "$stats_base/transformed_vs_gt/stats.json"
        fi
    done

    echo ""
    echo "✓ Completed cone-colour analysis for $model"
done

echo ""
echo "Analysis results: $ANALYSIS_BASE/<model>/<transformation>/"
