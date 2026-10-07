#!/bin/bash
# ============================================
# run_base_metamorphic_analysis.sh
#
# Analyze metamorphic testing results by comparing original vs transformed detections.
# Optionally compares against ground truth if available.
#
# Usage:
#   bash analysis/scripts/run_base_metamorphic_analysis.sh [OPTIONS]
#
# Options:
#   --dataset <name>           Dataset whose annotations are evaluated.
#   --model <name>             Analyze only the specified enabled model.
#   --detections-base <path>   Directory containing per-model detections
#                              (default: results).
#   --images-dir <path>        Source image directory. Used to record the
#                              correct image names and dimensions without GT.
#   --comparison <name>        Run all comparisons (default), or one of
#                              original_vs_transformed, transformed_vs_gt,
#                              original_vs_gt.
#
# Outputs: analysis/results/rq1/<dataset>/<model>/<rule>/<comparison_type>/stats.json
# ============================================

# Change to repo root if not already there
cd "$(dirname "$0")/../.." || exit 1

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Load model registry
source utils/model_registry.sh

# ============================================
# Parse arguments
# ============================================
DATASET="test"  # Default to sample dataset
RULE_FILTER=""
MODEL_FILTER=""
DETECTIONS_BASE="results"
IMAGES_DIR_OVERRIDE=""
COMPARISON_FILTER="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --dataset=*) DATASET="${1#*=}"; shift ;;
        --model)
            [ $# -ge 2 ] && [ -n "$2" ] || { echo "--model requires a model name."; exit 1; }
            MODEL_FILTER="$2"
            shift 2
            ;;
        --model=*)
            MODEL_FILTER="${1#*=}"
            [ -n "$MODEL_FILTER" ] || { echo "--model requires a model name."; exit 1; }
            shift
            ;;
        --rules) RULE_FILTER="$2"; shift 2 ;;
        --rules=*) RULE_FILTER="${1#*=}"; shift ;;
        --detections-base) DETECTIONS_BASE="$2"; shift 2 ;;
        --detections-base=*) DETECTIONS_BASE="${1#*=}"; shift ;;
        --images-dir) IMAGES_DIR_OVERRIDE="$2"; shift 2 ;;
        --images-dir=*) IMAGES_DIR_OVERRIDE="${1#*=}"; shift ;;
        --comparison) COMPARISON_FILTER="$2"; shift 2 ;;
        --comparison=*) COMPARISON_FILTER="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "$COMPARISON_FILTER" in
    all|original_vs_transformed|transformed_vs_gt|original_vs_gt) ;;
    *)
        echo "Unknown comparison: $COMPARISON_FILTER"
        exit 1
        ;;
esac

should_run_comparison() {
    [ "$COMPARISON_FILTER" = "all" ] || [ "$COMPARISON_FILTER" = "$1" ]
}

ENABLED_MODELS=$(get_model_names)
if [ -n "$MODEL_FILTER" ]; then
    model_found=false
    for model in $ENABLED_MODELS; do
        if [ "$model" = "$MODEL_FILTER" ]; then
            model_found=true
            break
        fi
    done

    if [ "$model_found" = false ]; then
        echo "Unknown or disabled model: $MODEL_FILTER"
        echo "Enabled models: $ENABLED_MODELS"
        exit 1
    fi
fi

if [ -n "$IMAGES_DIR_OVERRIDE" ]; then
    SOURCE_IMAGES_DIR="$IMAGES_DIR_OVERRIDE"
else
    SOURCE_IMAGES_DIR="data/base_images/$DATASET/images"
fi

# ============================================
# Configuration
# ============================================
RESULTS_BASE="analysis/results/rq1/$DATASET"
BRUTE_FORCE_CONFIG="config/brute_force_transformations.json"
BASE_TRANSFORMATIONS=(
    imgStretch
    imgResolution
    imgMotionBlur
    imgDarkBright
    imgGrayscale
    imgColorWheel
)
GEOMETRIC_TRANSFORMS=(
    imgStretch
    imgResolution
)

IFS=',' read -r -a SELECTED_RULES <<< "$RULE_FILTER"

should_process_rule() {
    local rule_name="$1"

    if [ -z "$RULE_FILTER" ]; then
        return 0
    fi

    for selected_rule in "${SELECTED_RULES[@]}"; do
        if [ "$rule_name" = "$selected_rule" ]; then
            return 0
        fi
    done

    return 1
}

is_geometric_transform() {
    local rule_name="$1"

    for geometric_rule in "${GEOMETRIC_TRANSFORMS[@]}"; do
        if [ "$rule_name" = "$geometric_rule" ]; then
            return 0
        fi
    done

    return 1
}

resolve_base_rule_name() {
    local rule_path="$1"
    local rule_name="$2"

    # Prefer explicit config metadata when available.
    for config_path in \
        "$rule_path/configs.json" \
        "$rule_path/config.json"; do
        [ ! -f "$config_path" ] && continue

        detected_rule=$(python3 - "$config_path" <<'PY'
import json
import sys

known_rules = {
    "imgStretch",
    "imgResolution",
    "imgMotionBlur",
    "imgDarkBright",
    "imgGrayscale",
    "imgColorWheel",
}

try:
    with open(sys.argv[1]) as f:
        config = json.load(f)
except (OSError, json.JSONDecodeError):
    sys.exit(0)

if isinstance(config, dict):
    transformation = config.get("transformation")
    if transformation in known_rules:
        print(transformation)
        sys.exit(0)

    for key in config:
        if key in known_rules:
            print(key)
            sys.exit(0)
PY
)
        if [ -n "$detected_rule" ]; then
            echo "$detected_rule"
            return 0
        fi
    done

    # Fall back to directory names for variant IDs such as imgStretch_0.
    for known_rule in "${BASE_TRANSFORMATIONS[@]}"; do
        case "$rule_name" in
            "$known_rule"|"$known_rule"_*)
                echo "$known_rule"
                return 0
                ;;
        esac
    done

    echo "$rule_name"
}

resolve_transform_params_json() {
    local base_rule_name="$1"
    local rule_name="$2"

    [ ! -f "$BRUTE_FORCE_CONFIG" ] && return 0

    python3 - "$BRUTE_FORCE_CONFIG" "$base_rule_name" "$rule_name" <<'PY'
import json
import sys

config_path, base_rule_name, rule_name = sys.argv[1:4]

try:
    with open(config_path) as f:
        config = json.load(f)
except (OSError, json.JSONDecodeError):
    sys.exit(0)

variants = config.get(base_rule_name, [])
if isinstance(variants, dict):
    variants = [variants]

for variant in variants:
    if not isinstance(variant, dict):
        continue
    if variant.get("variant_id") == rule_name:
        print(json.dumps(variant, separators=(",", ":")))
        sys.exit(0)
PY
}

# Set ground truth path based on dataset
if [ "$DATASET" = "test" ]; then
    GT_COCO="data/base_images/test/annotations_coco.json"
elif [ "$DATASET" = "fsoco_sample" ]; then
    GT_COCO="data/base_images/fsoco_sample/annotations_coco.json"
else
    GT_COCO="data/base_images/$DATASET/annotations_coco.json"
fi

# Generate ground truth COCO annotations if not present
if [ ! -f "$GT_COCO" ]; then
    GT_DIR="$(dirname "$GT_COCO")"
    GT_BB_DIR="${GT_DIR}/bounding_boxes"
    GT_LABELS_DIR="${GT_DIR}/labels"
    GT_IMAGES_DIR="${GT_DIR}/images"
    if [ -d "$GT_BB_DIR" ]; then
        echo "Ground truth COCO not found. Generating $GT_COCO from $GT_BB_DIR..."
        python3 analysis/utils/format_gt.py \
            --input_dir "$GT_BB_DIR" \
            --output_file "$GT_COCO" \
            --image_dir "$GT_IMAGES_DIR"
    elif [ -d "$GT_LABELS_DIR" ]; then
        echo "Ground truth COCO not found. Generating $GT_COCO from YOLO labels in $GT_LABELS_DIR..."
        python3 analysis/utils/format_detections.py \
            --detection_dir "$GT_LABELS_DIR" \
            --output_file "$GT_COCO" \
            --as_ground_truth \
            --yolo_format \
            --image_dir "$GT_IMAGES_DIR"
    fi
fi

process_rule_analysis() {
    local model_name="$1"
    local original_results="$2"
    local base_det="$3"
    local rule_path="$4"
    local rule_name="$5"
    local base_rule_name
    base_rule_name=$(resolve_base_rule_name "$rule_path" "$rule_name")
    local transform_params_json
    transform_params_json=$(resolve_transform_params_json "$base_rule_name" "$rule_name")

    if ! should_process_rule "$rule_name" && ! should_process_rule "$base_rule_name"; then
        return 0
    fi

    echo "=== Processing $model_name $rule_name detections ==="

    # Create comparison type subdirectories
    stats_dir_base="${RESULTS_BASE}/${model_name}/${rule_name}"
    if should_run_comparison original_vs_transformed; then
        mkdir -p "${stats_dir_base}/original_vs_transformed"
    fi
    if [ -f "$GT_COCO" ] && should_run_comparison transformed_vs_gt; then
        mkdir -p "${stats_dir_base}/transformed_vs_gt"
    fi

    # Format transformed detections
    python3 analysis/utils/format_detections.py \
        --detection_dir "$rule_path" \
        --output_file "${rule_path}/detections_coco.json" \
        --image_dir "$SOURCE_IMAGES_DIR"

    local -a color_wheel_args=()
    if [ "$base_rule_name" = "imgColorWheel" ]; then
        color_wheel_args=(--color-wheel-variant "$rule_name")
    fi

    # Comparison 1: Original detections vs Transformed detections (metamorphic testing)
    echo "Comparing original vs $rule_name (metamorphic)..."

    if is_geometric_transform "$base_rule_name"; then
        # For geometric transformations, transform coordinates before comparison
        echo "Transforming coordinates for geometric transformation..."
        transform_args=(
            --detection_file "${rule_path}/detections_coco.json"
            --transformation "$base_rule_name"
            --original_images "$original_results"
            --transformed_images "$rule_path"
            --output_file "${rule_path}/detections_coco_transformed.json"
        )
        if [ -n "$transform_params_json" ]; then
            transform_args+=(--transform_params "$transform_params_json")
        fi
        python3 analysis/utils/transform_geometric_detections.py "${transform_args[@]}"

        DT_FILE="${rule_path}/detections_coco_transformed.json"
    else
        DT_FILE="${rule_path}/detections_coco.json"
    fi

    if should_run_comparison original_vs_transformed; then
        python3 analysis/coco_eval.py \
            --gt "$base_det" \
            --dt "$DT_FILE" \
            --image-dir "$SOURCE_IMAGES_DIR" \
            --stats-out "${stats_dir_base}/original_vs_transformed/stats.json" "${color_wheel_args[@]}"
    fi


    # Comparison 3: Transformed detections vs Ground Truth (if GT available)
    if [ -f "$GT_COCO" ] && should_run_comparison transformed_vs_gt; then
        echo "Comparing $rule_name vs ground truth..."

        # Format detections with GT image IDs (strips transformation suffixes)
        python3 analysis/utils/format_detections.py \
            --detection_dir "$rule_path" \
            --output_file "${rule_path}/detections_for_gt.json" \
            --gt_coco "$GT_COCO" \
            --image_dir "$SOURCE_IMAGES_DIR"

        # For geometric transformations, transform coordinates
        if is_geometric_transform "$base_rule_name"; then
            echo "Transforming coordinates for GT comparison..."
            transform_gt_args=(
                --detection_file "${rule_path}/detections_for_gt.json"
                --transformation "$base_rule_name"
                --original_images "$original_results"
                --transformed_images "$rule_path"
                --output_file "${rule_path}/detections_for_gt_transformed.json"
            )
            if [ -n "$transform_params_json" ]; then
                transform_gt_args+=(--transform_params "$transform_params_json")
            fi
            python3 analysis/utils/transform_geometric_detections.py "${transform_gt_args[@]}"

            DT_GT_FILE="${rule_path}/detections_for_gt_transformed.json"
        else
            DT_GT_FILE="${rule_path}/detections_for_gt.json"
        fi

        python3 analysis/coco_eval.py \
            --gt "$GT_COCO" \
            --dt "$DT_GT_FILE" \
            --image-dir "$SOURCE_IMAGES_DIR" \
            --stats-out "${stats_dir_base}/transformed_vs_gt/stats.json" "${color_wheel_args[@]}" 2>&1 | grep -v "Warning: No matching image" || true
    fi
}

prepare_original_detections() {
    local model_name="$1"
    local original_results="$2"
    local base_det="${original_results}/detections_original_coco.json"

    # Format base detections (original images) - output as ground truth format
    echo "=== Processing $model_name original detections ==="
    python3 analysis/utils/format_detections.py \
        --detection_dir "$original_results" \
        --output_file "$base_det" \
        --as_ground_truth \
        --image_dir "$SOURCE_IMAGES_DIR" \
        --category_ids 1 2 3 4

    # Original detections vs Ground Truth (if GT available)
    if [ -f "$GT_COCO" ] && should_run_comparison original_vs_gt; then
        echo "Comparing original vs ground truth..."
        # Generate detections with GT image IDs (strips transformation suffixes)
        python3 analysis/utils/format_detections.py \
            --detection_dir "$original_results" \
            --output_file "${original_results}/detections_for_gt.json" \
            --gt_coco "$GT_COCO" \
            --image_dir "$SOURCE_IMAGES_DIR"
            
        stats_dir_base="${RESULTS_BASE}/${model_name}"
        python3 analysis/coco_eval.py \
            --gt "$GT_COCO" \
            --dt "${original_results}/detections_for_gt.json" \
            --image-dir "$SOURCE_IMAGES_DIR" \
            --stats-out "${stats_dir_base}/original_vs_gt/stats.json" 2>&1 | grep -v "Warning: No matching image" || true
        if should_process_rule imgColorWheel; then
            # Source-specific baselines for downstream colour-shift comparisons.
            python3 analysis/coco_eval.py \
                --gt "$GT_COCO" \
                --dt "${original_results}/detections_for_gt.json" \
                --image-dir "$SOURCE_IMAGES_DIR" \
                --color-wheel-variant imgColorWheel_0 \
                --stats-out "${stats_dir_base}/original_vs_gt/color_wheel_stats.json"
        fi
    fi

    BASE_DET="$base_det"
}

run_model_analysis() {
    local model_results="$1"   # e.g. results/yolov5
    local model_name
    model_name=$(basename "$model_results")
    local original_results="${model_results}/original"

    if [ ! -d "$original_results" ]; then
        return 0
    fi

    prepare_original_detections "$model_name" "$original_results"
    local base_det="$BASE_DET"

    for rule_path in "${model_results}"/*; do
        [ ! -d "$rule_path" ] && continue
        
        rule_name=$(basename "$rule_path")
        
        # Skip the original directory in the loop
        [ "$rule_name" = "original" ] && continue

        process_rule_analysis "$model_name" "$original_results" "$base_det" "$rule_path" "$rule_name"
    done
    
    echo "=== Completed $model_name analysis ==="
    echo ""
}

echo "======================================"
echo "Metamorphic Testing Analysis"
echo "======================================"
echo "Dataset: $DATASET"
if [ -n "$MODEL_FILTER" ]; then
    echo "Model: $MODEL_FILTER"
fi
if [ -n "$RULE_FILTER" ]; then
    echo "Rules: $RULE_FILTER"
fi
echo "Comparison: $COMPARISON_FILTER"
echo "Detections: $DETECTIONS_BASE/<model>/"
echo "Comparing original vs transformed detections"
echo ""

if [ -f "$GT_COCO" ]; then
    echo "Ground truth annotations found: $GT_COCO"
    echo "Will also compare against ground truth."
else
    echo "No ground truth annotations found at: $GT_COCO"
    echo "Only metamorphic comparison (original vs transformed) will run."
fi
echo ""

# Run analysis for the selected model, or all enabled models by default.
if [ -n "$MODEL_FILTER" ]; then
    MODELS_TO_ANALYZE="$MODEL_FILTER"
else
    MODELS_TO_ANALYZE="$ENABLED_MODELS"
fi

for model in $MODELS_TO_ANALYZE; do
    results_dir="$DETECTIONS_BASE/$model"
    run_model_analysis "$results_dir"
done
