#!/bin/bash
# ============================================
# run_all_analysis_matias_meta.sh
#
# Analyze metamorphic testing results by comparing original vs transformed detections.
# Optionally compares against ground truth if available.
#
# Usage:
#   bash analysis/scripts/run_all_analysis_matias_meta.sh [--dataset fsoco_sample|test]
#
# Outputs: analysis/results/<model>/<rule>/<comparison_type>/stats.json
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

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --dataset=*) DATASET="${1#*=}"; shift ;;
        --rules) RULE_FILTER="$2"; shift 2 ;;
        --rules=*) RULE_FILTER="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================
# Configuration
# ============================================
RESULTS_BASE="analysis/results"
BASE_TRANSFORMATIONS=(
    imgStretch
    imgResolution
    imgMotionBlur
    imgDarkBright
    imgGrayscale
    imgColorWheel
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

run_model_analysis() {
    local model_results="$1"   # e.g. results/yolov5
    local model_name
    model_name=$(basename "$model_results")
    local base_det="${model_results}/original/detections_original_coco.json"

    # Format base detections (original images) - output as ground truth format
    echo "=== Processing $model_name original detections ==="
    python3 analysis/utils/format_detections.py \
        --detection_dir "${model_results}/original" \
        --output_file "$base_det" \
        --as_ground_truth

    # Original detections vs Ground Truth (if GT available)
    if [ -f "$GT_COCO" ]; then
        echo "Comparing original vs ground truth..."
        # Generate detections with GT image IDs (strips transformation suffixes)
        python3 analysis/utils/format_detections.py \
            --detection_dir "${model_results}/original" \
            --output_file "${model_results}/original/detections_for_gt.json" \
            --gt_coco "$GT_COCO"
            
        stats_dir_base="${RESULTS_BASE}/${model_name}"
        python3 analysis/coco_eval.py \
            --gt "$GT_COCO" \
            --dt "${model_results}/original/detections_for_gt.json" \
            --stats-out "${stats_dir_base}/original_vs_gt/stats.json" 2>&1 | grep -v "Warning: No matching image" || true
    fi

    for rule_path in "${model_results}"/*; do
        [ ! -d "$rule_path" ] && continue
        
        rule_name=$(basename "$rule_path")
        base_rule_name="$rule_name"
        
        # Skip the original directory in the loop
        [ "$rule_name" = "original" ] && continue
        should_process_rule "$rule_name" || continue

        if [ -f "$rule_path/configs.json" ]; then
            detected_rule=$(python3 - "$rule_path/configs.json" <<'PY'
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

with open(sys.argv[1]) as f:
    config = json.load(f)

for key in config:
    if key in known_rules:
        print(key)
        break
PY
)
            if [ -n "$detected_rule" ]; then
                base_rule_name="$detected_rule"
            fi
        else
            for known_rule in "${BASE_TRANSFORMATIONS[@]}"; do
                case "$rule_name" in
                    "$known_rule"|"$known_rule"_*)
                        base_rule_name="$known_rule"
                        break
                        ;;
                esac
            done
        fi
        
        echo "=== Processing $model_name $rule_name detections ==="
        
        # Create comparison type subdirectories
        stats_dir_base="${RESULTS_BASE}/${model_name}/${rule_name}"
        mkdir -p "${stats_dir_base}/original_vs_transformed"
        mkdir -p "${stats_dir_base}/transformed_vs_gt"

        # Format transformed detections
        python3 analysis/utils/format_detections.py \
            --detection_dir "$rule_path" \
            --output_file "${rule_path}/detections_coco.json"

        # Comparison 1: Original detections vs Transformed detections (metamorphic testing)
        echo "Comparing original vs $rule_name (metamorphic)..."
        
        GEOMETRIC_TRANSFORMS=("imgStretch" "imgResolution")
        if [[ " ${GEOMETRIC_TRANSFORMS[@]} " =~ " ${base_rule_name} " ]]; then
            # For geometric transformations, transform coordinates before comparison
            echo "Transforming coordinates for geometric transformation..."
            python3 analysis/utils/transform_geometric_detections.py \
                --detection_file "${rule_path}/detections_coco.json" \
                --transformation "$base_rule_name" \
                --original_images "${model_results}/original" \
                --transformed_images "$rule_path" \
                --output_file "${rule_path}/detections_coco_transformed.json"
            
            DT_FILE="${rule_path}/detections_coco_transformed.json"
        else
            DT_FILE="${rule_path}/detections_coco.json"
        fi
        
        python3 analysis/coco_eval.py \
            --gt "$base_det" \
            --dt "$DT_FILE" \
            --stats-out "${stats_dir_base}/original_vs_transformed/stats.json"
    
        
        # Comparison 3: Transformed detections vs Ground Truth (if GT available)
        if [ -f "$GT_COCO" ]; then
            echo "Comparing $rule_name vs ground truth..."
            
            # Format detections with GT image IDs (strips transformation suffixes)
            python3 analysis/utils/format_detections.py \
                --detection_dir "$rule_path" \
                --output_file "${rule_path}/detections_for_gt.json" \
                --gt_coco "$GT_COCO"
            
            # For geometric transformations, transform coordinates
            if [[ " ${GEOMETRIC_TRANSFORMS[@]} " =~ " ${base_rule_name} " ]]; then
                echo "Transforming coordinates for GT comparison..."
                python3 analysis/utils/transform_geometric_detections.py \
                    --detection_file "${rule_path}/detections_for_gt.json" \
                    --transformation "$base_rule_name" \
                    --original_images "${model_results}/original" \
                    --transformed_images "$rule_path" \
                    --output_file "${rule_path}/detections_for_gt_transformed.json"
                
                DT_GT_FILE="${rule_path}/detections_for_gt_transformed.json"
            else
                DT_GT_FILE="${rule_path}/detections_for_gt.json"
            fi
            
            python3 analysis/coco_eval.py \
                --gt "$GT_COCO" \
                --dt "$DT_GT_FILE" \
                --stats-out "${stats_dir_base}/transformed_vs_gt/stats.json" 2>&1 | grep -v "Warning: No matching image" || true
        fi
    done
    
    echo "=== Completed $model_name analysis ==="
    echo ""
}

echo "======================================"
echo "Metamorphic Testing Analysis"
echo "======================================"
echo "Dataset: $DATASET"
if [ -n "$RULE_FILTER" ]; then
    echo "Rules: $RULE_FILTER"
fi
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

# Run analysis for all enabled models
for model in $(get_model_names); do
    results_dir=$(get_model_field "$model" "results_dir")
    run_model_analysis "$results_dir"
done
