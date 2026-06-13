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

# ============================================
# Parse arguments
# ============================================
DATASET="fsoco_sample"  # Default to sample dataset

for arg in "$@"; do
    case $arg in
        --dataset) DATASET="${2:-fsoco_sample}"; shift ;;
        --dataset=*) DATASET="${arg#*=}" ;;
    esac
done

# ============================================
# Configuration
# ============================================
RESULTS_BASE="analysis/results"

# Set ground truth path based on dataset
if [ "$DATASET" = "test" ]; then
    GT_COCO="data/base_images/test/annotations_coco.json"
elif [ "$DATASET" = "fsoco_sample" ]; then
    GT_COCO="data/base_images/fsoco_sample/annotations_coco.json"
else
    GT_COCO="data/base_images/$DATASET/annotations_coco.json"
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
        
        # Skip the original directory in the loop
        [ "$rule_name" = "original" ] && continue
        
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
        if [[ " ${GEOMETRIC_TRANSFORMS[@]} " =~ " ${rule_name} " ]]; then
            # For geometric transformations, transform coordinates before comparison
            echo "Transforming coordinates for geometric transformation..."
            python3 analysis/utils/transform_geometric_detections.py \
                --detection_file "${rule_path}/detections_coco.json" \
                --transformation "$rule_name" \
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
            if [[ " ${GEOMETRIC_TRANSFORMS[@]} " =~ " ${rule_name} " ]]; then
                echo "Transforming coordinates for GT comparison..."
                python3 analysis/utils/transform_geometric_detections.py \
                    --detection_file "${rule_path}/detections_for_gt.json" \
                    --transformation "$rule_name" \
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

run_model_analysis results/yolov5
run_model_analysis results/yolov8
run_model_analysis results/yolov12

