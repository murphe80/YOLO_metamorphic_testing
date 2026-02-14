#!/bin/bash
# Run from repo root. Saves analysis stats to analysis/results/{model}/{rule}/{comparison_type}/stats.json
# Compares original vs transformed detections (metamorphic testing)

RESULTS_BASE="analysis/results"
GT_COCO="data/base_images/test/annotations_coco.json"  # path to ground truth

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

    for rule_path in "${model_results}"/*; do
        [ ! -d "$rule_path" ] && continue
        
        rule_name=$(basename "$rule_path")
        
        # Skip the original directory in the loop
        [ "$rule_name" = "original" ] && continue
        
        echo "=== Processing $model_name $rule_name detections ==="
        
        # Create comparison type subdirectories
        stats_dir_base="${RESULTS_BASE}/${model_name}/${rule_name}"
        mkdir -p "${stats_dir_base}/original_vs_transformed"
        mkdir -p "${stats_dir_base}/original_vs_gt"
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
        
        # Comparison 2: Original detections vs Ground Truth (if GT available)
        # Only run once per model (not per transformation)
        if [ -f "$GT_COCO" ] && [ ! -f "${stats_dir_base}/original_vs_gt/stats.json" ]; then
            echo "Comparing original vs ground truth..."
            # Generate detections with GT image IDs (strips transformation suffixes)
            python3 analysis/utils/format_detections.py \
                --detection_dir "${model_results}/original" \
                --output_file "${model_results}/original/detections_for_gt.json" \
                --gt_coco "$GT_COCO"
            
            python3 analysis/coco_eval.py \
                --gt "$GT_COCO" \
                --dt "${model_results}/original/detections_for_gt.json" \
                --stats-out "${stats_dir_base}/original_vs_gt/stats.json" 2>&1 | grep -v "Warning: No matching image" || true
        fi
        
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

echo "Starting metamorphic testing analysis..."
echo "Comparing original vs transformed detections"
echo ""
echo "Note: GT comparisons will only include images that exist in the ground truth."
echo "      Images not in GT will be skipped (this is expected if using different test sets)."
echo ""

run_model_analysis results/yolov5
run_model_analysis results/yolov8
run_model_analysis results/yolov12

