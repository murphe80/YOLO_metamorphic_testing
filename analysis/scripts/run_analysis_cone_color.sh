#!/bin/bash
# Run from repo root. Saves analysis stats to analysis/results/{model}/cone_color/{rule}/{comparison_type}/stats.json
# Compares original vs transformed detections (metamorphic testing)

# Ensure we are in repo root (when run as standalone script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

RESULTS_BASE="analysis/results"

# format ground truth to coco format
python3 analysis/utils/format_gt.py \
    --input_dir "data/base_images/test/bounding_boxes" \
    --output_file "data/base_images/test/annotations_coco.json" \
    --image_dir "data/base_images/test/images"

GT_COCO="data/base_images/test/annotations_coco.json"

run_model_analysis() {
    local model_results="$1"   # e.g. results/yolov5/cone_color
    local model_name
    model_name=$(basename "$(dirname "$model_results")")   # e.g. yolov5 (so results don't overwrite across models)
    
    # format original detections
    python3 analysis/utils/format_detections.py \
        --detection_dir "${model_results}/original" \
        --output_file "${model_results}/original/detections_original_coco.json" \
        --as_ground_truth \

    for rule_path in "${model_results}"/*; do
        [ ! -d "$rule_path" ] && continue
        
        rule_name=$(basename "$rule_path")

        # Skip 'original' folder for metamorphic comparison (no transformation to apply)
        if [ "$rule_name" = "original" ]; then
            continue
        fi

        echo "=== Processing $model_name $rule_name detections ==="
        
        # Create comparison type subdirectories
        stats_dir_base="${RESULTS_BASE}/${model_name}/${rule_name}"
        mkdir -p "${stats_dir_base}/original_vs_transformed"
        mkdir -p "${stats_dir_base}/transformed_vs_gt"

        # Format transformed detections (use original COCO image ids / names so ids match GT)
        python3 analysis/utils/format_detections.py \
            --detection_dir "$rule_path" \
            --output_file "${rule_path}/detections_coco.json" \
            --gt_coco "${model_results}/original/detections_original_coco.json"

        # Comparison 1: Original detections vs Transformed detections (metamorphic testing)
        echo "Comparing original vs $rule_name (metamorphic)..."

        # Update original detections to expected cone color after transformation (this is the "expected" GT for coco_eval)
        GT_EXPECTED="${rule_path}/detections_coco_${rule_name}.json"
        DT_FILE="${rule_path}/detections_coco.json"
        python3 analysis/utils/cone_color_analysis.py \
            --ground_truth_file "${model_results}/original/detections_original_coco.json" \
            --transformation "$rule_name" \
            --output_path "$GT_EXPECTED"

        python3 analysis/coco_eval.py \
            --gt "$GT_EXPECTED" \
            --dt "$DT_FILE" \
            --stats-out "${stats_dir_base}/original_vs_transformed/stats.json"
        
        # delete GT_EXPECTED file after evaluation
        rm "$GT_EXPECTED"
        
        # Comparison 2: Transformed detections vs Ground Truth
        echo "Comparing transformed vs ground truth..."

        # Update GT to expected cone color after transformation (this is the "expected" GT for coco_eval)
        GT_EXPECTED="${rule_path}/ground_truth_coco_${rule_name}.json"
        python3 analysis/utils/cone_color_analysis.py \
            --ground_truth_file "$GT_COCO" \
            --transformation "$rule_name" \
            --output_path "$GT_EXPECTED"

        python3 analysis/coco_eval.py \
            --gt "$GT_EXPECTED" \
            --dt "$DT_FILE" \
            --stats-out "${stats_dir_base}/transformed_vs_gt/stats.json"
        
        # delete GT_EXPECTED file after evaluation
        rm "$GT_EXPECTED"
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

for model_results in results/yolov5/cone_color results/yolov8/cone_color results/yolov12/cone_color; do
    if [ -d "$model_results" ] && [ -d "${model_results}/original" ]; then
        run_model_analysis "$model_results"
    else
        echo "Skipping $model_results (not found or no original/ subdir)"
    fi
done

