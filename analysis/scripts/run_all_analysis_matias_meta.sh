#!/bin/bash
# Run from repo root. Saves analysis stats to analysis/results/{model}/{rule}/{comparison_type}.json

# Format FSOCO ground truth
python analysis/utils/format_gt.py --input_dir data/base_images/test/labels --output_file data/base_images/test/labels/annotations_coco.json

GT_COCO="data/base_images/test/labels/annotations_coco.json"
RESULTS_BASE="analysis/results"

run_model_analysis() {
    local model_results="$1"   # e.g. results/yolov5
    local model_name
    model_name=$(basename "$model_results")
    local base_det="${model_results}/base_images/detections_original_coco.json"

    # Format base detections (original images)
    python analysis/utils/format_detections.py --detection_dir "${model_results}/base_images" --output_file "$base_det" --gt_coco "$GT_COCO"

    for rule_path in "${model_results}"/*; do
        [ ! -d "$rule_path" ] && continue
        rule_name=$(basename "$rule_path")
        stats_dir="${RESULTS_BASE}/${model_name}/${rule_name}"
        mkdir -p "$stats_dir"

        python analysis/utils/format_detections.py --detection_dir "$rule_path" --output_file "${rule_path}/detections_coco.json" --gt_coco "$GT_COCO"

        # original det vs rule det (reference: base detections, compare: this rule's detections)
        python analysis/coco_eval.py --gt "$base_det" --dt "${rule_path}/detections_coco.json" \
            --stats-out "${stats_dir}/original_vs_rule.json"

        # original vs transformed (reference: rule det on original, compare: rule det on transformed)
        if [ -f "${rule_path}/detections_transformed_coco.json" ]; then
            python analysis/coco_eval.py --gt "${rule_path}/detections_coco.json" --dt "${rule_path}/detections_transformed_coco.json" \
                --stats-out "${stats_dir}/original_vs_transformed.json"
        fi

        # transformed vs gt (reference: ground truth, compare: detections on transformed images)
        if [ -f "${rule_path}/detections_transformed_coco.json" ]; then
            python analysis/coco_eval.py --gt "$GT_COCO" --dt "${rule_path}/detections_transformed_coco.json" \
                --stats-out "${stats_dir}/transformed_vs_gt.json"
        fi
    done
}

run_model_analysis results/yolov5
run_model_analysis results/yolov8
run_model_analysis results/yolov12
