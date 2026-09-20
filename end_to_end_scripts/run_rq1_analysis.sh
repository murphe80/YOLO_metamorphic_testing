#!/usr/bin/env bash
# =============================================================================
# Run the complete metamorphic-analysis workflow for one or more datasets.
#
# This runs the general metamorphic analysis, cone-colour analysis, confidence
# change plots, detection-overlap metrics, and parameter-variation plots.
#
# Usage:
#   bash end_to_end_scripts/run_rq1_analysis.sh
#   bash end_to_end_scripts/run_rq1_analysis.sh --dataset test --dataset val
#   bash end_to_end_scripts/run_rq1_analysis.sh --datasets test,val
#   bash end_to_end_scripts/run_rq1_analysis.sh --skip-steps 2
#
# Options:
#   --dataset <name>       Select a dataset; may be repeated.
#   --datasets <names>     Comma-separated dataset list.
#   --skip-steps <n>       Skip the first n workflow steps (0-7).
#   --help, -h             Show this help text.
#
# Dataset names resolve beneath data/base_images/. The default is fsoco_sample.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

declare -a DATASETS=()
SKIP_STEPS=0

usage() {
    sed -n '3,20p' "$0"
}

add_datasets() {
    local csv="$1"
    local dataset
    local -a requested=()
    IFS=',' read -r -a requested <<< "$csv"

    for dataset in "${requested[@]}"; do
        case "$dataset" in
            '')
                echo "Error: dataset names cannot be empty." >&2
                exit 2
                ;;
            *[!A-Za-z0-9._-]*|[._-]*)
                echo "Error: invalid dataset name '$dataset'." >&2
                exit 2
                ;;
            *) DATASETS+=("$dataset") ;;
        esac
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || { echo "Error: --dataset requires a value." >&2; exit 2; }
            add_datasets "$2"
            shift 2
            ;;
        --dataset=*)
            add_datasets "${1#*=}"
            shift
            ;;
        --datasets)
            [[ $# -ge 2 ]] || { echo "Error: --datasets requires a value." >&2; exit 2; }
            add_datasets "$2"
            shift 2
            ;;
        --datasets=*)
            add_datasets "${1#*=}"
            shift
            ;;
        --skip-steps)
            [[ $# -ge 2 ]] || { echo "Error: --skip-steps requires a value." >&2; exit 2; }
            SKIP_STEPS="$2"
            shift 2
            ;;
        --skip-steps=*)
            SKIP_STEPS="${1#*=}"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'." >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$SKIP_STEPS" =~ ^[0-7]$ ]]; then
    echo "Error: --skip-steps must be an integer from 0 to 7." >&2
    exit 2
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(fsoco_sample)
fi

# Preserve order while avoiding repeated work when a dataset is supplied twice.
declare -A SEEN_DATASETS=()
for dataset in "${DATASETS[@]}"; do
    if [[ -n "${SEEN_DATASETS[$dataset]:-}" ]]; then
        continue
    fi
    SEEN_DATASETS[$dataset]=1

    results_dir="results/$dataset"
    analysis_dir="analysis/results/rq1/$dataset"
    gt_path="data/base_images/$dataset/annotations_coco.json"

    echo
    echo "=============================================================================="
    echo "End-to-end analysis: $dataset"
    echo "=============================================================================="

    if (( SKIP_STEPS < 1 )); then
        echo "[1/7] General metamorphic analysis"
        bash analysis/scripts/run_base_metamorphic_analysis.sh \
            --dataset "$dataset" \
            --detections-base "$results_dir"
    fi

    if (( SKIP_STEPS < 2 )); then
        echo "[2/7] Cone-colour analysis"
        bash analysis/scripts/run_cone_colour_analysis.sh --dataset "$dataset"
    fi

    if (( SKIP_STEPS < 3 )); then
        echo "[3/7] Cone-colour summary plots"
        python3 analysis/visualisation/plot_cone_colour_analysis.py --datasets "$dataset"
    fi

    if (( SKIP_STEPS < 4 )); then
        echo "[4/7] Confidence-change plots"
        python3 analysis/visualisation/plot_confidence_change.py \
            --dataset "$dataset" \
            --results_dir "$results_dir" \
            --gt_path "$gt_path" \
            --output_dir "$analysis_dir/confidence_change"
    fi

    if (( SKIP_STEPS < 5 )); then
        echo "[5/7] Detection-overlap metrics"
        python3 analysis/utils/calculate_detection_overlap_metrics.py \
            --dataset "$dataset" \
            --results_dir "$results_dir" \
            --gt_path "$gt_path" \
            --output_path "$analysis_dir/detection_overlap_metrics/detection_overlap_metrics.json"
    fi

    if (( SKIP_STEPS < 6 )); then
        echo "[6/7] Detection-overlap plots"
        python3 analysis/visualisation/plot_detection_overlap_metrics.py \
            --dataset "$dataset" \
            --statistics "$analysis_dir/detection_overlap_metrics/detection_overlap_metrics.json" \
            --output-dir "$analysis_dir/detection_overlap_metrics/plots"
    fi

    if (( SKIP_STEPS < 7 )); then
        echo "[7/7] Parameter-variation plots"
        python3 analysis/visualisation/plot_parameter_variations.py \
            --dataset "$dataset" \
            --results_dir "$analysis_dir" \
            --output_dir "$analysis_dir/parameter_variations"
    fi

    echo "Completed $dataset: $analysis_dir"
done

echo
echo "End-to-end analysis complete."
