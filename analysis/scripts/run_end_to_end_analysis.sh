#!/usr/bin/env bash
# =============================================================================
# Run the complete metamorphic-analysis workflow for one or more datasets.
#
# This runs the general metamorphic analysis, cone-colour analysis, confidence
# change plots, detection-overlap Venn diagrams, and parameter-variation plots.
#
# Usage:
#   bash analysis/scripts/run_end_to_end_analysis.sh
#   bash analysis/scripts/run_end_to_end_analysis.sh --dataset test --dataset val
#   bash analysis/scripts/run_end_to_end_analysis.sh --datasets test,val
#   bash analysis/scripts/run_end_to_end_analysis.sh --skip-steps 2
#
# Options:
#   --dataset <name>       Select a dataset; may be repeated.
#   --datasets <names>     Comma-separated dataset list.
#   --skip-steps <n>       Skip the first n workflow steps (0-5).
#   --help, -h             Show this help text.
#
# Valid datasets: test, train_samples, val. The default is test.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

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
            test|train_samples|val)
                DATASETS+=("$dataset")
                ;;
            '')
                echo "Error: dataset names cannot be empty." >&2
                exit 2
                ;;
            *)
                echo "Error: invalid dataset '$dataset'. Choose: test, train_samples, val." >&2
                exit 2
                ;;
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

if [[ ! "$SKIP_STEPS" =~ ^[0-5]$ ]]; then
    echo "Error: --skip-steps must be an integer from 0 to 5." >&2
    exit 2
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(test)
fi

# Preserve order while avoiding repeated work when a dataset is supplied twice.
declare -A SEEN_DATASETS=()
for dataset in "${DATASETS[@]}"; do
    if [[ -n "${SEEN_DATASETS[$dataset]:-}" ]]; then
        continue
    fi
    SEEN_DATASETS[$dataset]=1

    results_dir="results/$dataset"
    analysis_dir="analysis/results/$dataset"
    gt_path="data/base_images/$dataset/annotations_coco.json"

    echo
    echo "=============================================================================="
    echo "End-to-end analysis: $dataset"
    echo "=============================================================================="

    if (( SKIP_STEPS < 1 )); then
        echo "[1/5] General metamorphic analysis"
        bash analysis/scripts/run_all_analysis_matias_meta.sh \
            --dataset "$dataset" \
            --detections-base "$results_dir"
    fi

    if (( SKIP_STEPS < 2 )); then
        echo "[2/5] Cone-colour analysis"
        bash analysis/scripts/run_analysis_cone_color.sh --dataset "$dataset"
    fi

    if (( SKIP_STEPS < 3 )); then
        echo "[3/5] Confidence-change plots"
        python3 analysis/visualisation/plot_confidence_change.py \
            --dataset "$dataset" \
            --results_dir "$results_dir" \
            --gt_path "$gt_path" \
            --output_dir "$analysis_dir/confidence_change"
    fi

    if (( SKIP_STEPS < 4 )); then
        echo "[4/5] Detection-overlap Venn diagrams"
        python3 analysis/visualisation/plot_detection_venn.py \
            --results_dir "$results_dir" \
            --gt_path "$gt_path" \
            --output_dir "$analysis_dir/venn_diagrams"
    fi

    if (( SKIP_STEPS < 5 )); then
        echo "[5/5] Parameter-variation plots"
        python3 analysis/visualisation/plot_parameter_variations.py \
            --dataset "$dataset" \
            --results_dir "$analysis_dir" \
            --output_dir "$analysis_dir/parameter_variations"
    fi

    echo "Completed $dataset: $analysis_dir"
done

echo
echo "End-to-end analysis complete."
