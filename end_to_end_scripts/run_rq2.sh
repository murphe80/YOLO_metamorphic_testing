#!/usr/bin/env bash
# Run RQ2a, RQ2b, and RQ2c in sequence, while retaining each Python program as
# a separately runnable step under scripts/rq2/.

# Usage:
#   bash end_to_end_scripts/run_rq2.sh [OPTIONS]

# Options:
#   --dataset <name>            Dataset under data/base_images (default: fsoco_sample)
#   --models <all|a,b,...>      Models to process (default: all enabled models)
#   --target-ap <value>         Accuracy threshold for RQ2a (default: 0.5)
#   --num-images <n>            Limit images for a sample/smoke experiment
#   --device <device>           Torch device (default: cuda:0)
#   --workers <n>               RQ2b scoring workers (default: 1)
#   --search-method <method>    RQ2a source for RQ2b/c: brute_force or
#                               golden_section (default: brute_force)
#   --start-at <rq2a|rq2b|rq2c> Resume from a later research-question step
#   --help, -h                  Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi
source utils/model_registry.sh

DATASET="fsoco_sample"
MODELS_CSV="all"
TARGET_AP="0.5"
NUM_IMAGES=""
DEVICE="cuda:0"
WORKERS=1
SEARCH_METHOD="brute_force"
START_AT="rq2a"

usage() { sed -n '2,21p' "$0"; }
require_value() {
    if [ -z "${2:-}" ] || [[ "${2:-}" == --* ]]; then
        echo "Error: $1 requires a value." >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset|--models|--target-ap|--num-images|--device|--workers|--search-method|--start-at)
            require_value "$1" "${2:-}"
            case "$1" in
                --dataset) DATASET="$2" ;;
                --models) MODELS_CSV="$2" ;;
                --target-ap) TARGET_AP="$2" ;;
                --num-images) NUM_IMAGES="$2" ;;
                --device) DEVICE="$2" ;;
                --workers) WORKERS="$2" ;;
                --search-method) SEARCH_METHOD="$2" ;;
                --start-at) START_AT="$2" ;;
            esac
            shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Error: unknown option '$1'." >&2; usage >&2; exit 2 ;;
    esac
done

case "$SEARCH_METHOD" in brute_force|golden_section) ;; *) echo "Invalid --search-method" >&2; exit 2 ;; esac
case "$START_AT" in rq2a) START_INDEX=1 ;; rq2b) START_INDEX=2 ;; rq2c) START_INDEX=3 ;; *) echo "Invalid --start-at" >&2; exit 2 ;; esac

if [ "$MODELS_CSV" = "all" ]; then
    read -r -a MODELS <<< "$(get_model_names)"
else
    IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
fi
if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "Error: no models were selected." >&2
    exit 2
fi

image_args=()
if [ -n "$NUM_IMAGES" ]; then image_args=(--num-images "$NUM_IMAGES"); fi

if (( START_INDEX <= 1 )); then
    echo "[RQ2 1/3] RQ2a search experiments and comparison"
    for model in "${MODELS[@]}"; do
        python3 scripts/rq2/rq2a_brute_force_search.py --dataset "$DATASET" --model "$model" --target-ap "$TARGET_AP" --device "$DEVICE" "${image_args[@]}"
        python3 scripts/rq2/rq2a_golden_section_search.py --dataset "$DATASET" --model "$model" --target-ap "$TARGET_AP" --device "$DEVICE" "${image_args[@]}"
        python3 scripts/rq2/rq2a_compare_results.py \
            --brute-force "analysis/results/rq2/rq2a/brute_force/${model}/results.json" \
            --golden-section "analysis/results/rq2/rq2a/golden_section/${model}/results.json"
    done
fi

if (( START_INDEX <= 2 )); then
    echo "[RQ2 2/3] RQ2b brightness metrics"
    python3 scripts/rq2/rq2b_brightness_calculation.py --model "${MODELS[@]}" --search-method "$SEARCH_METHOD" --workers "$WORKERS" "${image_args[@]}"
fi

echo "[RQ2 3/3] RQ2c performance distributions"
python3 scripts/rq2/rq2c_perf_comparison.py --models "${MODELS[@]}" --search-method "$SEARCH_METHOD" "${image_args[@]}"

echo "RQ2 end-to-end workflow complete."
