#!/usr/bin/env bash
# Run the complete RQ1 workflow: base metamorphic experiments, cone-colour
# experiments, and all analysis/plots.

# Usage:
#   bash end_to_end_scripts/run_rq1.sh [OPTIONS]

# Options:
#   --dataset <name[,name...]>  Dataset(s) to process (default: fsoco_sample)
#   --model <version|all>       Model version accepted by the experiment scripts
#   --viz-samples <n>           Detection visualisations per result directory
#   --skip-transform            Reuse existing transformed images
#   --help, -h                  Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATASETS="fsoco_sample"
MODEL="all"
VIZ_SAMPLES=5
SKIP_TRANSFORM=false

usage() { sed -n '2,14p' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset|--datasets) DATASETS="$2"; shift 2 ;;
        --dataset=*|--datasets=*) DATASETS="${1#*=}"; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --model=*) MODEL="${1#*=}"; shift ;;
        --viz-samples) VIZ_SAMPLES="$2"; shift 2 ;;
        --viz-samples=*) VIZ_SAMPLES="${1#*=}"; shift ;;
        --skip-transform) SKIP_TRANSFORM=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Error: unknown option '$1'." >&2; usage >&2; exit 2 ;;
    esac
done

common_args=(--dataset "$DATASETS" --model "$MODEL" --viz-samples "$VIZ_SAMPLES" --skip-analysis)
if [ "$SKIP_TRANSFORM" = true ]; then
    common_args+=(--skip-transform)
fi

echo "[RQ1 1/3] Base metamorphic experiments"
bash end_to_end_scripts/run_rq1_base_experiment.sh "${common_args[@]}"

echo "[RQ1 2/3] Cone-colour experiments"
bash end_to_end_scripts/run_rq1_cone_colour_experiment.sh "${common_args[@]}"

echo "[RQ1 3/3] Analysis and plots"
bash end_to_end_scripts/run_rq1_analysis.sh --datasets "$DATASETS"

echo "RQ1 end-to-end workflow complete."
