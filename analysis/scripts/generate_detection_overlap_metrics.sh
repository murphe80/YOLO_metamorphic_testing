#!/bin/bash
# Calculate detection-overlap metrics for the default test dataset.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=================================================="
echo "Detection Overlap Metrics"
echo "=================================================="
echo ""

if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

DATASET_NAME="test"
RESULTS_DIR="$PROJECT_ROOT/results/$DATASET_NAME"
GT_PATH="$PROJECT_ROOT/data/base_images/$DATASET_NAME/annotations_coco.json"
OUTPUT_PATH="$PROJECT_ROOT/analysis/results/rq1/$DATASET_NAME/detection_overlap_metrics/detection_overlap_metrics.json"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    echo "Please run YOLO inference first to generate detection files."
    exit 1
fi

if [ ! -f "$GT_PATH" ]; then
    echo "ERROR: Ground truth file not found: $GT_PATH"
    echo "Please ensure the ground truth annotations exist."
    exit 1
fi

python3 "$PROJECT_ROOT/analysis/utils/calculate_detection_overlap_metrics.py" \
    --dataset "$DATASET_NAME" \
    --results-dir "$RESULTS_DIR" \
    --gt-path "$GT_PATH" \
    --output-path "$OUTPUT_PATH"

echo ""
echo "=================================================="
echo "✓ Statistics generated successfully"
echo "=================================================="
echo "Output: analysis/results/rq1/$DATASET_NAME/detection_overlap_metrics/detection_overlap_metrics.json"
