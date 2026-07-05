#!/bin/bash
# Master script to generate all Venn diagram visualizations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "Venn Diagram Visualization Pipeline"
echo "=================================================="
echo ""

# Activate venv
source "$PROJECT_ROOT/venv/bin/activate"

DATASET_NAME="test"
# Check if data exists
RESULTS_DIR="$PROJECT_ROOT/results/$DATASET_NAME"
GT_PATH="$PROJECT_ROOT/data/base_images/$DATASET_NAME/annotations_coco.json"


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

echo "Step 1: Generating individual Venn diagrams..."
echo "-----------------------------------------------"
python3 "$PROJECT_ROOT/analysis/visualisation/plot_detection_venn.py" \
    --results_dir "$RESULTS_DIR" \
    --gt_path "$GT_PATH" \
    --output_dir "$PROJECT_ROOT/analysis/results/venn_diagrams/$DATASET_NAME"

echo ""
echo "=================================================="
echo "✓ All visualizations generated successfully!"
echo "=================================================="
echo ""
echo "Output files:"
echo "  - Individual diagrams: analysis/results/venn_diagrams/$DATASET_NAME/*_venn.png"
echo ""
