#!/bin/bash
# Master script to generate all Venn diagram visualizations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "Venn Diagram Visualization Pipeline"
echo "=================================================="
echo ""

# Check if venv exists
if [ ! -d "$PROJECT_ROOT/venv_venn" ]; then
    echo "Virtual environment not found. Creating..."
    bash "$SCRIPT_DIR/setup_venn_env.sh"
fi

# Activate venv
source "$PROJECT_ROOT/venv_venn/bin/activate"

# Check if data exists
RESULTS_DIR="$PROJECT_ROOT/results"
GT_PATH="$PROJECT_ROOT/data/base_images/fsoco_sample/annotations_coco.json"

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
    --output_dir "$PROJECT_ROOT/analysis/results/venn_diagrams"

echo ""
echo "Step 2: Generating grid visualization..."
echo "-----------------------------------------------"
python3 "$PROJECT_ROOT/analysis/visualisation/plot_venn_grid.py" \
    --results_dir "$RESULTS_DIR" \
    --gt_path "$GT_PATH" \
    --output "$PROJECT_ROOT/analysis/results/venn_diagrams/all_models_grid.png"

echo ""
echo "Step 3: Exporting statistics to CSV..."
echo "-----------------------------------------------"
python3 "$PROJECT_ROOT/analysis/utils/venn_statistics_analysis.py" \
    --input "$PROJECT_ROOT/analysis/results/venn_diagrams/venn_statistics.json" \
    --output "$PROJECT_ROOT/analysis/results/venn_diagrams/venn_statistics.csv" \
    --summary

echo ""
echo "=================================================="
echo "✓ All visualizations generated successfully!"
echo "=================================================="
echo ""
echo "Output files:"
echo "  - Individual diagrams: analysis/results/venn_diagrams/*_venn.png"
echo "  - Grid view: analysis/results/venn_diagrams/all_models_grid.png"
echo "  - Statistics JSON: analysis/results/venn_diagrams/venn_statistics.json"
echo "  - Statistics CSV: analysis/results/venn_diagrams/venn_statistics.csv"
echo ""
