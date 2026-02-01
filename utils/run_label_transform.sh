#!/usr/bin/env bash

set -e  # fail fast
set -u  # undefined vars are errors

STACKED_IMAGES_DIR="../data/base_images/segmented_fsoco_test_gt"
LABELS_DIR="../data/base_images/test/labels"
OUTPUT_DIR="../data/transformed_images/fsoco_test_segmented/label_transformations"

SCRIPT_PATH="cone_label_transformation_bb.py"

if [[ ! -d "$STACKED_IMAGES_DIR" ]]; then
  echo "ERROR: stacked images dir not found: $STACKED_IMAGES_DIR"
  exit 1
fi

if [[ ! -d "$LABELS_DIR" ]]; then
  echo "ERROR: labels dir not found: $LABELS_DIR"
  exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: script not found: $SCRIPT_PATH"
  exit 1
fi

# -----------------------------
# Run for all modes
# -----------------------------

for MODE in avg white black; do
    MODE_OUTPUT_DIR="$OUTPUT_DIR/$MODE"
    mkdir -p "$MODE_OUTPUT_DIR"

    echo
    echo "=============================================="
    echo "Running GT cone label removal"
    echo "Mode           : $MODE"
    echo "Stacked images : $STACKED_IMAGES_DIR"
    echo "Labels         : $LABELS_DIR"
    echo "Output         : $MODE_OUTPUT_DIR"
    echo "=============================================="

    python3 "$SCRIPT_PATH" \
        --stacked_images_dir "$STACKED_IMAGES_DIR" \
        --labels_dir "$LABELS_DIR" \
        --output_dir "$MODE_OUTPUT_DIR" \
        --transform_mode "$MODE"

done

echo
echo "All modes finished. Results saved under: $OUTPUT_DIR"
