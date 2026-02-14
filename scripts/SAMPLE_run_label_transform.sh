#!/bin/bash
# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# script to run the backprojection and label transformation pipeline using FSOCO sample images

SAMPLE_DIR="data/base_images/fsoco_sample"

# run the backprojection pipeline using the FSOCO sample images
python utils/extract_cone_regions_gt.py \
    --images_dir "$SAMPLE_DIR/images" \
    --labels_dir "$SAMPLE_DIR/bounding_boxes" \
    --hist_samples_dir "utils/hist_backprojection" \
    --output_dir "$SAMPLE_DIR/segmented_images"

# run the label transformation pipeline using the FSOCO sample images
python utils/cone_label_transformation_bb.py \
    --stacked_images_dir "$SAMPLE_DIR/segmented_images" \
    --labels_dir "$SAMPLE_DIR/bounding_boxes" \
    --output_dir "data/transformed_images/fsoco_sample_segmented"

