#!/bin/bash

# Shell script to extract poorly segmented images caused by overly bright/dark lighting conditions
# Iterate through each file in fsoco_test_segmented/all_images and if image is present 
# and not present in /transformed_images/fsoco_test_segmented/remove_labels, 
# add to new subdirectory in fsoco_test_segmented/poorly_segmented_images

# Define directories
ALL_IMAGES_DIR="../../data/base_images/fsoco_test_segmented/all_images"
FILTERED_DIR="../../data/base_images/fsoco_test_segmented/filtered_segmented"
OUTPUT_DIR="../../data/base_images/fsoco_test_segmented/poorly_segmented_images"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Counter for copied files
count=0

# Iterate through each file in all_images directory
for image_file in "$ALL_IMAGES_DIR"/*; do
    # Check if it's a file (not a directory)
    if [ -f "$image_file" ]; then
        # Get the basename of the file
        filename=$(basename "$image_file")
        
        # Check if the file does NOT exist in remove_labels directory
        if [ ! -f "$FILTERED_DIR/$filename" ]; then
            # Copy the file to poorly_segmented_images directory
            cp "$image_file" "$OUTPUT_DIR/"
            echo "Copied: $filename"
            ((count++))
        fi
    fi
done

echo "Done! Copied $count poorly segmented images to $OUTPUT_DIR"