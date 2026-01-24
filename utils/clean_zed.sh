#!/bin/bash

# Script to delete every fifth image matching pattern frame_XXXXX.png

DIRECTORY="../data/base_images/zed_frames"

# Check if directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory '$DIRECTORY' does not exist"
    exit 1
fi

# Change to the directory
cd "$DIRECTORY" || exit 1

# Get all matching images and sort them
mapfile -t images < <(ls -1 frame_[0-9][0-9][0-9][0-9][0-9].png 2>/dev/null | sort)

# Check if any images were found
if [ ${#images[@]} -eq 0 ]; then
    echo "No images matching pattern frame_XXXXX.png found in $DIRECTORY"
    exit 1
fi

echo "Found ${#images[@]} images in $DIRECTORY"
echo ""

# Preview which images will be deleted
echo "The following images will be deleted (every 5th image):"
count=0
for ((i=4; i<${#images[@]}; i+=5)); do
    echo "  ${images[$i]}"
    ((count++))
done

echo ""
echo "Total images to delete: $count"
echo "Remaining images: $((${#images[@]} - count))"
echo ""

# Ask for confirmation
read -p "Continue with deletion? (yes/no): " response

if [ "$response" != "yes" ]; then
    echo "Operation cancelled."
    exit 0
fi

# Delete every fifth image
deleted=0
for ((i=4; i<${#images[@]}; i+=5)); do
    if rm "${images[$i]}"; then
        echo "Deleted: ${images[$i]}"
        ((deleted++))
    else
        echo "Error deleting: ${images[$i]}"
    fi
done

echo ""
echo "Deletion complete!"
echo "Total images deleted: $deleted"