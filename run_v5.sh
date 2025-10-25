#!/bin/bash
set -e  # Exit immediately on error

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate metamorphic images
for image in data/base_images/fsoco_sample/images/*; do
    python models/metamorphic_test.py \
        --rules imgGrayscale \
        --base_image "$image" \
        --output_path ./data/transformed_images/fsoco_sample
done

# Run original images on YOLOv5
for image in data/base_images/fsoco_sample/images/*; do
    python models/yolov5/infer_v5.py \
        --weights ./models/yolov5/yolov5_parameters.pt \
        --image "$image"
done

# Run transformed images on YOLOv5
for image in data/transformed_images/fsoco_sample/*; do
    python models/yolov5/infer_v5.py \
        --weights ./models/yolov5/yolov5_parameters.pt \
        --image "$image"
done
