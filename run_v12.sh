#!/bin/bash
set -e  # Exit immediately on error

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate metamorphic images
# for image in data/base_images/fsoco_sample/images/*; do
#     python models/metamorphic_test.py \
#         --rules imgGrayscale \
#         --base_image "$image" \
#         --output_path ./data/transformed_images/fsoco_sample
# done

# Run original images on yolov12
# for image in data/base_images/fsoco_sample/images/*; do
#     python models/yolov12/infer_v12.py \
#         --weights ./models/yolov12/yolov12_parameters.pt \
#         --image "$image"
# done

# Run transformed images on yolov12
for image in data/transformed_images/*.png; do
    python models/yolov12/infer_v12.py \
        --weights ./models/yolov12/yolov12_parameters.pt \
        --image "$image"
done
