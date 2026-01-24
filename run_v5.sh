#!/bin/bash
set -e  # Exit immediately on error

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


# Run original images on YOLOv5
for image in data/base_images/fsoco_sample/images/*; do
    python models/yolov5/infer_v5.py \
        --weights ./models/yolov5/yolov5_parameters_Cora.pt \
        --image "$image"
done

# Run transformed images on YOLOv5
for image in data/transformed_images/fsoco_test/*; do
    python models/yolov5/infer_v5.py \
        --weights ./models/yolov5/yolov5_parameters_Cora.pt \
        --image "$image"
done
