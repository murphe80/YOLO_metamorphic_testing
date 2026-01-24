#!/bin/bash

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# **Assumes all metamorphic test images have been generated** 

# Run original images on v5 
for image in data/base_images/test/images/*; do
    python models/yolov5/infer_v5_kfold.py \
        --weights ./models/yolov5/yolov5_param_kfold.pt \
        --image "$image"
done

# Run transformed images on v5 
for dir in data/transformed_images/fsoco_test/*; do
    for image in "$dir"/*; do
        python models/yolov5/infer_v5_kfold.py \
            --weights ./models/yolov5/yolov5_param_kfold.pt \
            --image "$image"
    done
done

# Run original images on v8 
for image in data/base_images/test/images/*; do
    python models/yolov8/infer_v8.py \
        --weights ./models/yolov8/yolov8_param_kfold.pt \
        --image "$image"
done

# Run transformed images on v8 
for dir in data/transformed_images/fsoco_test/*; do
    for image in "$dir"/*; do
        python models/yolov8/infer_v8.py \
            --weights ./models/yolov8/yolov8_param_kfold.pt \
            --image "$image"
    done
done

# Run original images on v12
for image in data/base_images/test/images/*; do
    python models/yolov12/infer_v12.py \
        --weights ./models/yolov12/yolov12_param_kfold.pt \
        --image "$image"
done

# Run transformed images on v12
for dir in data/transformed_images/fsoco_test/*; do
    for image in "$dir"/*; do
        python models/yolov12/infer_v12.py \
            --weights ./models/yolov12/yolov12_param_kfold.pt \
            --image "$image"
    done
done