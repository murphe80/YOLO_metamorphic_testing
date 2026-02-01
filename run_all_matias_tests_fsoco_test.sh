#!/bin/bash

# Number of images per (model, transform) to save visualizations for validation
NUM_VIZ=3

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# **Assumes all metamorphic test images have been generated**

# Run original images on v5
img_count=0
for image in data/base_images/test/images/*; do
    [ ! -f "$image" ] && continue
    save_viz=""
    [ $img_count -lt $NUM_VIZ ] && save_viz="--save-viz"
    python models/yolov5/infer_v5_kfold.py \
        --weights ./models/yolov5/yolov5_param_kfold.pt \
        --image "$image" \
        --output-subdir original \
        $save_viz
    ((img_count++)) || true
done

# Run transformed images on v5
for dir in data/transformed_images/fsoco_test/*; do
    [ ! -d "$dir" ] && continue
    subdir=$(basename "$dir")
    img_count=0
    for image in "$dir"/*; do
        [ ! -f "$image" ] && continue
        save_viz=""
        [ $img_count -lt $NUM_VIZ ] && save_viz="--save-viz"
        python models/yolov5/infer_v5_kfold.py \
            --weights ./models/yolov5/yolov5_param_kfold.pt \
            --image "$image" \
            --output-subdir "$subdir" \
            $save_viz
        ((img_count++)) || true
    done
done

# Run original images on v8
img_count=0
for image in data/base_images/test/images/*; do
    [ ! -f "$image" ] && continue
    save_viz=""
    [ $img_count -lt $NUM_VIZ ] && save_viz="--save-viz"
    python models/yolov8/infer_v8.py \
        --weights ./models/yolov8/yolov8_param_kfold.pt \
        --image "$image" \
        --output-subdir original \
        $save_viz
    ((img_count++)) || true
done

# Run transformed images on v8
for dir in data/transformed_images/fsoco_test/*; do
    [ ! -d "$dir" ] && continue
    subdir=$(basename "$dir")
    img_count=0
    for image in "$dir"/*; do
        [ ! -f "$image" ] && continue
        save_viz=""
        [ $img_count -lt $NUM_VIZ ] && save_viz="--save-viz"
        python models/yolov8/infer_v8.py \
            --weights ./models/yolov8/yolov8_param_kfold.pt \
            --image "$image" \
            --output-subdir "$subdir" \
            $save_viz
        ((img_count++)) || true
    done
done

# Run original images on v12
img_count=0
for image in data/base_images/test/images/*; do
    [ ! -f "$image" ] && continue
    save_viz=""
    [ $img_count -lt $NUM_VIZ ] && save_viz="--save-viz"
    python models/yolov12/infer_v12.py \
        --weights ./models/yolov12/yolov12_param_kfold.pt \
        --image "$image" \
        --output-subdir original \
        $save_viz
    ((img_count++)) || true
done

# Run transformed images on v12
for dir in data/transformed_images/fsoco_test/*; do
    [ ! -d "$dir" ] && continue
    subdir=$(basename "$dir")
    img_count=0
    for image in "$dir"/*; do
        [ ! -f "$image" ] && continue
        save_viz=""
        [ $img_count -lt $NUM_VIZ ] && save_viz="--save-viz"
        python models/yolov12/infer_v12.py \
            --weights ./models/yolov12/yolov12_param_kfold.pt \
            --image "$image" \
            --output-subdir "$subdir" \
            $save_viz
        ((img_count++)) || true
    done
done
