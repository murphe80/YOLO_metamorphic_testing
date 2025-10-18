#!/bin/bash

python -m venv v5
source v5/bin/activate
pip install -r requirements.txt

for image in data/base_images/fsoco_sample/images/*.jpg; do
    python models/yolov5/metamorphic_test.py \
        --weights ./models/yolov5/yolov5_parameters.pt \
        --image "$image" \
        --rules imgBlur imgGrayscale
done