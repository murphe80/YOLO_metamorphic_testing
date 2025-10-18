#!/bin/bash

python -m venv v5
source v5/bin/activate
pip install -r requirements.txt

python ./models/yolov5/infer_v5.py --weights ./models/yolov5/yolov5_parameters.pt --image ./data/fsoco_sample/images/amz_00016.jpg --conf_thres 0.25 --iou_thres 0.45 --device cuda:0

python ./models/yolov5/infer_v5.py --weights ./models/yolov5/yolov5_parameters.pt --image ./data/transformed_images/img2_foggy.png --conf_thres 0.25 --iou_thres 0.45 --device cuda:0