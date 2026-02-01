# run analysis for yolov5
# compare original vs ground truth, original vs transformed, transformed vs ground truth

# format yolov5 detection data for coco evaluation
python analysis/utils/format_detections.py --detection_file results/yolov5/ --output_file results/yolov5/fsoco_test/detections_original_coco.json
python analysis/utils/format_detections.py --detection_file results/yolov5/ --output_file results/yolov5/fsoco_test/detections_transformed_coco.json

# evaluate performance (original vs gt)
python analysis/coco_eval.py --gt data/base_images/fsoco_sample/annotations_coco.json --dt results/yolov5/fsoco_test/detections_origianl_coco.json

# evaluate performance (original vs transformed)
python analysis/coco_eval.py --gt results/yolov5/fsoco_test/detections_original_coco.json --dt results/yolov5/fsoco_test/detections_transformed_coco.json

# evaluate performance (transformed vs gt)
python analysis/coco_eval.py --gt data/base_images/fsoco_sample/annotations_coco.json --dt results/yolov5/fsoco_test/detections_transformed_coco.json