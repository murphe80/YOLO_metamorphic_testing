#format fsoco data 
python analysis/utils/format_gt.py --input_dir data/base_images/test/labels --output_file data/base_images/test/labels/annotations_coco.json

#format each rule directory in /results/{image_rule}/ to coco format
# format base detections
python analysis/utils/format_detections.py --detection_file results/yolov5/base_images/ --output_file results/yolov5/base_images/detections_original_coco.json
for rule in results/yolov5/*; do
    python analysis/utils/format_detections.py --detection_file results/{rule}/ --output_file results/{rule}/detections_coco.json
    # evaluate performance (original vs gt)
    python analysis/coco_eval.py --gt results/yolov5/base_images/detections_original_coco.json --dt results/{rule}/detections_coco.json
    # evaluate performance (original vs transformed)
    python analysis/coco_eval.py --gt results/{rule}/detections_coco.json --dt results/{rule}/detections_transformed_coco.json
    # evaluate performance (transformed vs gt)
    python analysis/coco_eval.py --gt data/base_images/test/labels/annotations_coco.json --dt results/{rule}/detections_transformed_coco.json
done

#format base
python analysis/utils/format_detections.py --detection_file results/yolov8/base_images/ --output_file results/yolov8/detections_original_coco.json
for rule in results/yolov8/*; do
    python analysis/utils/format_detections.py --detection_file results/{rule}/ --output_file results/{rule}/detections_coco.json
    # evaluate performance (original vs gt)
    python analysis/coco_eval.py --gt results/yolov8/base_images/detections_original_coco.json --dt results/{rule}/detections_coco.json
    # evaluate performance (original vs transformed)
    python analysis/coco_eval.py --gt results/{rule}/detections_coco.json --dt results/{rule}/detections_transformed_coco.json
    # evaluate performance (transformed vs gt)
    python analysis/coco_eval.py --gt data/base_images/test/labels/annotations_coco.json --dt results/{rule}/detections_transformed_coco.json
done

#format base
python analysis/utils/format_detections.py --detection_file results/yolov12/base_images/ --output_file results/yolov12/detections_original_coco.json
for rule in results/yolov12/*; do
    python analysis/utils/format_detections.py --detection_file results/{rule}/ --output_file results/{rule}/detections_coco.json
    # evaluate performance (original vs gt)
    python analysis/coco_eval.py --gt results/yolov12/base_images/detections_original_coco.json --dt results/{rule}/detections_coco.json
    # evaluate performance (original vs transformed)
    python analysis/coco_eval.py --gt results/{rule}/detections_coco.json --dt results/{rule}/detections_transformed_coco.json
    # evaluate performance (transformed vs gt)
    python analysis/coco_eval.py --gt data/base_images/test/labels/annotations_coco.json --dt results/{rule}/detections_transformed_coco.json
done