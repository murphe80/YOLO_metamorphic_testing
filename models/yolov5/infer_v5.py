#!/usr/bin/env python3
"""
YOLOv5 object detection script 
"""

import random
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from datetime import datetime
import cv2
from yolov5.utils.general import check_img_size, non_max_suppression
from yolov5.models.experimental import attempt_load
import torch.backends.cudnn as cudnn
import torch
import argparse
import numpy as np
from metamorphic_test import gen_metamorphic


try:
    from yolov5.utils.augmentations import letterbox
except:
    print("Could not import letterbox from yolov5.utils.augmentations. Trying to import from  yolov5.utils.datasets…")
    from yolov5.utils.datasets import letterbox


def load_model(weights, img_size=640, device='cuda:0'):
    """Load YOLOv5 model and return model, device, and settings"""
    device = torch.device(device)
    half = device.type != 'cpu'

    print(f"Loading model from {weights}...")
    model = attempt_load(weights, device=device)
    stride = 32
    img_size = check_img_size(img_size, s=stride)

    if half:
        model.half()

    cudnn.benchmark = True

    # Warm up
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size)
              .to(device).type_as(next(model.parameters())))

    print("Model loaded successfully!")
    return model, device, half, img_size


def img_preprocess(img, device, half, net_size):
    """Preprocess image for model input"""

    net_image, ratio, pad = letterbox(img[:, :, :3], net_size, auto=False)
    net_image = net_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    net_image = np.ascontiguousarray(net_image)

    img = torch.from_numpy(net_image).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, ratio, pad


def inference(model, device, half, img_size, img, conf_thres, iou_thres):
    """Run inference on image"""
    with torch.no_grad():
        img_tensor, ratio, pad = img_preprocess(img, device, half, img_size)
        pred = model(img_tensor)[0]
        det = non_max_suppression(pred, conf_thres, iou_thres)
        return det, ratio, pad


def extract_detections(det, img_shape, ratio, pad):
    """Extract detection information into list of dicts, scaled to original image"""
    detections = []

    if det[0] is not None and len(det[0]):
        det_output = det[0].cpu().numpy()

        for *xyxy, conf, cls in det_output:
            # Scale coordinates back to original image
            x1 = (xyxy[0] - pad[0]) / ratio[0]
            y1 = (xyxy[1] - pad[1]) / ratio[1]
            x2 = (xyxy[2] - pad[0]) / ratio[0]
            y2 = (xyxy[3] - pad[1]) / ratio[1]

            # Clip to image boundaries
            x1 = max(0, min(x1, img_shape[1]))
            y1 = max(0, min(y1, img_shape[0]))
            x2 = max(0, min(x2, img_shape[1]))
            y2 = max(0, min(y2, img_shape[0]))

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'center': [cx, cy],
                'size': [w, h],
                'confidence': float(conf),
                'class': int(cls)
            })

    return detections


def main():
    parser = argparse.ArgumentParser(description='YOLOv5 Metamorphic Testing')
    parser.add_argument('--weights', nargs='+', type=str,
                        default=None, help='model.pt path(s)')
    parser.add_argument('--img_size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float,
                        default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--device', type=str,
                        default="cuda:0", help='pytorch device to use')
    parser.add_argument('--image', required=True, type=str, help='image path')
    parser.add_argument('--rules', nargs='+', default=['imgBlur'],
                        help='List of metamorphic rules to test')

    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    # Get transformations 
    transformed_images = gen_metamorphic(img, args.rules)

    # Load model
    model, device, half, img_size = load_model(
        args.weights, args.img_size, args.device)

    # Run inference on each transformation
    for i in range(len(transformed_images)):
        img = transformed_images[i]
        det, ratio, pad = inference(
            model, device, half, img_size, img, args.conf_thres, args.iou_thres)

        # Process detections
        detections = extract_detections(det, img.shape, ratio, pad)

        # Print detections
        print(detections)

        # Visualise detections
        from utils.visualise import visualize_detections as visualise
        visualise(img, detections, show_plot=True,
                output_path=f"./results/matias'/yolov5/YOLOv5_{os.path.basename(args.image)}_{i}_{datetime.now().strftime('%Y%m%d')}.jpg")


if __name__ == '__main__':
    main()
