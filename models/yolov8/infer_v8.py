#!/usr/bin/env python3
"""
yolov8 object detection script
Compatible with Ultralytics yolov8 (.pt) models.
"""

import os
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import torch
from ultralytics import YOLO
from utils.visualise import visualize_detections as visualise


def load_model(weights, device="cuda:0"):
    """Load yolov8 model."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Loading yolov8 model from {weights} on {device}...")
    model = YOLO(weights)
    print("Model loaded successfully.")
    return model, device


def run_inference(model, image_path, conf_thres=0.25, iou_thres=0.45, device="cuda:0"):
    """Perform inference on a single image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    results = model.predict(
        source=img,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        verbose=False
    )

    return results[0]


def extract_detections(result):
    """Convert yolov8 result to structured data."""
    detections = []
    boxes = result.boxes

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf.cpu().numpy())
        cls = int(box.cls.cpu().numpy())

        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "center": [cx, cy],
            "size": [w, h],
            "confidence": conf,
            "class": cls
        })

    return detections


def main():
    parser = argparse.ArgumentParser(description="yolov8 Object Detection")
    parser.add_argument("--weights", type=str, required=True, help="Path to yolov8 .pt weights file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")

    args = parser.parse_args()
    
    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image from {args.image}")
        return

    # Load model
    model, device = load_model(args.weights, args.device)

    # Run inference
    result = run_inference(model, args.image, args.conf_thres, args.iou_thres, device)

    # Extract detection data
    detections = extract_detections(result)
    print("Detections:")
    for det in detections:
        print(det)

    # Save visualization
    img_path = os.path.splitext(os.path.basename(args.image))
    visualise(img, detections, show_plot=True,
            output_path=f"./results/yolov8/{img_path[0]}{img_path[1]}", model="v8")


if __name__ == "__main__":
    main()
