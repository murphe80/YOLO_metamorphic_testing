#!/usr/bin/env python3
"""
Visualization utilities for YOLOvX detection results
"""

import argparse
import cv2
import numpy as np
import os
from typing import Any, Dict, List, Optional


CLASS_NAMES = [
    "Yellow Cone",
    "Orange Cone",
    "Big Orange Cone",
    "Blue Cone",
    "Unknown Cone",
]

COLOURS = {
    "Yellow Cone": (255, 255, 0),  # Yellow
    "Orange Cone": (255, 165, 0),  # Orange
    "Big Orange Cone": (255, 0, 0),  # Big Orange
    "Blue Cone": (0, 0, 255),  # Blue
    "Unknown Cone": (0, 255, 255),  # Unknown Cone
}


def read_detection_file(detections_path: str) -> List[Dict[str, Any]]:
    """Read detections saved as: class confidence x1 y1 x2 y2."""
    detections = []

    with open(detections_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(
                    f"{detections_path}:{line_number} expected 6 fields, got {len(parts)}"
                )

            cls_id = int(parts[0])
            confidence = float(parts[1])
            x1, y1, x2, y2 = map(float, parts[2:])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                "size": [x2 - x1, y2 - y1],
                "confidence": confidence,
                "class": cls_id,
            })

    return detections


def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]],
                    class_names: Optional[List[str]] = None,
                    conf_threshold: float = 0.0) -> np.ndarray:
    """
    Draw bounding boxes and labels on image

    Args:
        image: Input image as numpy array
        detections: List of detection dictionaries from extract_detections()
        class_names: List of class names
        conf_threshold: Minimum confidence threshold to display

    Returns:
        Image with drawn detections
    """

    class_names = class_names or CLASS_NAMES
    vis_image = image.copy()

    for detection in detections:
        if detection['confidence'] < conf_threshold:
            continue

        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        conf = detection['confidence']
        cls_id = detection['class']

        # Get class name and colour
        class_name = class_names[cls_id] if cls_id < len(
            class_names) else f'Class {cls_id}'
        colour = COLOURS.get(class_name, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), colour, 2)

        # Prepare label text
        label = f'{class_name}: {conf:.2f}'

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw background rectangle for text
        cv2.rectangle(vis_image,
                      (x1, y1 - text_height - baseline),
                      (x1 + text_width, y1),
                      colour, -1)

        # Draw text
        cv2.putText(vis_image, label,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return vis_image


def visualize_detections(image: np.ndarray, detections: List[Dict[str, Any]],
                         output_path: Optional[str] = None,
                         conf_threshold: float = 0.25,
                         show_plot: bool = False,
                         model: str = "v5") -> np.ndarray:
    """
    Visualize detections on an image

    Args:
        image: Input image as BGR numpy array
        detections: List of detection dictionaries
        output_path: Optional path to save visualization
        conf_threshold: Minimum confidence threshold
        show_plot: Whether to display the image in an OpenCV window
    """

    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    vis_image = draw_detections(
        image_rgb, detections, CLASS_NAMES, conf_threshold)

    if show_plot:
        cv2.imshow(f"{model} detections", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_image_bgr)
        print(f"Visualization saved to {output_path}")

    return vis_image


def print_detection_summary(detections: List[Dict[str, Any]],
                            class_names: Optional[List[str]] = None) -> None:
    """
    Print a summary of detections

    Args:
        detections: List of detection dictionaries
    """
    if not detections:
        print("No detections found.")
        return

    class_names = class_names or CLASS_NAMES
    print(f"\nDetection Summary ({len(detections)} objects):")
    print("-" * 50)

    for i, det in enumerate(detections):
        bbox = det['bbox']
        center = det['center']
        size = det['size']
        conf = det['confidence']
        cls_id = det['class']

        class_name = class_names[cls_id] if cls_id < len(
            class_names) else f'Class {cls_id}'

        print(f"Detection {i+1}:")
        print(f"  Class: {class_name} (ID: {cls_id})")
        print(f"  Confidence: {conf:.3f}")
        print(
            f"  Bounding Box: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
        print(f"  Center: ({center[0]:.1f}, {center[1]:.1f})")
        print(f"  Size: {size[0]:.1f} x {size[1]:.1f}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO detections')
    parser.add_argument('--image', required=True, type=str,
                        help='Path to input image')
    parser.add_argument('--detections', required=True, type=str,
                        help='Path to detection .txt file')
    parser.add_argument('--output', type=str,
                        help='Path to save visualization')
    parser.add_argument('--model', type=str, default='v5',
                        help='Model version label for display')
    parser.add_argument('--conf_threshold', type=float, default=0.0,
                        help='Confidence threshold for display')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display the plot')

    args = parser.parse_args()

    try:
        image = cv2.imread(args.image)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")

        detections = read_detection_file(args.detections)
        print_detection_summary(detections)
        visualize_detections(
            image=image,
            detections=detections,
            output_path=args.output,
            conf_threshold=args.conf_threshold,
            show_plot=not args.no_show,
            model=args.model,
        )

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
