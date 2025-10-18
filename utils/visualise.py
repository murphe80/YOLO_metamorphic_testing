#!/usr/bin/env python3
"""
Visualization utilities for YOLOv5 detection results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import argparse
import os

class_names = [
    "Yellow Cone",
    "Orange Cone",
    "Big Orange Cone",
    "Blue Cone",
    "Unknown Cone",
]

# Define colours for different cone classes
colours = [
    (255, 255, 0),  # Yellow
    (255, 165, 0),  # Orange
    (255, 0, 0),  # Big Orange
    (0, 0, 255),  # Blue
    (0, 255, 255),  # Unknown Cone
]


def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]],
                    conf_threshold: float = 0.0) -> np.ndarray:
    """
    Draw bounding boxes and labels on image

    Args:
        image: Input image as numpy array
        detections: List of detection dictionaries from extract_detections()
        conf_threshold: Minimum confidence threshold to display

    Returns:
        Image with drawn detections
    """

    # Create a copy of the image to draw on
    vis_image = image.copy()

    for detection in detections:
        # Filter by confidence threshold
        if detection['confidence'] < conf_threshold:
            continue

        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        conf = detection['confidence']
        cls_id = detection['class']

        # Get class name and colour
        class_name = class_names[cls_id] if cls_id < len(
            class_names) else f'Class {cls_id}'
        colour = colours[cls_id]

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


def visualize_detections(image_path: str, detections: List[Dict[str, Any]],
                         output_path: Optional[str] = None,
                         conf_threshold: float = 0.25,
                         show_plot: bool = True) -> None:
    """
    Visualize detections on an image

    Args:
        image_path: Path to input image
        detections: List of detection dictionaries
        output_path: Optional path to save visualization
        conf_threshold: Minimum confidence threshold
        show_plot: Whether to display the plot
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw detections
    vis_image = draw_detections(
        image_rgb, detections, conf_threshold)

    # Display or save
    if show_plot:
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.title(f'YOLOv5 Detections ({len(detections)} objects found)')
        plt.tight_layout()
        plt.show()

    if output_path:
        # Convert back to BGR for saving
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_image_bgr)
        print(f"Visualization saved to {output_path}")


def print_detection_summary(detections: List[Dict[str, Any]]) -> None:
    """
    Print a summary of detections

    Args:
        detections: List of detection dictionaries
    """
    if not detections:
        print("No detections found.")
        return

    print(f"\nDetection Summary ({len(detections)} objects):")
    print("-" * 50)

    for i, det in enumerate(detections):
        bbox = det['bbox']
        center = det['center']
        size = det['size']
        conf = det['confidence']
        cls_id = det['class']

        class_name = class_names[cls_id] if class_names and cls_id < len(
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
    """Example usage of visualization functions"""
    parser = argparse.ArgumentParser(description='Visualize YOLO detections')
    parser.add_argument('--image', required=True, type=str,
                        help='Path to input image')
    parser.add_argument('--detections', type=str,
                        help='List of detections')
    parser.add_argument('--output', type=str,
                        help='Path to save visualization')
    parser.add_argument('--conf_threshold', type=float, default=0.0,
                        help='Confidence threshold for display')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display the plot')

    args = parser.parse_args()

    try:
        print_detection_summary(args.detections)
        # Visualize detections
        visualize_detections(
            image_path=args.image,
            detections=args.detections,
            output_path=args.output,
            conf_threshold=args.conf_threshold,
            show_plot=not args.no_show
        )

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
