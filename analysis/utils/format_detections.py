#Formats detection data for evaluation for compatibility with COCO API
# Detection format: confidence: float, class: int, bbox: [x1, y1, x2, y2]
# COCO format: bbox: [x, y, width, height], category_id: int, score: float

import json
import os
from pathlib import Path
from datetime import datetime

def format_detections(detection_file, output_file):
    """
    Format detection data for evaluation for compatibility with COCO API
    """