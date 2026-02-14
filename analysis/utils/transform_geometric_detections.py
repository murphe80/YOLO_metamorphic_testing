"""
Transform detection coordinates to account for geometric image transformations.

This script takes detections from geometrically transformed images (imgStretch, imgResolution)
and transforms the bounding box coordinates back to the original image space for proper comparison.
"""

import json
import argparse
from pathlib import Path
import re


def get_image_dimensions(image_path):
    """
    Get image dimensions from an image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple (width, height) or None if image not found
    """
    try:
        # Try PIL/Pillow first
        from PIL import Image
        img = Image.open(image_path)
        return img.size  # (width, height)
    except ImportError:
        # Fallback to OpenCV if available
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is not None:
                height, width = img.shape[:2]
                return (width, height)
        except:
            pass
    except:
        pass
    
    # Last resort: try to parse PNG/JPEG headers manually
    try:
        import struct
        with open(image_path, 'rb') as f:
            data = f.read(24)
            if data[:8] == b'\x89PNG\r\n\x1a\n':  # PNG
                w, h = struct.unpack('>LL', data[16:24])
                return (w, h)
            elif data[:2] == b'\xff\xd8':  # JPEG
                f.seek(0)
                # Basic JPEG dimension parsing
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    f.seek(size, 1)
                    byte = f.read(1)
                    while ord(byte) == 0xff:
                        byte = f.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', f.read(2))[0] - 2
                f.seek(1, 1)
                h, w = struct.unpack('>HH', f.read(4))
                return (w, h)
    except:
        pass
    
    return None


def parse_transformation_params(transformation_name, original_img_path, transformed_img_path):
    """
    Parse transformation parameters by comparing original and transformed images.
    
    Args:
        transformation_name: Name of transformation (imgStretch, imgResolution, etc.)
        original_img_path: Path to original image
        transformed_img_path: Path to transformed image
        
    Returns:
        Dictionary with transformation parameters
    """
    # Get dimensions
    orig_dims = get_image_dimensions(original_img_path)
    trans_dims = get_image_dimensions(transformed_img_path)
    
    if orig_dims is None or trans_dims is None:
        # Fall back to common transformation parameters
        print(f"Warning: Could not read image dimensions, using default parameters for {transformation_name}")
        if transformation_name == 'imgStretch':
            return {'type': 'stretch', 'scale_x': 1.5, 'scale_y': 1.0}
        elif transformation_name == 'imgResolution':
            return {'type': 'resize', 'scale_x': 0.5, 'scale_y': 0.5}
        else:
            return {'type': 'identity', 'scale_x': 1.0, 'scale_y': 1.0}
    
    orig_w, orig_h = orig_dims
    trans_w, trans_h = trans_dims
    
    # Calculate scaling factors
    scale_x = orig_w / trans_w if trans_w > 0 else 1.0
    scale_y = orig_h / trans_h if trans_h > 0 else 1.0
    
    params = {
        'type': transformation_name,
        'orig_width': orig_w,
        'orig_height': orig_h,
        'trans_width': trans_w,
        'trans_height': trans_h,
        'scale_x': scale_x,
        'scale_y': scale_y
    }
    
    return params


def transform_bbox(bbox, transform_params):
    """
    Transform bounding box coordinates back to original image space.
    
    Args:
        bbox: Bounding box in [x, y, width, height] format
        transform_params: Dictionary with transformation parameters
        
    Returns:
        Transformed bbox in [x, y, width, height] format
    """
    x, y, w, h = bbox
    scale_x = transform_params.get('scale_x', 1.0)
    scale_y = transform_params.get('scale_y', 1.0)
    
    # Scale coordinates back to original image space
    x_orig = x * scale_x
    y_orig = y * scale_y
    w_orig = w * scale_x
    h_orig = h * scale_y
    
    return [x_orig, y_orig, w_orig, h_orig]


def transform_detections_file(detection_file, original_image_dir, transformed_image_dir, 
                              transformation_name, output_file=None):
    """
    Transform detections from a single file.
    
    Args:
        detection_file: Path to detection COCO JSON file
        original_image_dir: Directory with original images
        transformed_image_dir: Directory with transformed images
        transformation_name: Name of transformation (e.g., 'imgStretch')
        output_file: Optional output file path
        
    Returns:
        Transformed detections as list
    """
    # Load detections
    with open(detection_file, 'r') as f:
        detections = json.load(f)
    
    # If it's a full COCO format, extract annotations
    if isinstance(detections, dict):
        if 'annotations' in detections:
            annotations = detections['annotations']
            images = detections.get('images', [])
        else:
            print("Warning: Unexpected COCO format")
            return detections
    else:
        annotations = detections
        images = []
    
    # Build image ID to actual filename mapping by checking txt files
    # in the transformed directory
    txt_files = list(Path(transformed_image_dir).glob("*.txt"))
    image_id_to_basename = {}
    
    for txt_file in txt_files:
        # e.g., "amz_00811_imgStretch.txt" -> "amz_00811_imgStretch"
        full_name = txt_file.stem
        # Strip transformation suffix -> "amz_00811"
        if f'_{transformation_name}' in full_name:
            base_name = full_name.replace(f'_{transformation_name}', '')
        else:
            base_name = full_name
        
        # Find which image_id this corresponds to
        for img in images:
            img_name = Path(img.get('file_name', '')).stem
            if img_name == f"image_{img['id']}":
                # This is a placeholder - try to match by order or other means
                pass
    
    # Group annotations by image_id to process by image
    from collections import defaultdict
    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann['image_id']].append(ann)
    
    # Get default transformation parameters based on transformation type
    # These are scale factors to convert FROM transformed TO original coordinates
    # For imgStretch: images are stretched by 1.5x, so scale back by 1/1.5 = 0.667
    # For imgResolution: downscaled to 0.6x, so scale back by 1/0.6 = 1.667
    if transformation_name == 'imgStretch':
        default_params = {'scale_x': 0.667, 'scale_y': 0.667}  # 1/1.5 to reverse 1.5x stretch
    elif transformation_name == 'imgResolution':
        default_params = {'scale_x': 1.667, 'scale_y': 1.667}  # 1/0.6 to reverse 0.6x downscale
    else:
        default_params = {'scale_x': 1.0, 'scale_y': 1.0}
    
    # Find actual transformation parameters from real images
    transform_params_cache = {}
    
    # Sample a few images to determine actual transformation
    sample_count = 0
    for txt_file in txt_files[:5]:  # Sample first 5 images
        full_name = txt_file.stem
        if f'_{transformation_name}' in full_name:
            base_name = full_name.replace(f'_{transformation_name}', '')
        else:
            base_name = full_name
        
        # Find images
        orig_img = None
        trans_img = None
        for ext in ['.jpg', '.png', '.jpeg']:
            orig_path = Path(original_image_dir) / f"{base_name}{ext}"
            if orig_path.exists():
                orig_img = orig_path
                break
        
        for ext in ['.jpg', '.png', '.jpeg']:
            trans_path = Path(transformed_image_dir) / f"{full_name}{ext}"
            if trans_path.exists():
                trans_img = trans_path
                break
        
        if orig_img and trans_img:
            params = parse_transformation_params(transformation_name, orig_img, trans_img)
            transform_params_cache[base_name] = params
            sample_count += 1
            if sample_count >= 3:  # Get average from 3 samples
                break
    
    # Calculate average transformation parameters
    if transform_params_cache:
        avg_scale_x = sum(p['scale_x'] for p in transform_params_cache.values()) / len(transform_params_cache)
        avg_scale_y = sum(p['scale_y'] for p in transform_params_cache.values()) / len(transform_params_cache)
        default_params = {'scale_x': avg_scale_x, 'scale_y': avg_scale_y}
        print(f"Detected transformation parameters: scale_x={avg_scale_x:.3f}, scale_y={avg_scale_y:.3f}")
    else:
        print(f"Using default transformation parameters: scale_x={default_params['scale_x']}, scale_y={default_params['scale_y']}")
    
    # Transform all annotations using the determined parameters
    transformed_annotations = []
    for ann in annotations:
        # Transform the bbox
        transformed_bbox = transform_bbox(ann['bbox'], default_params)
        
        # Create transformed annotation
        trans_ann = ann.copy()
        trans_ann['bbox'] = transformed_bbox
        trans_ann['area'] = transformed_bbox[2] * transformed_bbox[3]
        
        transformed_annotations.append(trans_ann)
    
    # Save to output file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(detections, dict):
            # Save in full COCO format
            output_data = detections.copy()
            output_data['annotations'] = transformed_annotations
        else:
            # Save as array
            output_data = transformed_annotations
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Transformed {len(transformed_annotations)} detections")
        print(f"Saved to: {output_file}")
    
    return transformed_annotations if isinstance(detections, list) else {**detections, 'annotations': transformed_annotations}


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Transform detection coordinates to account for geometric transformations"
    )
    parser.add_argument(
        '--detection_file',
        type=str,
        required=True,
        help='Path to detection COCO JSON file from transformed images'
    )
    parser.add_argument(
        '--transformation',
        type=str,
        required=True,
        choices=['imgStretch', 'imgResolution'],
        help='Type of geometric transformation'
    )
    parser.add_argument(
        '--original_images',
        type=str,
        required=True,
        help='Directory containing original images'
    )
    parser.add_argument(
        '--transformed_images',
        type=str,
        required=True,
        help='Directory containing transformed images'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output path for transformed detections'
    )
    
    args = parser.parse_args()
    
    transform_detections_file(
        args.detection_file,
        args.original_images,
        args.transformed_images,
        args.transformation,
        args.output_file
    )


if __name__ == "__main__":
    main()

