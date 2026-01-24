#!/usr/bin/env python3
"""
Generates transformed images to be used for metamorphic testing 
"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metamorphic_rules.instantiate_rules import instantiate_rule
import cv2

rule_configs = {
    'imgDarkBright': {'brightness_constant': -20, 'brightness_multiplier': 0.2, 'gamma': 1, 'method': 'gamma'},
    'imgResolution': {'size_ratio': 0.6},
    'imgMotionBlur': {'kernel_size': 12, 'orientation': 45},
    'imgStretch': {'height_ratio': 1.5, 'width_ratio': 1.5},
    'imgGrayscale': {},
    'imgColorWheel': {'hue_rotation': 15},
}


def gen_metamorphic(original_img, rules, img_path, output_path):
    """Generates transformed images
        First element is the original image 
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Convert BGR (from cv2.imread) to RGB for rule processing
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    selected_rules = {k: v for k, v in rule_configs.items() if k in rules}
    print(selected_rules.items())
    
    saved_count = 0
    for rule_name, rule_kwargs in selected_rules.items():
        print(f"\n=== Transforming using {rule_name} ===")

        try:
            # Create and apply transformation
            extra_data = {}
            rule = instantiate_rule(rule_name, extra_data, **rule_kwargs)
            transformed_img = rule.apply(original_img_rgb, {})
            
            # Convert RGB back to BGR for saving with cv2.imwrite
            transformed_img_bgr = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
            
            # Construct output path using os.path.join
            output_filename = f'{img_path[0]}_{rule_name}{img_path[1]}'
            os.makedirs(output_path, exist_ok=True)
            output_filepath = os.path.join(output_path, output_filename)
            
            # Save image and check for success
            
            success = cv2.imwrite(output_filepath, transformed_img_bgr)
            if success:
                print(f"Saved: {output_filepath}")
                saved_count += 1
            else:
                print(f"Failed to save: {output_filepath}")
        except Exception as e:
            print(f"Error applying {rule_name}: {e}")
    
    return saved_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metamorphic Testing Image Generation')
    parser.add_argument('--rules', nargs='+', default=['imgBlur'],
                        help='List of metamorphic rules to test')
    parser.add_argument('--base_image', help='folder containing original images')
    parser.add_argument('--output_path', default='./data/transformed_images')
    
    args = parser.parse_args()
    rules = args.rules
    img = cv2.imread(args.base_image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.base_image}")
    
    img_path = os.path.splitext(os.path.basename(args.base_image))
    saved_count = gen_metamorphic(img, rules, img_path, args.output_path)
    
    print(f"\nSaved {saved_count} transformed images to {args.output_path}")
    