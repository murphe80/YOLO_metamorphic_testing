#!/usr/bin/env python3
"""
Generates transformed images to be used on yolov5 
"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metamorphic_rules.instantiate_rules import instantiate_rule
import cv2

rule_configs = {
    'imgBlur': {'blur_strength': 140, 'filter_size': 10},
    'imgRotation': {'rotation_angle': 15, 'center':(0.5,0.5)},
    'imgDarkBright': {'brightness_constant': -20, 'brightness_multiplier': 0.2, 'gamma':1, 'method': 'gamma'},
    'imgMirror': {'flip_code': 1},
    'imgResolution': {'size_ratio': 0.6},
    'imgGrayscale': {},
    'imgEdges': {'aperture_size': 3, 'sigma': 0.33}
}


def gen_metamorphic(original_img, rules, img_path):
    """Generates transformed images
        First element is the original image 
    """

    
    selected_rules = {k: v for k, v in rule_configs.items() if k in rules}
    print(selected_rules.items())
    for rule_name, rule_kwargs in selected_rules.items():
        print(f"\n=== Transforming using {rule_name} ===")

        # Create and apply transformation
        extra_data = {}
        rule = instantiate_rule(rule_name, extra_data, **rule_kwargs)
        transformed_img = rule.apply(original_img, {})
        cv2.imwrite(f'{args.output_path}/{img_path[0]}_{rule_name}{img_path[1]}', transformed_img)


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
    gen_metamorphic(img, rules, img_path)
    
    print(f"Transformed images saved")
    