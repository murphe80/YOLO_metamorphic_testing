#!/usr/bin/env python3
"""
Generates transformed images to be used on yolov5 
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
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


def gen_metamorphic(original_img, rules):
    """Generates transformed images
        First element is the original image 
    """

    results = []
    results.append(original_img)
    
    selected_rules = {k: v for k, v in rule_configs.items() if k in rules}
    print(selected_rules.items())
    for rule_name, rule_kwargs in selected_rules.items():
        print(f"\n=== Transforming using {rule_name} ===")

        # Create and apply transformation
        extra_data = {}
        rule = instantiate_rule(rule_name, extra_data, **rule_kwargs)
        transformed_img = rule.apply(original_img, {})
        results.append(transformed_img)
        
    return results
