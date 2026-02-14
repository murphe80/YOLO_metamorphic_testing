#!/usr/bin/env python3
"""
Generates transformed images to be used for metamorphic testing 
"""
import argparse
import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metamorphic_rules.instantiate_rules import instantiate_rule
import cv2


def gen_metamorphic(original_img, rule_name, rule_config, img_path, output_path):
    """Generates transformed image for a single rule
    
    Args:
        original_img: The original image (BGR format from cv2.imread)
        rule_name: Name of the transformation rule
        rule_config: Dictionary with rule parameters
        img_path: Tuple of (filename_without_ext, extension)
        output_path: Directory to save transformed images
    
    Returns:
        Tuple of (success: bool, output_filepath: str or None)
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Convert BGR (from cv2.imread) to RGB for rule processing
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    print(f"\n=== Transforming using {rule_name} ===")
    print(f"Parameters: {rule_config}")

    try:
        # Create and apply transformation
        extra_data = {}
        rule = instantiate_rule(rule_name, extra_data, **rule_config)
        transformed_img = rule.apply(original_img_rgb, {})
        
        # Convert RGB back to BGR for saving with cv2.imwrite
        transformed_img_bgr = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
        
        # Construct output path using os.path.join
        output_filename = f'{img_path[0]}_{rule_name}{img_path[1]}'
        output_filepath = os.path.join(output_path, output_filename)
        
        # Save image and check for success
        success = cv2.imwrite(output_filepath, transformed_img_bgr)
        if success:
            print(f"Saved: {output_filepath}")
            return True, output_filepath
        else:
            print(f"Failed to save: {output_filepath}")
            return False, None
    except Exception as e:
        print(f"Error applying {rule_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metamorphic Testing Image Generation')
    parser.add_argument('--rule', required=True,
                        help='Metamorphic rule to apply')
    parser.add_argument('--config', type=str, default='{}',
                        help='JSON string with rule configuration parameters')
    parser.add_argument('--base_image', required=True,
                        help='Path to original image')
    parser.add_argument('--output_path', required=True,
                        help='Directory to save transformed images')
    parser.add_argument('--save-config', action='store_true',
                        help='Save configuration to configs.json in output directory')
    
    args = parser.parse_args()
    
    # Read the image
    img = cv2.imread(args.base_image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.base_image}")
    
    # Parse configuration
    try:
        rule_config = json.loads(args.config)
    except json.JSONDecodeError as e:
        print(f"Error parsing config JSON: {e}")
        print(f"Config string: {args.config}")
        sys.exit(1)
    
    # Get image name components
    img_path = os.path.splitext(os.path.basename(args.base_image))
    
    # Apply transformation
    success, output_filepath = gen_metamorphic(img, args.rule, rule_config, img_path, args.output_path)
    
    if success:
        print(f"\n✓ Successfully saved transformed image")
        
        # Save configuration file if requested
        if args.save_config:
            config_file = os.path.join(args.output_path, 'configs.json')
            
            # Load existing configs if file exists
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    all_configs = json.load(f)
            else:
                all_configs = {}
            
            # Update with current rule config
            all_configs[args.rule] = rule_config
            
            # Save updated configs
            with open(config_file, 'w') as f:
                json.dump(all_configs, f, indent=2)
            
            print(f"✓ Configuration saved to {config_file}")
    else:
        print(f"\n✗ Failed to save transformed image")
        sys.exit(1)
    