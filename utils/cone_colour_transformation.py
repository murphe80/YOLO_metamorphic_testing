# apply colour transformation (color channel rule) to segmented cone images
# combine result with original image 
import os
import cv2 as cv
import numpy as np
from metamorphic_rules.instantiate_rules import instantiate_rule

# Colour multipliers
RED_MULTIPLIER = 0.1
GREEN_MULTIPLIER = 1
BLUE_MULTIPLIER = 1

segmented_path = "/home/murphe80/metamorphic_testing/YOLO_metamorphic_testing/data/filtered_segmented"
output_path = "/home/murphe80/metamorphic_testing/YOLO_metamorphic_testing/data/transformed_images/fsoco_test_segmented/decrease_red"

os.makedirs(output_path, exist_ok=True)

def apply_color_channel_rule(image):
    original_img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    rule_kwargs = {'color_channels_out': 'RGB', 'channel_multipliers': (RED_MULTIPLIER, GREEN_MULTIPLIER, BLUE_MULTIPLIER)}
    
    rule = instantiate_rule('imgColorChannels', {}, **rule_kwargs)
    transformed_img = rule.apply(original_img_rgb, {})
    res = cv.cvtColor(transformed_img, cv.COLOR_RGB2BGR)
    return res

image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(segmented_path) 
               if f.lower().endswith(image_extensions)]

print(f"Found {len(image_files)} images to transform")

for idx, image_name in enumerate(image_files):
    
    # Read the stacked image
    image_path = os.path.join(segmented_path, image_name)
    stacked_image = cv.imread(image_path)
    
    if stacked_image is None:
        print(f"Warning: Could not read {image_name}")
        continue
 
    height = stacked_image.shape[0]
    single_height = height // 2
    
    original = stacked_image[0:single_height, :]
    filtered_cones = stacked_image[single_height:, :]
    
    # apply rule
    transformed = apply_color_channel_rule(filtered_cones)
    
    # merge original and transformed cones 
    gray = cv.cvtColor(filtered_cones, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    mask_3ch = cv.merge((mask, mask, mask))
    
    # Apply color channel rule to transform cones
    transformed_cones = apply_color_channel_rule(filtered_cones)
    
    # Create inverse mask (everything except cones)
    inverse_mask = cv.bitwise_not(mask_3ch)
    
    # Extract background from original image
    background = cv.bitwise_and(original, inverse_mask)
    
    # Extract only the transformed cone pixels
    transformed_cones_only = cv.bitwise_and(transformed_cones, mask_3ch)
    
    # Combine transformed cones with original background
    result = cv.add(background, transformed_cones_only)
    
    
    # write to file 
    output_filename = os.path.join(output_path, image_name)
    cv.imwrite(output_filename, result)