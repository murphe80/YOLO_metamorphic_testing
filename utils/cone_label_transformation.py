# script to modify segmented cones (after backprojection) to remove label by filling middle of cones
import os 
import cv2 as cv
import numpy as np

segmented_path = "/home/murphe80/metamorphic_testing/YOLO_metamorphic_testing/data/base_images/segmented_fsoco_test_gt"
bounding_boxes_path = "/home/murphe80/metamorphic_testing/YOLO_metamorphic_testing/data/base_images/test/labels"
output_path = "/home/murphe80/metamorphic_testing/YOLO_metamorphic_testing/data/transformed_images/fsoco_test_segmented/remove_labels_gt"

os.makedirs(output_path, exist_ok=True)

image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(segmented_path) 
               if f.lower().endswith(image_extensions)]

print(f"Found {len(image_files)} images to transform")

def locate_cones(image, min_area, DIST_THRES):
    """
    Locate cones by finding connected components and matching upper/lower parts by finding a matching line equation
    
    Parameters:
    - min__area: Minimum area for  components
    - dist_thres: Minimum difference in distnace to match components 
    
    Returns dictionary of matches
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold to get foreground pixels
    _, binary = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)
    # dilation to fill holes 
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # dilated = cv.dilate(binary, kernel, iterations=1)
    
    # Connected components analysis
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8)
    
    
    # calculate slope and intercept of LHS of cone and save
    component_slopes = [0] * num_labels
    component_intercepts = [0] * num_labels
    used_components = set()
    for i in range(1, num_labels): 
        area = stats[i, cv.CC_STAT_AREA]
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        if area < min_area: 
            component_slopes[i] = np.Inf
            component_intercepts[i] = np.Inf
            used_components.add(i)
        
        # calculate slope and intercept (y = mx + b)
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h 
        
        slope = (y2-y1) / (x2-x1)
        intercept = y1 - slope * x1
        component_slopes[i] = slope
        component_intercepts[i] = intercept
    
    # match based on line equation and proximity 
    matched_cones = {} 
    for i in range(1, num_labels): 
        # skip if alrady matched 
        if i in used_components:
            continue
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]

        curr_slope = component_slopes[i]
        curr_intercept = component_intercepts[i]
        # find best match 
        best_match = None
        best_match_index = -1
        
        for j in range(1, num_labels):
            slope = component_slopes[j]
            if i != j and j not in used_components:
                intercept = component_intercepts[j]
                # Calculate line similarity: combine slope and intercept differences
                line_diff = np.abs(curr_slope - slope) + np.abs(curr_intercept - intercept) / 100
                if best_match is None or (line_diff < best_match):
                    x2 = stats[j, cv.CC_STAT_LEFT]
                    y2 = stats[j, cv.CC_STAT_TOP]
                    comp_1 = np.array([x, y])
                    comp_2 = np.array([x2, y2])
                    horizontal_dist = np.abs(x - x2)
                    if np.linalg.norm(comp_1 - comp_2) < DIST_THRES  and horizontal_dist < (w*2):
                        best_match = line_diff
                        best_match_index = j
        
        # match cone pairs and mark slopes as used
        if best_match_index != -1:
            used_components.add(i)
            used_components.add(best_match_index)
            matched_cones[tuple(stats[i])] = tuple(stats[best_match_index])
            component_slopes[i] = np.Inf
            component_slopes[best_match_index] = np.Inf
    return matched_cones

def fill_cone_labels(cone_data, filtered_image, original_image, black_label=False):
    """
    Fill the region between matched upper and lower cone components with the average cone color
    
    Parameters:
    - cone_data: Dictionary of matched cones from locate_cones
    - filtered_image: Binary/processed image used for segmentation
    - original_image: Original color image
    - black_label: If True, fill gap with white (255), otherwise use average cone color
    
    Returns:
    - result: Image with filled gaps between cone parts
    """
    result = original_image.copy()
    
    for upper_stats, lower_stats in cone_data.items():
        # Unpack stats: [left, top, width, height, area]
        ux, uy, uw, uh, _ = upper_stats
        lx, ly, lw, lh, _ = lower_stats
        
        # Define the coordinates for the gap
        # We use a polygon approach (Trapezoid) to handle perspective where 
        # the lower component might be wider than the upper component.
        # Points: Bottom-Left Upper -> Top-Left Lower -> Top-Right Lower -> Bottom-Right Upper
        pts = np.array([
            [ux, uy + (uh*0.9)],          # Bottom-left of upper component
            [(lx+lw*0.2), ly+(lh*0.1)],               # Top-left of lower component
            [lx + (lw*0.7), ly],          # Top-right of lower component
            [ux + (uw*1.1), uy + (uh*0.9)]      # Bottom-right of upper component
        ], np.int32)
        
        pts = pts.reshape((-1, 1, 2))
        
        if black_label:
            color = (0,0,0)
        else:
            # Calculate average color from the adjacent regions to blend seamlessly
            # We take the bottom slice of the upper cone and top slice of the lower cone
            roi_upper = original_image[uy + uh//2 : uy + uh, ux : ux + uw]
            roi_lower = original_image[ly : ly + lh//2, lx : lx + lw]
            
            # Compute average color (BGR)
            # Handle cases where ROIs might be effectively empty or invalid
            avg_b, avg_g, avg_r = 0, 0, 0
            count = 0
            
            if roi_upper.size > 0:
                mean_u = cv.mean(roi_upper)
                avg_b += mean_u[0]
                avg_g += mean_u[1]
                avg_r += mean_u[2]
                count += 1
                
            if roi_lower.size > 0:
                mean_l = cv.mean(roi_lower)
                avg_b += mean_l[0]
                avg_g += mean_l[1]
                avg_r += mean_l[2]
                count += 1
            
            if count > 0:
                color = (int(avg_b / count), int(avg_g / count), int(avg_r / count))
            else:
                color = (0, 0, 0) # Fallback

        # Fill the polygon gap
        cv.fillPoly(result, [pts], color)
        
    return result

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

def visualize_cone_detection(image, matched_cones, show_labels=True):
    """
    Visualize the output of locate_cones function
    
    Parameters:
    - image: Original input image
    - matched_cones: Dictionary of matched cone pairs from locate_cones
    - show_labels: Whether to show match pair numbers
    
    Returns:
    - fig: matplotlib figure object
    """
    # Create figure with original image
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display the original image
    if len(image.shape) == 3:
        ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    else:
        ax.imshow(image, cmap='gray')
    
    # Generate random colors for each matched pair
    num_pairs = len(matched_cones)
    colors = []
    for _ in range(num_pairs):
        colors.append((random.random(), random.random(), random.random()))
    
    # Draw bounding boxes for matched pairs
    for idx, (comp1, comp2) in enumerate(matched_cones.items()):
        color = colors[idx]
        
        # Component 1 (upper part)
        x1, y1, w1, h1, area1 = comp1
        rect1 = Rectangle((x1, y1), w1, h1, linewidth=2, 
                          edgecolor=color, facecolor='none', linestyle='-')
        ax.add_patch(rect1)
        
        # Component 2 (lower part)
        x2, y2, w2, h2, area2 = comp2
        rect2 = Rectangle((x2, y2), w2, h2, linewidth=2, 
                          edgecolor=color, facecolor='none', linestyle='--')
        ax.add_patch(rect2)
        
        # Draw line connecting the centroids
        cx1, cy1 = x1 + w1/2, y1 + h1/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2
        ax.plot([cx1, cx2], [cy1, cy2], color=color, linewidth=1.5, alpha=0.6)
        
        # Add labels if requested
        if show_labels:
            ax.text(cx1, cy1, f'{idx+1}', color='white', fontsize=10, 
                   fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    ax.set_title(f'Cone Detection Results - {num_pairs} Matched Pairs', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='gray', linewidth=2, linestyle='-', 
                   label='Upper component'),
        plt.Line2D([0], [0], color='gray', linewidth=2, linestyle='--', 
                   label='Lower component'),
        plt.Line2D([0], [0], color='gray', linewidth=1.5, 
                   label='Match connection')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig
        

for idx, image_name in enumerate(image_files):
    print(f"Processing {idx+1}/{len(image_files)}: {image_name}")
    
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
    
    # Locate cones in the filtered image
    cone_data = locate_cones(filtered_cones, 250, 100)
    #fig = visualize_cone_detection(filtered_cones, cone_data, True)
    #plt.show()
    # Fill cone labels in the original image
    labels_removed = fill_cone_labels(cone_data, filtered_cones, original)
    
    # # Save the result
    output_file = os.path.join(output_path, image_name)
    cv.imwrite(output_file, labels_removed)
    print(f"saved to {output_file}")
