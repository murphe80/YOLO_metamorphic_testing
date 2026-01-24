# histogram backprojection to extract cones from an image 
# Source: https://docs.opencv.org/4.x/dc/df6/tutorial_py_histogram_backprojection.html 

import numpy as np
import cv2 as cv
import os 

test_data_path = "/home/murphe80/metamorphic_testing/YOLO_metamorphic_testing/data/base_images/zed_frames"
output_path = "/home/murphe80/metamorphic_testing/YOLO_metamorphic_testing/data/base_images/zed_segmented/all_images"

os.makedirs(output_path, exist_ok=True)

roi_blue = cv.imread('blue_cones_samples_comp.PNG')
assert roi_blue is not None, "file could not be read, check with os.path.exists()"
hsv_blue = cv.cvtColor(roi_blue,cv.COLOR_BGR2HSV)

roi_yell = cv.imread('yellow_cones_samples2.png')
assert roi_yell is not None, "file could not be read, check with os.path.exists()"
hsv_yell = cv.cvtColor(roi_yell,cv.COLOR_BGR2HSV)

roi_orange = cv.imread('orange_cones_samples2.PNG')
assert roi_orange is not None, "file could not be read, check with os.path.exists()"
hsv_orange = cv.cvtColor(roi_orange,cv.COLOR_BGR2HSV)

# calculating object histogram
roihist_blue = cv.calcHist([hsv_blue],[0, 1], None, [180, 256], [0, 180, 0, 256] )
roihist_yell = cv.calcHist([hsv_yell],[0, 1], None, [180, 256], [0, 180, 0, 256] )
roihist_orange = cv.calcHist([hsv_orange],[0, 1], None, [180, 256], [0, 180, 0, 256] )

# normalize histogram
cv.normalize(roihist_blue,roihist_blue,0,255,cv.NORM_MINMAX)
cv.normalize(roihist_yell,roihist_yell,0,255,cv.NORM_MINMAX)
cv.normalize(roihist_orange,roihist_orange,0,255,cv.NORM_MINMAX)

image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(test_data_path) 
               if f.lower().endswith(image_extensions)]

# apply backprojection to every test image from FSOCO 
for idx, image_name in enumerate(image_files):
    image_path = os.path.join(test_data_path, image_name)
    target = cv.imread(image_path)
    if target is None:
        continue
    hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)

    # apply backprojection
    dst_blue = cv.calcBackProject([hsvt],[0,1],roihist_blue,[0,180,0,256],1)
    dst_yell = cv.calcBackProject([hsvt],[0,1],roihist_yell,[0,180,0,256],1)
    dst_orange = cv.calcBackProject([hsvt],[0,1],roihist_orange,[0,180,0,256],1)

    dst_combined = cv.max(dst_blue, cv.max(dst_yell, dst_orange))

    # convolute with circular disc
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    cv.filter2D(dst_combined,-1,disc,dst_combined)

    # threshold
    ret,thresh = cv.threshold(dst_combined,20,255,0)

    # opening to remove noise, closing to fill bridges 
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    thres = cv.morphologyEx(thresh, cv.MORPH_ERODE, kernel)
    thresh = cv.morphologyEx(thresh, cv.MORPH_DILATE, kernel)

    # CCA and filter by geometric shape 
    analysis = cv.connectedComponentsWithStats(thresh, 8, cv.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # Create output mask for valid cones
    filtered_mask = np.zeros_like(thresh)

    # Create visualization image with bounding boxes
    vis_image = target.copy()

    # filter by minimum size and geometric shape
    for i in range(1, totalLabels):
        area = values[i, cv.CC_STAT_AREA]  
        
        # Get bounding box dimensions
        x = values[i, cv.CC_STAT_LEFT]
        y = values[i, cv.CC_STAT_TOP]
        w = values[i, cv.CC_STAT_WIDTH]
        h = values[i, cv.CC_STAT_HEIGHT]
        
        # Calculate aspect ratio
        # aspect_ratio = h / w if w > 0 else 0
        
        if (area > 140):
            # Considered using aspect ratio here but decided not to since top and bottom of cones are segmented separately
            componentMask = (label_ids == i).astype("uint8") * 255
            filtered_mask = cv.bitwise_or(filtered_mask, componentMask)
            # Draw valid components in green
            cv.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                

    # Apply filtered mask to original image
    filtered_mask_3ch = cv.merge((filtered_mask, filtered_mask, filtered_mask))
    res_filtered = cv.bitwise_and(target, filtered_mask_3ch)

    result_stack = np.vstack((
        target,                
        res_filtered             
    ))
    output_filename = os.path.join(output_path, image_name)
    cv.imwrite(output_filename, result_stack)
