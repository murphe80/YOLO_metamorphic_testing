#Divides test data into calibration and hold-out test data

import os
import json
import cv2 as cv
import numpy as np
import argparse
import shutil

# Define directories
TEST_DATA_DIR = "../../data/base_images/test"
CALIBRATION_DATA_DIR = "../../data/base_images/test_calibration"
HOLD_OUT_TEST_DATA_DIR = "../../data/base_images/test_hold_out"

os.makedirs(CALIBRATION_DATA_DIR, exist_ok=True)
os.makedirs(HOLD_OUT_TEST_DATA_DIR, exist_ok=True)

# Load the test data
test_data = os.listdir(TEST_DATA_DIR)

# Split the test data into calibration and hold-out test data
#  70% of the data is used for calibration, 30% is used for hold-out test
calibration_data = test_data[:int(len(test_data) * 0.7)]
hold_out_test_data = test_data[int(len(test_data) * 0.3):]

# Copy the calibration data to the calibration data directory
for image in calibration_data:
    shutil.copy(os.path.join(TEST_DATA_DIR, image), os.path.join(CALIBRATION_DATA_DIR, image))

# Copy the hold-out test data to the hold-out test data directory
for image in hold_out_test_data:
    shutil.copy(os.path.join(TEST_DATA_DIR, image), os.path.join(HOLD_OUT_TEST_DATA_DIR, image))