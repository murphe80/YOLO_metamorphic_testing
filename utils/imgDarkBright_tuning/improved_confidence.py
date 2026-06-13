# Create a distribution using brightness, confidence level and AP score
# Aim is to show that using brightness in conjunction with confidence level and AP score can improve the performance of the model

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the results
OUTPUT_DIR = "../../data/transformed_images/tuned_brightness/results.json"
TEST_DATA_DIR = "../../data/base_images/test"
CALIBRATION_DATA_DIR = "../../data/base_images/test_calibration"
HOLD_OUT_TEST_DATA_DIR = "../../data/base_images/test_hold_out"


results = json.load(open(OUTPUT_DIR))

# Create a distribution using brightness, confidence level and AP score
# 1. Create a distribtuion for each different brightness metric (rms, low_high_regions_ratio, mean_shift_variance, luminosity_709)
# Maybe use both AR and AP scores to create a distribution? 



#2. Create a distribution for each brightness metric (original and transformed), confidence level and AP score


#3. Use hold-out test data to validate distributions created in 1 and 2


