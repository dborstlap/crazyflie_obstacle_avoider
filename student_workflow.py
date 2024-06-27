# TODO Make so pickle files are not needed but data comes from images directly

# insert some asserts to make sure student is following right steps
"""
Welcome to the course ... 
This is the main file that you can edit to do what you want.
"""

import os
import numpy as np
import tensorflow as tf

from data.get_training_data import get_data


# STEP 1
# GET IMAGES

# STEP 2
# MAKE DATASET

# STEP 3
#




image_width = 324
image_height = 244



# GET DATA
data_dir = 'data/datasets'

data_files = [
    'cyberzoo_set1.pickle',
    'cyberzoo_set2.pickle',
    'cyberzoo_set3.pickle',
    'all_data.pickle',
]

all_images, all_labels_array = get_data(data_files)
