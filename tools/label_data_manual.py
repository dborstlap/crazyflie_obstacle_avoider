
"""
images that don't include one of labels in the name have to be labelled by user
other images skipped
"""


import cv2
import numpy as np
import pickle
import os



"""
-1 = left
 0 = forward
 1 = right
"""

# folder name to store data in
folder = 'data1'

# Folder path relative to the script location
script_dir = os.path.abspath(os.path.dirname(__file__))
folder_path = os.path.join(script_dir, '..', 'data/datasets/'+folder)

# Initialiŵe a list to store the training data
training_data = []

# Iterate through each image in the folder
for image_file in os.listdir(folder_path):
    # Read the image
    image = cv2.imread(os.path.join(folder_path, image_file))

    # Show the image to the user
    cv2.imshow("Image", image)

    # Wait for the user to press a key
    while True:
        pressed_key = cv2.waitKey(1) & 0xFF

        if pressed_key == ord("w"):
            # Label the image as "forward"
            label = 0
            break
        elif pressed_key == ord("a"):
            # Label the image as "left"
            label = -1
            break
        elif pressed_key == ord("d"):
            # Label the image as "right"
            label = 1
            break

    # Convert the image to a NumPy array
    image_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add the image and label to the training data
    training_data.append((image_array, label))

# Save the training data to a file
pickle_file_name = f"{folder_path}.pickle"
with open(pickle_file_name, "wb") as f:
    pickle.dump(training_data, f)
