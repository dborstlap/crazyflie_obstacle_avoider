import pickle
import cv2
import os
import numpy as np
from get_training_data import get_data, data_files
import sys


def visualize_labeled_data(filename):

    # open pickled data file
    with open(filename, "rb") as f:
        data = pickle.load(f)

    images = data['images']
    labels = data['labels']

    # image_data = np.array([row[0] for row in data if row])
    # label_data = [row[1] for row in data if row]

    # Iterate over the data
    for image, label in zip(images, labels):
        # Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a black overlay image
        overlay = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # Draw the label text in red
        if label == 0:
            text = 'forward'
        elif label == -1:
            text = 'left'
        elif label == 1:
            text = 'right'
        cv2.putText(overlay, text, (int(image.shape[1]/2), int(image.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Combine the image and overlay
        combined = cv2.addWeighted(image, 1.0, overlay, 1.0, 0)

        # Display the image with overlay
        cv2.imshow("Labeled Data", combined)

        # Wait for the user to press the enter key
        cv2.waitKey(0)

        # Clear the display
        cv2.destroyAllWindows()



if __name__ == '__main__':
    # name of the pickle file to visualize
    filename = 'cyberzoo_set2_augmented.pickle'

    # Get the directory where the file is located
    script_dir = os.path.abspath(os.path.dirname(__file__))
    filedir = os.path.join(script_dir, '..', 'data/datasets', filename)

    # Visualize the labeled data
    visualize_labeled_data(filedir)



    