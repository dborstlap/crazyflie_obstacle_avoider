# why is label always 0.02 lower than if brightness is recalculated?

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

    # Iterate over the data
    for image, label in zip(images, labels):

        img_brightness = round(image.mean(), 4)
        image = image*255
        image = image.astype('uint8')


        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Create a black overlay image
        overlay = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        label_steer = label['steer']
        label_brightness = round(label['brightness'], 4)

        # Draw the steer label text in red
        if label_steer == 0:
            text = 'forward' + ' ; brightness='+str(label_brightness)
        elif label_steer == -1:
            text = 'left' + ' ; brightness='+str(label_brightness)
        elif label_steer == 1:
            text = 'right' + ' ; brightness='+str(label_brightness)
        cv2.putText(overlay, text, (int(image.shape[1]/8), int(image.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Combine the image and overlay
        combined = cv2.addWeighted(image, 1.0, overlay, 1.0, 0)

        # Display the image with overlay
        cv2.imshow("Labeled Data", combined)
        print(img_brightness)
        print(label_brightness)

        # Wait for the user to press the enter key
        cv2.waitKey(0)

        # Clear the display
        cv2.destroyAllWindows()



if __name__ == '__main__':
    # name of the pickle file to visualize
    filename = 'cyberzoo_set1_augmented.pickle'

    # Get the directory where the file is located
    script_dir = os.path.abspath(os.path.dirname(__file__))
    filedir = os.path.join(script_dir, '..', 'data/datasets', filename)


    # open pickled data file
    with open(filedir, "rb") as f:
        data = pickle.load(f)

    images = data['images']
    labels = data['labels']

    # Visualize the labeled data
    visualize_labeled_data(filedir)



    