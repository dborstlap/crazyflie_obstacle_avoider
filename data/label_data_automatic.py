"""
Generates labelled data based on filename.
Can only be used if labels are stored as last word of image filenamen (see operate/fly_fpv)
"""


import os
import pickle
import cv2

def load_images_from_folder(folder):
    # training_data = []
    images = []  # List to store image data
    labels = []  # List to store labels

    for filename in os.listdir(folder):
        # Check if the file is an image
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Extract label from filename
                # Assuming format: img_timestamp_label.png
                label_string = filename.split('_')[-1].split('.')[0]
                if label_string == 'forward':
                    label = 0
                if label_string == 'right':
                    label = 1
                if label_string == 'left':
                    label = -1                    

                # training_data.append((img, label))
                images.append(img)
                labels.append(label)

    # Creating a dictionary to store data and labels
    training_data = {
        'images': images,
        'labels': labels
    }

    return training_data




if __name__ == '__main__':
    # folder name to store data in
    folder = 'cyberzoo_set3'

    # Folder path relative to the script location
    script_dir = os.path.abspath(os.path.dirname(__file__))
    folder_path = os.path.join(script_dir, '..', 'data/datasets/'+folder)

    training_data = load_images_from_folder(folder_path)

    with open(folder_path + '.pickle', 'wb') as f:
        pickle.dump(training_data, f, pickle.HIGHEST_PROTOCOL)

    print('labelled data saved as ', folder_path + '.pickle')


