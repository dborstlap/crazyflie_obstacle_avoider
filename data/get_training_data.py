

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import os


# list of all .pickle datafiles to be combined
data_files = [
    'cyberzoo_set1_augmented.pickle',
    'cyberzoo_set2_augmented.pickle',
    'cyberzoo_set3_augmented.pickle'
]


def get_data(pickle_files, test_size=0.2):
    all_images = []
    all_labels = []

    # Get the directory where the file is located
    script_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(script_dir , 'datasets')

    # Load and append data from each pickle file
    for file in pickle_files:
        file_dir = os.path.join(data_folder, file)
        with open(file_dir, 'rb') as f:
            data = pickle.load(f)
            images, labels = data['images'], data['labels']
            all_images.extend(images)
            all_labels.extend(labels)

    # Convert lists to numpy arrays for processing
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    # Split the dataset into training and testing sets
    data_train, data_test, labels_train, labels_test = train_test_split(all_images, all_labels, test_size=test_size)

    # Return the training and validation datasets
    return data_train, labels_train, data_test, labels_test



if __name__ == '__main__':

    data_train, labels_train, data_test, labels_test = get_data(data_files)