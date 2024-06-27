
# imports
import numpy as np
from sklearn.model_selection import train_test_split
import os
import h5py


datasets = [
    'my_dataset_1.h5',
]


def get_data(datasets):
    images = []
    labels1 = []    

    # Load and append data from each pickle file
    for dataset in datasets:
        script_dir = os.path.abspath(os.path.dirname(__file__))
        dataset_path = os.path.join(script_dir, 'datasets', dataset)

        with h5py.File(dataset_path, 'r') as h5file:
            images_loaded = h5file['images'][:]
            labels1_loaded = h5file['labels1'][:]
            # labels2_loaded = h5file['labels2'][:]
        
        images.extend(images_loaded)
        labels1.extend(labels1_loaded)

    # Convert lists to numpy arrays for processing
    images = np.array(images)
    labels1 = np.array(labels1)

    return images, labels1



if __name__ == '__main__':

    images, labels1 = get_data(datasets)

    print(images.shape)
    print(labels1.shape)