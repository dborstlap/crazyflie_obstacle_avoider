
# imports
import numpy as np
import os
import h5py


datasets = [
    'my_dataset_1.h5',
]


def get_data(datasets, target_shape=(324, 244, 1)):
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
    data = np.array(images).astype('float32') / 255.0   # Normalize
    data = np.transpose(data, (0, 2, 1, 3))

    labels = np.array(labels1)

    return data, labels



if __name__ == '__main__':

    data, labels = get_data(datasets)

    print(data.shape)
    print(labels.shape)