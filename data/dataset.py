# TODO make loadlabel more robust. Now is hardcoded.
# load_image defined here and in make_dataset. combine

import tensorflow as tf
import cv2
import pandas as pd
import os
import numpy as np
import sys
sys.path.append(os.getcwd())

class Dataset:
    def __init__(self, image_dir, csv_file, input_shape, output_shape):
        # Initialize the class
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.labels_df = pd.read_csv(csv_file)  # Load the CSV file into a Pandas dataframe

    def _loadImage(self, filename):
        # Load the image based on the filename
        image_file = tf.io.read_file(filename)
        image = tf.io.decode_png(image_file, channels=1)
        image = tf.image.resize(image, self.input_shape[:2])
        image = tf.cast(image, tf.float32)
        return image

    def _loadLabel(self, filename):
        # Load the label based on the filename
        filename = filename.numpy().decode('utf-8')
        relative_filename = os.path.relpath(filename, start=self.image_dir)

        rows = self.labels_df[self.labels_df['filename'] == relative_filename]
        label1 = rows['label_0'].values[0]
        label2 = rows['label_1'].values[0]
        label3 = rows['label_2'].values[0]

        return np.array([label1, label2, label3], dtype=np.float32)

    def createDataset(self, batch_size=32, train_test_split=0.90):
        # Recursively get a list of all image filenames in the directory and subdirectories
        image_file_names = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith('.png'):
                    image_file_names.append(os.path.join(root, file))

        # Create a TensorFlow dataset from the image filenames
        filenames_ds = tf.data.Dataset.from_tensor_slices(image_file_names)

        # Map the load_image and load_label functions to the filenames to create the final dataset
        image_ds = filenames_ds.map(self._loadImage, num_parallel_calls=tf.data.AUTOTUNE)
        label_ds = filenames_ds.map(lambda x: tf.py_function(self._loadLabel, [x], tf.float32))

        # Create validation and training sets
        train_size = int(train_test_split * len(image_file_names))
        train_dataset = tf.data.Dataset.zip((image_ds.take(train_size), label_ds.take(train_size)))
        val_dataset = tf.data.Dataset.zip((image_ds.skip(train_size), label_ds.skip(train_size)))

        # Batch the dataset
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        return train_dataset, val_dataset


if __name__ == '__main__':
    # Define the image directory and CSV file
    image_dir = 'data/training_data/images'
    csv_file = 'data/training_data/labels/example_labels.csv'

    # Define the input and output shapes of the model
    input_shape = (120,180,1)  # height, width, channels
    output_shape = (3,)

    # Initialize the class
    dataset = Dataset(image_dir, csv_file, input_shape, output_shape)

    # Create the dataset
    train_dataset, val_dataset = dataset.createDataset()

    # Visualize the dataset
    batch = next(iter(val_dataset))
    images, labels = batch

    # Plot the images in the batch to check
    for i in range(len(images)):
        image = cv2.cvtColor((images[i].numpy().astype('float')).astype('uint8'), cv2.COLOR_GRAY2RGB)
        print(f"Label {i}: {labels[i].numpy()}")
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    print('done')