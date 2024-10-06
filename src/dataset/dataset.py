import tensorflow as tf
import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, image_dir, csv_file, input_shape, output_shape):
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.input_shape = input_shape
        self.output_shape = output_shape
        # Load the CSV file into a Pandas dataframe
        self.labels_df = pd.read_csv(csv_file)

    def _loadImage(self, filename):
        """a function to load and preprocess the images

        Args:
            filename (string): image directory

        Returns:
            numpy array: image
        """        
        image_file = tf.io.read_file(filename)
        image = tf.io.decode_png(image_file, channels = 1)
        image = tf.image.resize(image, self.input_shape[:2])
        image = tf.cast(image, tf.float32)

        return image   

    def _loadLabel(self, filename):
        # Load the label based on the filename
        filename = filename.numpy().decode('utf-8')

        rows =  self.labels_df[self.labels_df['filename'] == os.path.basename(filename)]
        label1 = rows['label1']
        label2 = rows['label2']
        label3 = rows['label3']
        
        return np.array([label1, label2, label3], dtype=np.float32)

    
    def createDataset(self, batch_size=32, train_test_split=0.90):
        """function to create a TensorFlow dataset from the image directory and CSV file

        Args:
            batch_size (int, optional): batch size. Defaults to 32.

        Returns:
            dict: training dataset, validation dataset
        """        

        # Get a list of all image filenames
        image_file_names = [os.path.join(self.image_dir,file) for file in os.listdir(self.image_dir) if file.endswith('.png')]
        
        # Create a TensorFlow dataset from the image filenames
        filenames_ds = tf.data.Dataset.from_tensor_slices(image_file_names)

        # Map the load_image and load_label functions to the filenames to create the final dataset
        image_ds = filenames_ds.map(self._loadImage, num_parallel_calls=tf.data.AUTOTUNE) # float32 [0.-255.] image (rescale makes uint8 into float)
        label_ds = filenames_ds.map(lambda x: tf.py_function(self._loadLabel, [x], tf.float32))

        #Create validation and training sets
        train_size = int(train_test_split * len(image_file_names))
        train_dataset = tf.data.Dataset.zip((image_ds.take(train_size), label_ds.take(train_size)))
        val_dataset = tf.data.Dataset.zip((image_ds.skip(train_size), label_ds.skip(train_size)))

        # Batch the dataset
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        
        return train_dataset, val_dataset


if __name__ == '__main__':
    # Define the image directory and CSV file
    image_dir = 'images/all_data'
    csv_file = 'output_labels.csv'

    # Define the input and output shapes of the model
    input_shape = (244, 324, 1) #height, width, channels
    output_shape = (3,)

    # Iniltiaize Class
    dataset = Dataset(image_dir, csv_file, input_shape, output_shape)

    # Create the dataset
    train_dataset, val_dataset = dataset.createDataset()

    #Visualize the dataset
    batch = next(iter(val_dataset))
    images, labels = batch

    # Plot the images in the batch to check
    for i in range(len(images)):
        image = cv2.cvtColor((images[i].numpy().astype('float')).astype('uint8'), cv2.COLOR_GRAY2RGB)
        print(labels[i])
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    print('done')