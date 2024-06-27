## TODO simplicity!!!! delete all label files, just do basic example let students do rest!

# imports
import os
import h5py
import numpy as np
import tensorflow as tf


# ------------------ PARAMETERS ---------------------------
# name of the dataset (it will be stored under this name)
dataset_name = 'my_dataset_1'

# define which sets of image you want to use
image_sets = [
    'cyberzoo_set1',
    'cyberzoo_set2',
    'cyberzoo_set3',
]

# images will be reshaped to match the desired shape
resolution = (244, 324)  # Resolution that the image should be


# ------------------ RETRIEVE DATA ---------------------------

images = []  # List to store image data
for image_set in image_sets:
    file_path = os.path.abspath(os.path.dirname(__file__))
    image_folder = os.path.join(file_path, 'images', image_set)
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)

        # Check if the file is an image
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=resolution)
            img = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img)


# ------------------ AUGMENT DATA ---------------------------

# perform data augmentation as desired


# ------------------ LABEL DATA ---------------------------

labels1 = []

for img in images:
    # Split the image into 3 vertical strips
    width = img.shape[1]
    strip_width = width // 3
    strips = [img[:, i*strip_width:(i+1)*strip_width] for i in range(3)]

    # Calculate the average brightness of each strip
    brightness_left = strips[0].mean() / 255.
    brightness_middle = strips[1].mean() / 255.
    brightness_right = strips[2].mean() / 255.

    # Output the brightness values in an array
    label1 = [brightness_left, brightness_middle, brightness_right]
    labels1.append(label1)


# ------------------ SAVE DATASET ---------------------------
dataset_path = 1
dataset_path = os.path.join(file_path, 'datasets', dataset_name + '.h5')
with h5py.File(dataset_path, 'w') as h5file:
    # Create datasets for images and labels
    h5file.create_dataset('images', data=images)
    h5file.create_dataset('labels1', data=labels1)
    # h5file.create_dataset('labels2', data=labels2)


print("Data has been successfully stored in " + dataset_path)

