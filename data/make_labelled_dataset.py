import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
sys.path.append(os.getcwd())

def load_image(filename, input_shape=(120, 180, 1)):
    """
    Load an image from a file and resize it to the required dimensions.

    Args:
        filename (str): Path to the image file.

    Returns:
        np.array: Image data.
    """
    image_file = tf.io.read_file(filename)
    image = tf.io.decode_png(image_file, channels=1)
    # image = tf.image.resize(image, input_shape)
    image = tf.cast(image, tf.float32).numpy()
    return image

def example_label_func(image):
    """
    Computes brightness labels for the whole image, the left side, and the right side.

    Args:
        image (np.array): Image data array

    Returns:
        np.array: Array containing brightness values for the full image, left half, and right half.
    """

    # Split image into left and right halves
    mid_point = image.shape[1] // 2
    left_side = image[:, :mid_point]
    right_side = image[:, mid_point:]

    # Compute mean brightness for full image, left, and right halves
    average_brightness = image.mean()
    average_brightness_left = left_side.mean()
    average_brightness_right = right_side.mean()

    return np.array([average_brightness, average_brightness_left, average_brightness_right], dtype=np.float32)


def make_labeled_dataset(image_dir, output_csv, label_func=None):
    """
    Creates a CSV file with image file paths and corresponding labels.

    Args:
        image_dir (str): Directory with all training data.
        output_csv (str): Path to the output CSV file.
        label_func (callable, optional): Function to generate labels from image.
        base_dir (str, optional): Base directory to compute relative paths.

    Returns:
        None
    """

    image_file_names = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.png'):
                image_file_names.append(os.path.join(root, file))

    data = []  # List to store file paths and labels

    # Loop through each image file and compute the label
    for image_file in image_file_names:

        # Load the image
        image = load_image(image_file)

        # Use label_func to generate labels
        label = label_func(image)

        # Compute filename relative to base_dir
        relative_filename = os.path.relpath(image_file, start=image_dir)

        # Append the data: filename and labels
        # Create a dictionary to store the filename and labels
        data_entry = {'filename': relative_filename}
        
        # Add each label to the dictionary
        for i, label_i in enumerate(label):
            data_entry['label_' + str(i)] = label_i
        
        # Append the dictionary to the data list
        data.append(data_entry)

    # Convert list to a DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")


if __name__ == '__main__':
    # Define the image directory and CSV output path
    image_dir = 'data/training_data/images'

    # Create the CSV file with image labels
    make_labeled_dataset(
        image_dir = image_dir,
        output_csv = 'data/training_data/labels/example_labels.csv',
        label_func = example_label_func,
    )

    print("Done!")