import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2

def label_func(filename):

    image_file = tf.io.read_file(filename)
    image = tf.io.decode_png(image_file, channels = 1)
    # image = tf.image.resize(image, input_shape[:2])
    image = tf.cast(image, tf.float32).numpy()

    left_side = image[:, :image.shape[1]//2]
    right_side = image[:, image.shape[1]//2:]

    brightness = image.mean()
    brightness_left = left_side.mean()
    brightness_right = right_side.mean()

    return np.array([brightness, brightness_left, brightness_right], dtype=np.float32)




def make_labeled_dataset(image_files, output_csv, label_func=None):
    """
    Creates a CSV file with image file paths and corresponding labels.
    
    Args:
        image_files (list): List of image file paths.
        output_csv (str): Path to the output CSV file.
        label_func (callable, optional): Function to generate labels from file names. If None, labels default to 0.
    
    Returns:
        None
    """
    # List to store file paths and labels
    data = []
    
    # Loop through image files
    for image_file in image_files:
        # use label_func to get the label
        label = label_func(image_file)
        
        # Append the data
        data.append({'filename': os.path.basename(image_file), 'label1': label[0], 'label2': label[1], 'label3': label[2]})
    
    # Convert to a DataFrame
    df = pd.DataFrame(data)
    
    # Save the DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")

# Example usage
image_dir = 'images/all_data'
image_dir = os.path.join(os.path.dirname(__file__), image_dir)
image_file_names = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')]

# Create CSV output
make_labeled_dataset(image_file_names, 'output_labels.csv', label_func=label_func)










