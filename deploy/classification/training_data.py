import os
import numpy as np
from PIL import Image

def get_label(image_path):
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    av_brightness = np.array(image).mean()  # Calculate average brightness
    return av_brightness


def prepare_data():
    data_dir = os.path.join(os.path.dirname(__file__), 'training_data')
    images = []
    labels = []
    
    # Recursively collect data from the directory and its subdirectories
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other image formats if needed
                image_path = os.path.join(root, filename)
                image = Image.open(image_path).convert('L')  # Convert image to grayscale
                
                # Resize image to (244, 324) if it has a different size
                if image.size != (244, 324):
                    image = image.resize((244, 324))
                
                images.append(np.array(image))
                labels.append(get_label(image_path))
    
    # Split the data into training and test sets
    split_ratio = 0.8
    split_index = int(len(images) * split_ratio)
    X_train, y_train = images[:split_index], labels[:split_index]
    X_test, y_test = images[split_index:], labels[split_index:]
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare_data()
    print("Training data shapes:", np.array(X_train).shape, np.array(y_train).shape)
    print("Validation data shapes:", np.array(X_test).shape, np.array(y_test).shape)