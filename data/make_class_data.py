import os
import cv2
import numpy as np
import shutil
import os
import cv2
import numpy as np
import shutil


def calculate_brightness(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate and return the mean brightness
    return np.mean(gray_image)


# Get the current working directory
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'data/images')
dst_folder = os.path.join(data_dir, 'class_brightness')

train_dark_folder = os.path.join(dst_folder, 'train/dark')
train_light_folder = os.path.join(dst_folder, 'train/light')
val_dark_folder = os.path.join(dst_folder, 'validation/dark')
val_light_folder = os.path.join(dst_folder, 'validation/light')

# Create 'left' and 'right' directories if they don't exist
os.makedirs(train_dark_folder, exist_ok=True)
os.makedirs(train_light_folder, exist_ok=True)
os.makedirs(val_dark_folder, exist_ok=True)
os.makedirs(val_light_folder, exist_ok=True)

# Walk through the directory recursively
for root, _, files in os.walk(data_dir):
    for i, file in enumerate(files):
        # Check if the file is an image based on extension
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Get the full path of the image
            file_path = os.path.join(root, file)
            
            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                print(f"Skipping {file_path}, unable to read image.")
                continue

            # Calculate brightness of the left and right halves
            brightness = image.mean()
            # print(left_brightness, right_brightness)

            # Determine which side is brighter and copy to corresponding folder
            try:
                if brightness < 40:
                    shutil.copy(file_path, train_dark_folder)
                    print(f"Copied {file} to {train_dark_folder}")
                else:
                    shutil.copy(file_path, train_light_folder)
                    print(f"Copied {file} to {train_light_folder}")
            except:
                pass

