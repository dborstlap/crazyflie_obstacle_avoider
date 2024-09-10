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

left_folder = os.path.join(dst_folder, 'train/left')
right_folder = os.path.join(dst_folder, 'train/right')

# Create 'left' and 'right' directories if they don't exist
os.makedirs(left_folder, exist_ok=True)
os.makedirs(right_folder, exist_ok=True)

# Walk through the directory recursively
for root, _, files in os.walk(data_dir):
    for file in files:
        # Check if the file is an image based on extension
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Get the full path of the image
            file_path = os.path.join(root, file)
            
            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                print(f"Skipping {file_path}, unable to read image.")
                continue

            # Split the image into left and right halves
            height, width = image.shape[:2]
            left_half = image[:, :width // 2]
            right_half = image[:, width // 2:]

            # Display the left and right halves of the image
            # cv2.imshow("Left Half", left_half)
            # cv2.imshow("Right Half", right_half)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Calculate brightness of the left and right halves
            left_brightness = left_half.mean()
            right_brightness = right_half.mean()
            # print(left_brightness, right_brightness)

            # Determine which side is brighter and copy to corresponding folder
            try:
                if left_brightness > right_brightness:
                    shutil.copy(file_path, left_folder)
                    print(f"Copied {file} to {left_folder}")
                else:
                    shutil.copy(file_path, right_folder)
                    print(f"Copied {file} to {right_folder}")
            except:
                pass

