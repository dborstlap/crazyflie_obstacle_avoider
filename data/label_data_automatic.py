"""
Generates labelled data based on filename.
Can only be used if labels are stored as last word of image filenamen (see operate/fly_fpv)
"""


import os
import pickle
import cv2



def generate_labels(folder):
    # training_data = []
    images = []  # List to store image data
    labels = []  # List to store labels

    for filename in os.listdir(folder):
        # Check if the file is an image
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:

                # normalize
                img = img/255.

                # Extract label from filename
                # Assuming format: img_timestamp_label.png
                label_steer_string = filename.split('_')[-1].split('.')[0]
                if label_steer_string == 'forward':
                    label_steer = 0
                if label_steer_string == 'right':
                    label_steer = 1
                if label_steer_string == 'left':
                    label_steer = -1          

                # compute angle based on brightness
                # Split the image into 3 vertical strips
                height, width = img.shape
                strip_width = width // 3
                strips = [img[:, i*strip_width:(i+1)*strip_width] for i in range(3)]
                
                # Calculate the average brightness of each strip
                brightness_left = strips[0].mean()
                brightness_middle = strips[1].mean()
                brightness_right = strips[2].mean()
                
                # Output the brightness values in an array
                brightness_distribution = [brightness_left, brightness_middle, brightness_right]

                # compute average image brightness. Since greyscale, it is just average value.
                av_brightness = img.mean()

                # define labels in label dictionary
                label = {
                    'steer': label_steer,
                    'brightness': av_brightness,
                    'brightness_distribution': brightness_distribution
                }

                # training_data.append((img, label))
                images.append(img)
                labels.append(label)

    # Creating a dictionary to store data and labels
    training_data = {
        'images': images,
        'labels': labels,
    }

    return training_data




if __name__ == '__main__':
    # folder name to store data in
    for i in range(3):
        folder = 'cyberzoo_set'+str(i+1)

        # Folder path relative to the script location
        script_dir = os.path.abspath(os.path.dirname(__file__))
        folder_path = os.path.join(script_dir, '..', 'data/datasets/'+folder)

        training_data = generate_labels(folder_path)

        with open(folder_path + '.pickle', 'wb') as f:
            pickle.dump(training_data, f, pickle.HIGHEST_PROTOCOL)

        print('labelled data saved as ', folder_path + '.pickle')


