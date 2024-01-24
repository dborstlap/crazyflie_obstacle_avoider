import cv2
import numpy as np
import pickle
import os

def horizontal_flip(image):
    # Horizontally flip the image
    flipped_image = cv2.flip(image, 1)
    return flipped_image

def shear_x(image, intensity):
    # Shear the image horizontally
    shear_x_matrix = np.array([[1, intensity, 0], [0, 1, 0]])
    sheared_image = cv2.warpAffine(image, shear_x_matrix, (image.shape[1], image.shape[0]))
    return sheared_image

def shear_y(image, intensity):
    # Shear the image vertically
    shear_y_matrix = np.array([[1, 0, 0], [intensity, 1, 0]])
    sheared_image = cv2.warpAffine(image, shear_y_matrix, (image.shape[1], image.shape[0]))
    return sheared_image


def rotate(image, angle):
    # Rotate the image
    rotate_matrix = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotate_matrix, (image.shape[1], image.shape[0]))
    return rotated_image

# def translate(image, x_offset, y_offset):
#     # Translate the image horizontally and vertically
#     translation_matrix = np.array([[1, 0, x_offset], [0, 1, y_offset]])
#     print(translation_matrix)
#     translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
#     return translated_image

def brightness_shift(image, intensity):
    # Add or subtract intensity from the image
    image_shifted = image + intensity
    image_shifted = np.clip(image, 0, 255)
    return image_shifted


def apply_data_augmentation(image, label, num_augmentations):
    # Apply data augmentation to the image and label
    augmented_images = []
    augmented_labels = []

    for _ in range(num_augmentations):

        # Randomly flip the image horizontally
        flip = np.random.choice([0, 1])
        if flip == 0:
            augmented_image = image
        else:
            augmented_image = horizontal_flip(image)

        # Randomly shear the image horizontally or vertically
        shear = np.random.choice([-0.05, 0.05, -0.1, 0.1])
        if shear != 0:
            if shear > 0:
                augmented_image = shear_x(augmented_image, shear)
            else:
                augmented_image = shear_y(augmented_image, shear)

        # Randomly rotate the image
        rotate_angle = np.random.choice([-5, 5, -10, 10])
        if rotate_angle:
            augmented_image = rotate(augmented_image, rotate_angle)

        # Randomly translate the image horizontally and vertically
        # translate_offset_x = np.random.choice([-5, 5])
        # translate_offset_y = np.random.choice([-5, 5])
        # augmented_image = translate(augmented_image, translate_offset_x, translate_offset_y)

        # Randomly adjust the brightness of the image
        brightness_offset = np.random.randint(-50, 51)
        augmented_image = brightness_shift(augmented_image, brightness_offset)
        
        # If the image is flipped vertically, reverse the label
        if flip == 1:
            if label == -1: # left
                augmented_label = 1 # right
            elif label == 1: # right
                augmented_label = -1 # left
            elif label == 0: # forward
                augmented_label = 0 # forward
        else:
            augmented_label = label

        # store the augmented image and labels in array
        augmented_images.append(augmented_image)
        augmented_labels.append(augmented_label)

    return augmented_images, augmented_labels



def augment_all_data(images, labels, num_augmentations):
    # Create lists to store augmented images and labels
    augmented_images = []
    augmented_labels = []

    # Iterate over the images and labels
    for image, label in zip(images, labels):
        
        # Apply data augmentation to the image and label
        augmented_image, augmented_label = apply_data_augmentation(image, label, num_augmentations)

        # Extend the augmented images and labels lists
        augmented_images.extend(augmented_image)
        augmented_labels.extend(augmented_label)

    # Create a dictionary to store the augmented data
    augmented_data = {
        'images': augmented_images,
        'labels': augmented_labels
    }

    return augmented_data



if __name__ == "__main__":

    # folder name containing data
    folder = 'cyberzoo_set1'

    # Get the directory where the script is located
    script_dir = os.path.abspath(os.path.dirname(__file__))

    # Specify the filename of the pickle file
    filename = os.path.join(script_dir, '..', 'data/datasets', folder+'.pickle')

    with open(filename, "rb") as f:
        data = pickle.load(f)

    images = data['images']
    labels = data['labels']


    augmented_data = augment_all_data(images, labels, num_augmentations=10)

    # print(type(augmented_data))
    # print(type(augmented_data[2]))
    # print(augmented_data[2])

    # Save the augmented data to a pickle file
    folder_path = os.path.join(script_dir, '..', 'data/datasets', folder)
    dumpname = f"{folder_path}_augmented.pickle"
    with open(dumpname, "wb") as f:
        pickle.dump(augmented_data, f, pickle.HIGHEST_PROTOCOL)
    print('Dumped pickle at', dumpname)



