"""
Experiment with image preprocessing that might improve quality/reduce calculating time

"""


import numpy as np



def pool_image(image, new_height, new_width):
    """
    Applies average pooling to reduce the size of the image.

    :param image: A numpy array representing the grayscale image.
    :param new_height: The height of the output image.
    :param new_width: The width of the output image.
    :return: A numpy array representing the pooled image.
    """
    old_height, old_width = image.shape
    pooled_image = np.zeros((new_height, new_width))

    pool_height = old_height // new_height
    pool_width = old_width // new_width

    for i in range(new_height):
        for j in range(new_width):
            h_start = i * pool_height
            w_start = j * pool_width
            pooled_image[i, j] = np.mean(image[h_start:h_start + pool_height, w_start:w_start + pool_width])

    return pooled_image




def normalize_image(image):
    """
    Normalizes the pixel values of the image to be between 0 and 1.

    :param image: A numpy array representing the pooled image.
    :return: A numpy array representing the normalized image.
    """
    return image / 255



if __name__ == '__main__':
    # Example usage
    original_image = np.random.randint(0, 256, (244, 324))  # A random 244x324 grayscale image
    pooled_image = pool_image(original_image, 24, 24)
    normalized_image = normalize_image(pooled_image)




