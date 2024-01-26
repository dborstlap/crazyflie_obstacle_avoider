from PIL import Image
import numpy as np

def png_to_resized_array(image_path, new_size):
    image = Image.open(image_path)
    resized_image = image.resize(new_size)
    resized_array = np.array(resized_image)
    print(resized_array.shape)
    flattened_array = resized_array.reshape(-1)
    return flattened_array

# Example usage
image_path = 'vision/camera_stream/img_2.png'
new_size = (180, 120)  # Set the desired new size (width, height)
flattened_array = png_to_resized_array(image_path, new_size)

# sava to a c array
with open('vision/camera_stream/img_2.h', 'w') as f:
    f.write('static const uint8_t img_2[] = {\n')
    for i in range(len(flattened_array)):
        f.write(str(flattened_array[i]))
        if i != len(flattened_array) - 1:
            f.write(', ')
        if i % 16 == 15:
            f.write('\n')
    f.write('};\n')
