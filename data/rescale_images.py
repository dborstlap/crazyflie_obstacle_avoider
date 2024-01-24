
import os
from PIL import Image

def resize_images(directory):

    for filename in os.listdir(directory):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                image_path = os.path.join(directory, filename)
                image = Image.open(image_path)
                image = image.resize((324, 244))
                image.save(image_path)


if __name__ == '__main__':
    script_dir = os.path.abspath(os.path.dirname(__file__))
    directory = os.path.join(script_dir, 'datasets/cyberzoo_set3')
    resize_images(directory)



