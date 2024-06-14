
import tensorflow as tf
import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt




def run_model(model_path, image):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    image = image/255.
    # print(image.mean())

    height, width = image.shape
    strip_width = width // 3
    strips = [image[:, i*strip_width:(i+1)*strip_width] for i in range(3)]
    
    # Calculate the average brightness of each strip
    brightness_left = strips[0].mean()
    brightness_middle = strips[1].mean()
    brightness_right = strips[2].mean()

    brightness_distribution = np.round([brightness_left, brightness_middle, brightness_right],3)
    print(brightness_distribution)

    input_details = interpreter.get_input_details()

    #Resize
    image_in = cv2.resize(image,(input_details[0]['shape'][2],input_details[0]['shape'][1]))
    image_in = np.expand_dims(image_in, axis=0)
    image_in = np.expand_dims(image_in, axis=3)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image_in.astype('float32'))

    # Perform inference
    interpreter.invoke()

    # Get the output tensor and post-process if needed
    output_details = interpreter.get_output_details()
    output = interpreter.get_tensor(output_details[0]['index'])
    output = np.round(output,3)
    print(output)

    cv2.imshow('out',image)
    cv2.waitKey(0)
      


if __name__ == '__main__':
    # define model and image folders
    image_dir = "data/datasets/cyberzoo_set2"

    # name of the trained model
    trained_model_file = 'my_classification_brightness_distribution.tflite'

    # Get the directory where the file is located
    script_dir = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(script_dir, '..', 'deploy/classification/model', trained_model_file)


    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)

            # Read the image from the folder
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            run_model(model_dir, image)







