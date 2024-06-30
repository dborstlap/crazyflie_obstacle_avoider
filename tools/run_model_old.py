
import tensorflow as tf
import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.get_training_data import get_data, datasets


def set_input_tensor(interpreter, input, quant):
    input_details = interpreter.get_input_details()[0]

    #Resize
    image_in = cv2.resize(input,(input_details['shape'][2],input_details['shape'][1]))
    image_in = np.expand_dims(image_in, axis=0)
    image_in = np.expand_dims(image_in, axis=3)

    if not quant:
        interpreter.set_tensor(input_details['index'], image_in.astype('float32'))
    else:
        scale, zero_point = input_details['quantization']
        interpreter.set_tensor(input_details['index'], np.uint8(image_in / scale + zero_point))


def get_output_quant(interpreter, input, quant=False):
    image_in = set_input_tensor(interpreter, input, quant)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details["index"])

    # Outputs from the TFLite model are uint8, so we dequantize the results:
    if quant:
        scale, zero_point = output_details["quantization"]
        output = scale * (output - zero_point)
    return output


def run_model(model_path, image, quantized=False):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    #Resize
    image_in = cv2.resize(image,(input_details[0]['shape'][2],input_details[0]['shape'][1]))
    image_in = np.expand_dims(image_in, axis=0)
    image_in = np.expand_dims(image_in, axis=3)

    # Set the input tensor
    if not quantized:
        interpreter.set_tensor(input_details[0]['index'], image_in.astype('float32'))
    else:
        interpreter.set_tensor(input_details[0]['index'], image_in.astype('uint8'))

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
    # define model and image folders or dataset
    image_dir = "data/images/cyberzoo_set1"
    images, labels = get_data(datasets)

    # name of the trained model
    trained_model_file = 'my_classification_brightness_distribution.tflite'
    trained_model_file_q = 'my_classification_brightness_distribution_q.tflite'

    # Get the directory where the file is located
    script_dir = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(script_dir, 'trained_models', trained_model_file)
    model_dir_q = os.path.join(script_dir, 'trained_models', trained_model_file_q)


    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)

            # Read the image from the folder
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # returns [0-255] uint8 image
            image_norm = image.astype('float32') / 255.0

            width, height = image.shape
            strip_width = width // 3
            strips = [image_norm[i*strip_width:(i+1)*strip_width, :] for i in range(3)]
            
            # Calculate the average brightness of each strip
            brightness_left = strips[0].mean()
            brightness_middle = strips[1].mean()
            brightness_right = strips[2].mean()
            brightness_distribution = np.round([brightness_left, brightness_middle, brightness_right],3)


            # run_model(model_dir, image_norm)
            # run_model(model_dir_q, image, quantized=True)

            interpreter = tf.lite.Interpreter(model_path=model_dir)
            interpreter.allocate_tensors()

            interpreter_q = tf.lite.Interpreter(model_path=model_dir_q)
            interpreter_q.allocate_tensors()

            output = get_output_quant(interpreter, image_norm)
            output_q = get_output_quant(interpreter_q, image_norm, quant = True)

            print('Label:', brightness_distribution)
            print('output:', output)
            print('output_q:', output_q)

            cv2.imshow('out',image)
            cv2.waitKey(0)







