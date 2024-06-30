
import tensorflow as tf
import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt


def set_input_tensor(interpreter, input):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    # Inputs for the TFLite model must be uint8, so we quantize our input data.
    # NOTE: This step is necessary only because we're receiving input data from
    # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
    # bitmap inputs, they're already uint8 [0,255] so this can be replaced with:
        #    input_tensor[:, :] = input

    input = input.reshape(244, 324, 1)
    input_tensor[:, :] = input
    scale, zero_point = input_details['quantization']
    input_tensor[:, :] = np.uint8(input / scale + zero_point)
    

def get_output(interpreter, input):
  set_input_tensor(interpreter, input)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = interpreter.get_tensor(output_details['index'])
  # Outputs from the TFLite model are uint8, so we dequantize the results:
  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)
  return output


if __name__ == '__main__':
    # define model and image folders
    image_dir = "data/datasets/cyberzoo_set2"

    # name of the trained model
    trained_model_file_q = 'my_classification_brightness_distribution_1000iter_q.tflite'

    # Get the directory where the file is located
    script_dir = os.path.abspath(os.path.dirname(__file__))
    model_dir_q = os.path.join(script_dir, '..', 'deploy/classification/model', trained_model_file_q)


    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)

            # Read the image from the folder
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # returns [0-255] uint8 image
            image_norm = image.astype('float32') / 255.0

            height, width = image_norm.shape
            strip_width = width // 3
            strips = [image_norm[:, i*strip_width:(i+1)*strip_width] for i in range(3)]
            
            # Calculate the average brightness of each strip
            brightness_left = strips[0].mean()
            brightness_middle = strips[1].mean()
            brightness_right = strips[2].mean()
            brightness_distribution = np.round([brightness_left, brightness_middle, brightness_right],3)

            interpreter_q = tf.lite.Interpreter(model_path=model_dir_q)
            interpreter_q.allocate_tensors()

            output_q = get_output(interpreter_q, image_norm)

            print('Label:', brightness_distribution)
            print('output_q:', output_q)

            cv2.imshow('out',image_norm)
            cv2.waitKey(0)







