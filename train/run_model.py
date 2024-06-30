
import tensorflow as tf
import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.get_training_data import get_data, datasets


def load_tflite_model(model_path):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference_batch(model_path, images, quant=False):

    interpreter = load_tflite_model(model_path)

    outputs = []

    for image in images:
        # set input tensor
        input_details = interpreter.get_input_details()[0]

        #Resize
        image_in = cv2.resize(image,(input_details['shape'][2],input_details['shape'][1]))
        image_in = np.expand_dims(image_in, axis=0)
        image_in = np.expand_dims(image_in, axis=3)

        if not quant:
            interpreter.set_tensor(input_details['index'], image_in.astype('float32'))
        else:
            scale, zero_point = input_details['quantization']
            interpreter.set_tensor(input_details['index'], np.uint8(image_in / scale + zero_point))
    
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = interpreter.get_tensor(output_details["index"])

        # Outputs from the TFLite model are uint8, so we dequantize the results:
        if quant:
            scale, zero_point = output_details["quantization"]
            output = scale * (output - zero_point)
        outputs.append(output[0])

    return outputs




if __name__ == '__main__':
    # define model and image folders or dataset
    data, labels = get_data(datasets)
    images = np.transpose(data, (0, 2, 1, 3))

    # Get the directory of the trained models
    script_dir = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(script_dir, 'trained_models/my_classification_brightness_distribution.tflite')
    model_path_q = os.path.join(script_dir, 'trained_models/my_classification_brightness_distribution_q.tflite')

    outputs = run_inference_batch(model_path, data)
    outputs_q = run_inference_batch(model_path_q, data, quant=True)

    for image, label, output, output_q in zip(images, labels, outputs, outputs_q):

        print('Label:', label)
        print('output:', output)
        print('output_q:', output_q)

        cv2.imshow('out',image)
        cv2.waitKey(0)







