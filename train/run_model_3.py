# THIS workds better than run_model because no pre-computation/scaling is needed on the image before feeding into quant network! Dont know how it workds tho :'D


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

def set_input_tensor(interpreter, input):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = input

def classify_image(interpreter, input, quant=False):
    set_input_tensor(interpreter, input)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details["index"])
    
    if quant:
        # Outputs from the TFLite model are uint8, so we dequantize the results:
        scale, zero_point = output_details["quantization"]
        output = scale * (output - zero_point)
    return output

def run_inference_batch(model_path, batch_images, quant=False):
    interpreter = load_tflite_model(model_path)

    outputs = []
    for i in range(len(batch_images)):
        prediction = classify_image(interpreter, batch_images[i], quant)
        outputs.append(prediction)
    return outputs




if __name__ == '__main__':
    # define model and image folders or dataset
    data, labels = get_data(datasets)
    images = np.transpose(data, (0, 2, 1, 3))

    # Get the directory of the trained models
    script_dir = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(script_dir, 'trained_models/255_input.tflite')
    model_path_q = os.path.join(script_dir, 'trained_models/255_input_q.tflite')

    outputs = run_inference_batch(model_path, data)
    outputs_q = run_inference_batch(model_path_q, data, quant=True)

    for image, label, output, output_q in zip(images, labels, outputs, outputs_q):

        print('Label:', label)
        print('output:', output)
        print('output_q:', output_q)

        cv2.imshow('out', image.astype('uint8'))
        cv2.waitKey(0)







