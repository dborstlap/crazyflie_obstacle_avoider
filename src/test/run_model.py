# THIS workds better than old run_model because no pre-computation/scaling is needed on the image before feeding into quant network! Dont know how it workds tho :'D


# TODO: might not work because of np.transpose(data, (0, 2, 1, 3)) data ordered/transpose/loaded differently in run_model vs training???

import tensorflow as tf
import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset.dataset import Dataset


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
    image_dir = 'all_data'
    csv_name = 'output_labels.csv'
    input_shape = (120, 180, 1)
    output_shape = 3

    csv_file = os.path.join('data/labels', csv_name)
    data_dir = os.path.join('data/images', image_dir)
    dataset = Dataset(data_dir, csv_file, input_shape, output_shape)
    train, val = dataset.createDataset(batch_size=20)

    images, labels = next(iter(val))
    
    # Get the directory of the trained models
    model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models/trained_models')

    model_path_q = os.path.join(model_dir, 'GateNet_180x120_quant.tflite')

    # outputs = run_inference_batch(model_path, data)
    outputs_q = run_inference_batch(model_path_q, images, quant=False)

    for image, label, output_q in zip(images, labels, outputs_q):
        image = image.numpy()
        print(image.shape)
        print('Truth:', label,'output_q:', np.array(output_q))
        cv2.imshow('out', image.astype('uint8'))
        cv2.waitKey(0)







