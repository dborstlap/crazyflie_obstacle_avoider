# general imports
import tensorflow as tf
import cv2
import os
import numpy as np
import sys

# own imports
sys.path.append(os.getcwd())
from data.dataset import Dataset

# Load the TFLite model and allocate tensors
def load_tflite_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        sys.exit(1)

# Set the input tensor for the interpreter
def set_input_tensor(interpreter, input_data):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = input_data

# Classify a single image using the TFLite model
def classify_image(interpreter, input_data, quant=False):
    set_input_tensor(interpreter, input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details["index"])
    
    if quant:
        scale, zero_point = output_details["quantization"]
        output = scale * (output - zero_point)
    
    return output

# Run inference on a batch of images
def run_inference_batch(model_path, batch_images, quant=False):
    interpreter = load_tflite_model(model_path)
    outputs = [classify_image(interpreter, image, quant) for image in batch_images]
    return outputs

# Main function for running the inference
if __name__ == '__main__':
    data_dir = 'data/training_data/images' # takes all images in this directory and subdirectories recursively
    csv_file = 'data/training_data/labels/example_labels.csv'
    input_shape = (120,180,1) # input shape of network you want to test. Images will be loaded in correct shape
    output_shape = 3

    # Load images and labels
    dataset = Dataset(data_dir, csv_file, input_shape, output_shape)
    train, val = dataset.createDataset(batch_size=20)
    images, labels = next(iter(val))

    # Load the quantized model and run inference
    model_path_q = 'train/models/trained_models/examplebrightness_net_180x120_quant.tflite'
    
    outputs_q = run_inference_batch(model_path_q, images, quant=False)

    # Display results
    for image, label, output_q in zip(images, labels, outputs_q):
        print(f'Truth: {label}, Predicted: {np.array(output_q)}')
        image = image.numpy().astype('uint8')
        cv2.imshow('Output', image)
        cv2.waitKey(0)