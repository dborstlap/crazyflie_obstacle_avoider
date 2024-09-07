# TODO maybe classification is retrained and is now bullshit?

import argparse
import os
import numpy as np
import tensorflow as tf
import cv2

MODEL_NAME = "255_input.tflite"
MODEL_NAME_QUANT = "255_input_q.tflite"

# path of repository (crazyflie_obstacle_avoider)
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_path = '/data/images/training_data_christmas_packet'
DATASET_PATH = f"{ROOT_PATH}{dataset_path}"

image_width = 324
image_height = 244

batch_size = 8

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
)

train_generator = train_datagen.flow_from_directory(
    f"{DATASET_PATH}/train",
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="grayscale",
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator() # imports as float32
val_generator = val_datagen.flow_from_directory(
    f"{DATASET_PATH}/validation",
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="grayscale",
)

batch_images, batch_labels = next(val_generator)


# Load the TFLite model
# script_dir = os.path.abspath(os.path.dirname(__file__))
# model_path = os.path.join(script_dir, 'trained_models/classification_q.tflite')
# interpreter = tf.lite.Interpreter(model_path=model_path)
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# def predict(batch_images):
#     # Assuming batch_images is a numpy array with shape (batch_size, height, width, channels)
#     interpreter.set_tensor(input_details[0]['index'], batch_images)
#     interpreter.invoke()
#     return interpreter.get_tensor(output_details[0]['index'])

# Make predictions
# logits = predict(batch_images)
# prediction = np.argmax(logits, axis=1)
# truth = np.argmax(batch_labels, axis=1)
# keras_accuracy = tf.keras.metrics.Accuracy()
# keras_accuracy(prediction, truth)
# print("Raw model accuracy: {:.3%}".format(keras_accuracy.result()))

def set_input_tensor(interpreter, input):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = input

def classify_image(interpreter, input):
    set_input_tensor(interpreter, input)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details["index"])
    # Outputs from the TFLite model are uint8, so we dequantize the results:
    scale, zero_point = output_details["quantization"]
    output = scale * (output - zero_point)
    top_1 = np.argmax(output)
    return output

interpreter = tf.lite.Interpreter(
    f"{ROOT_PATH}/train/trained_models/classification_q.tflite"
)
interpreter.allocate_tensors()

# Collect all inference predictions in a list
batch_prediction = []
batch_truth = np.argmax(batch_labels, axis=1)

for i in range(len(batch_images)):
    prediction = classify_image(interpreter, batch_images[i])
    batch_prediction.append(prediction)

    print("Prediction: ", prediction)
    print("label: ", batch_labels[i])

    cv2.imshow('out', batch_images[i].astype('uint8'))
    cv2.waitKey(0)

# Compare all predictions to the ground truth
# tflite_accuracy = tf.keras.metrics.Accuracy()
# tflite_accuracy(batch_prediction, batch_truth)
# print("Quant TF Lite accuracy: {:.3%}".format(tflite_accuracy.result()))


