# TODO check accuracy of newly trained my_classification.tflite model
# TODO       - make file to run model and print predictions and labels
# TODO       - check if predictions are in correct range. Did I normalize input data? Dont think so but better to check.
# TODO go over https://github.com/google-coral/tutorials/blob/52b60653698a10e7c83c5761cf6a2acc3db57d22/retrain_classification_ptq_tf2.ipynb and improve code based on it
# TODO Use gatenet to improve this file/model
# TODO add batch size variable, make train_generator

# TODO mismatching inputs? Becasue example trained on size 324x224 and I am using 244x324

# TODO new trained model might be too big for crazyfly. Check size of model and reduce if necessary.

"""
documentation:
Got first 3-part brightness network running
trick was to use self-defined network. Not the classification one provided. It is not good in numerical value prediction
quantized and normal are also similar now. Was not the case before. 
Still dont know why quantization sometimes is so seperate from normal, even with the 1000iter model
"""

import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.get_training_data import get_data, datasets
from sklearn.model_selection import train_test_split
from models.model4 import model4


MODEL_NAME = "my_classification_brightness_distribution.tflite"
MODEL_NAME_QUANT = "my_classification_brightness_distribution_q.tflite"

# path of repository (crazyflie_obstacle_avoider)
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH_QUANT = 'data/images/cyberzoo_set1'

image_width = 324
image_height = 244
number_of_labels = 3

FIRST_LAYER_STRIDE = 2
epochs = 100

# define model
model = model4


# retrieve training data
data, labels = get_data(datasets)

# Assert data shape
expected_shape = (image_width, image_height, 1)
assert data.shape[1:] == expected_shape, "Data shape does not match expected shape"
assert labels.shape[1:][0] == number_of_labels, "Label shape does not match expected shape"


# make train and test split
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="mean_squared_error", metrics=["mae"])
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])  # mean_absolute_error

model.summary()
print("Number of trainable weights = {}".format(len(model.trainable_weights)))




# Train the custom head
history = model.fit(
    x=data_train,
    y=labels_train,
    epochs=epochs,
    validation_data=(data_test, labels_test)
)


##  FINE TUNE the model
# print("Number of layers in the base model: ", len(base_model.layers))

# base_model.trainable = True
# fine_tune_at = 100

# # Freeze all the layers before the `fine_tune_at` layer
# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable = False

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-5),
#     loss="mean_squared_error",
#     metrics=["mae"],
# )

# model.summary()

# print("Number of trainable weights = {}".format(len(model.trainable_weights)))

# history_fine = model.fit(
#     x=data_train,
#     y=labels_train,
#     epochs=fine_tune_epochs,
#     validation_data=(data_test, labels_test)
# )

# Convert to TensorFlow lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(f"{ROOT_PATH}/train/trained_models/{MODEL_NAME}", "wb") as f:
    f.write(tflite_model)


# QUANTIZE MODEL
def representative_data_gen():
    dataset_list = tf.data.Dataset.list_files(DATASET_PATH_QUANT + "/*")
    for i in range(100):
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=1) # grayscale, for color channels=3
        image = tf.image.resize(image, [image_width, image_height])
        image = tf.cast(image / 255.0, tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]



converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()


with open(f"{ROOT_PATH}/train/trained_models/{MODEL_NAME_QUANT}", "wb") as f:
    f.write(tflite_model)



# VALIDATE MODEL

# batch_images, batch_labels = next(val_generator)

# logits = model(batch_images)
# prediction = np.argmax(logits, axis=1)
# truth = np.argmax(batch_labels, axis=1)

# keras_accuracy = tf.keras.metrics.Accuracy()
# keras_accuracy(prediction, truth)

# print("Raw model accuracy: {:.3%}".format(keras_accuracy.result()))

# def set_input_tensor(interpreter, input):
#     input_details = interpreter.get_input_details()[0]
#     tensor_index = input_details["index"]
#     input_tensor = interpreter.tensor(tensor_index)()[0]
#     input_tensor[:, :] = input

# def classify_image(interpreter, input):
#     set_input_tensor(interpreter, input)
#     interpreter.invoke()
#     output_details = interpreter.get_output_details()[0]
#     output = interpreter.get_tensor(output_details["index"])
#     # Outputs from the TFLite model are uint8, so we dequantize the results:
#     scale, zero_point = output_details["quantization"]
#     output = scale * (output - zero_point)
#     top_1 = np.argmax(output)
#     return top_1

# interpreter = tf.lite.Interpreter(
#     f"{ROOT_PATH}/model/classification_q.tflite"
# )
# interpreter.allocate_tensors()

# # Collect all inference predictions in a list
# batch_prediction = []
# batch_truth = np.argmax(batch_labels, axis=1)

# for i in range(len(batch_images)):
#     prediction = classify_image(interpreter, batch_images[i])
#     batch_prediction.append(prediction)

# # Compare all predictions to the ground truth
# tflite_accuracy = tf.keras.metrics.Accuracy()
# tflite_accuracy(batch_prediction, batch_truth)
# print("Quant TF Lite accuracy: {:.3%}".format(tflite_accuracy.result()))
