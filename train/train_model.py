# TODO go over https://github.com/google-coral/tutorials/blob/52b60653698a10e7c83c5761cf6a2acc3db57d22/retrain_classification_ptq_tf2.ipynb and improve code based on it
# TODO Use gatenet to improve this file/model
# TODO add batch size variable, make train_generator
# TODO make it cutoff training at certain point.


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
from data.get_data import get_data
from sklearn.model_selection import train_test_split
from models.model1 import model1
from models.model5 import model5
from models.model_classification import model
from models.model_brightness_determinator import model_brightness
import tensorflow_model_optimization as tfmot


NAME = "classification_brightness_model5"
MODEL_NAME = NAME + ".tflite"
MODEL_NAME_QUANT = NAME + "_q.tflite"

# path of repository (crazyflie_obstacle_avoider)
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH_QUANT = 'data/images/cyberzoo_set1'

image_width = 324
image_height = 244
number_of_labels = 2
batch_size = 50

epochs = 10

# define model
model = model5

# model = tfmot.quantization.keras.quantize_model(model)

# retrieve training data
data, labels = get_data(datasets=['dataset_2.h5'])

# Assert data shape
expected_data_shape = (image_width, image_height, 1)
assert data.shape[1:] == expected_data_shape, "Data shape does not match expected shape"
assert labels.shape[1:][0] == number_of_labels, "Label shape does not match expected shape"


# make train and test split
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-6), loss='categorical_crossentropy', metrics=["accuracy"])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

model.summary()
print("Number of trainable weights = {}".format(len(model.trainable_weights)))


# Train the custom head
# history = model.fit(
#     x=data_train,
#     y=labels_train,
#     epochs=epochs,
#     validation_data=(data_test, labels_test)
# )

ROOT_PATH = os.path.abspath(os.curdir)
DATASET_PATH = f"{ROOT_PATH}/data/images/class_brightness"

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

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


# """own generator for numerical value prediction"""
# def custom_data_generator(image_paths, labels):
#     for img_path, label in zip(image_paths, labels):
#         image = load_image(img_path)  # Your custom image loading logic
#         yield image, label

# # Create a tf.data.Dataset from the custom generator
# dataset = tf.data.Dataset.from_generator(
#     lambda: custom_data_generator(image_paths, labels),
#     output_signature=(
#         tf.TensorSpec(shape=(image_height, image_width, 3), dtype=tf.float32),
#         tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
#     )
# )


history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=len(val_generator),
)

# model.fit(datasetTrain, validation_data=datasetVal, epochs = EPOCHS, callbacks = [tensorboard_callback])



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
    print(f"Model saved as {MODEL_NAME}")

# QUANTIZE MODEL
def representative_data_gen():
    dataset_list = tf.data.Dataset.list_files(DATASET_PATH_QUANT + "/*")
    for i in range(100):
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=1) # grayscale, for color channels=3
        image = tf.image.resize(image, [image_width, image_height])
        image = tf.cast(image, tf.float32)  # / 255.0 
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
    print(f"Model saved as {MODEL_NAME_QUANT}")



