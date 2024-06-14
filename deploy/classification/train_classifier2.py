# TODO check accuracy of newly trained my_classification.tflite model
# TODO       - make file to run model and print predictions and labels
# TODO       - check if predictions are in correct range. Did I normalize input data? Dont think so but better to check.

import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.get_training_data import get_data, data_files
from sklearn.model_selection import train_test_split
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()


image_width = 324
image_height = 244

FIRST_LAYER_STRIDE = 2
epochs = 200
# fine_tune_epochs = 10

# path of repository (crazyflie_obstacle_avoider)
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Create the base model from the pre-trained MobileNet V2
# base_model = tf.keras.applications.MobileNetV2(
#     input_shape=(
#         96,
#         96,
#         3,
#     ),
#     include_top=False,
#     weights="imagenet",
#     alpha=0.35,
# )
# base_model.trainable = True

# # Add a custom head, which will predict the numerical output!!!!
# model = tf.keras.Sequential(
#     [
#         tf.keras.Input(shape=(image_height, image_width, 1)),
#         tf.keras.layers.SeparableConvolution2D(
#             filters=3,
#             kernel_size=1,
#             # activation="relu",
#             activation=None,
#             strides=FIRST_LAYER_STRIDE,
#         ),
#         tf.keras.layers.experimental.preprocessing.Resizing(
#             96, 96, interpolation="bilinear"
#         ),
#         base_model,
#         tf.keras.layers.SeparableConvolution2D(
#             filters=32, kernel_size=3, activation="relu"
#         ),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dense(units=2, activation="softmax"),
#     ]
# )

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(image_height, image_width, 1)),
        tf.keras.layers.SeparableConvolution2D(
            filters=16,
            kernel_size=3,
            activation='relu',
            strides=FIRST_LAYER_STRIDE,
            padding='same'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.SeparableConvolution2D(
            filters=32,
            kernel_size=3,
            activation='relu',
            padding='same'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.SeparableConvolution2D(
            filters=64,
            kernel_size=3,
            activation='relu',
            padding='same'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3)  # Output layer for regression
    ]
)

# model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="mean_squared_error", metrics=["mae"])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])  # mean_absolute_error

model.summary()
print("Number of trainable weights = {}".format(len(model.trainable_weights)))


# retrieve data
all_data, all_labels = get_data(data_files)
all_data = all_data.reshape(all_data.shape[0], 244, 324, 1)

# Example data shape check for image data (height=64, width=64, channels=3)
expected_data_shape = (len(all_data), 244, 324, 1)  # None for the number of samples
expected_label_shape = (len(all_data), 2)  # Assuming labels are single integers (e.g., for classification)

# Check data shape
print("Data shape:", all_data.shape, "Expected:", expected_data_shape)
print("Label shape:", all_labels.shape, "Expected:", expected_label_shape)

# make train test split
data_train, data_test, labels_train, labels_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)


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

# with open(f"{ROOT_PATH}/deploy/classification/model/my_classification_2outputs.tflite", "wb") as f:
with open(f"{ROOT_PATH}/deploy/classification/model/my_classification_brightness_distribution.tflite", "wb") as f:
    f.write(tflite_model)




# Convert to quantized TensorFlow Lite
def representative_data_gen():
    dataset_list = tf.data.Dataset.list_files(DATASET_PATH + "/*/*/*")
    for i in range(100):
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=1)
        image = tf.image.resize(
            image, [args.image_width, args.image_height]
        )
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open(
    f"{ROOT_PATH}/model/classification_q.tflite", "wb"
) as f:
    f.write(tflite_model)

batch_images, batch_labels = next(val_generator)

logits = model(batch_images)
prediction = np.argmax(logits, axis=1)
truth = np.argmax(batch_labels, axis=1)

keras_accuracy = tf.keras.metrics.Accuracy()
keras_accuracy(prediction, truth)

print("Raw model accuracy: {:.3%}".format(keras_accuracy.result()))

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
    return top_1

interpreter = tf.lite.Interpreter(
    f"{ROOT_PATH}/model/classification_q.tflite"
)
interpreter.allocate_tensors()

# Collect all inference predictions in a list
batch_prediction = []
batch_truth = np.argmax(batch_labels, axis=1)

for i in range(len(batch_images)):
    prediction = classify_image(interpreter, batch_images[i])
    batch_prediction.append(prediction)

# Compare all predictions to the ground truth
tflite_accuracy = tf.keras.metrics.Accuracy()
tflite_accuracy(batch_prediction, batch_truth)
print("Quant TF Lite accuracy: {:.3%}".format(tflite_accuracy.result()))
