import argparse
import os

import numpy as np
import tensorflow as tf
import PIL.Image
import scipy


image_width = 324
image_height = 244

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(
        96,
        96,
        3,
    ),
    include_top=False,
    weights="imagenet",
    alpha=0.35,
)
base_model.trainable = False

# Add a custom head, which will predict the classes
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(image_width, image_height, 1)),
        tf.keras.layers.SeparableConvolution2D(
            filters=3,
            kernel_size=1,
            # activation="relu",
            activation=None,
            strides=2,
        ),
        tf.keras.layers.experimental.preprocessing.Resizing(
            96, 96, interpolation="bilinear"
        ),
        base_model,
        tf.keras.layers.SeparableConvolution2D(
            filters=32, kernel_size=3, activation="relu"
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=2, activation="softmax"),
    ]
)

