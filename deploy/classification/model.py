import tensorflow as tf
import os

import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()


FIRST_LAYER_STRIDE = 2

image_width = 324
image_height = 244


base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96,96,3),
        include_top=False,
        weights="imagenet",
        alpha=0.35,
    )
base_model.trainable = False


model_orig = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(image_height, image_width, 1)),
            tf.keras.layers.SeparableConvolution2D(
                filters=3,
                kernel_size=1,
                # activation="relu",
                activation=None,
                strides=FIRST_LAYER_STRIDE,
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
            tf.keras.layers.Dense(units=3, activation="softmax"),
        ]
    )




model1 = tf.keras.Sequential(
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





model2 = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(image_height, image_width, 1)),
        tf.keras.layers.SeparableConvolution2D(
            filters=16,
            kernel_size=3,
            activation='relu',
            strides=FIRST_LAYER_STRIDE,
            padding='same'
        ),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3)  # Output layer for regression
    ]
)


model3 = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(244, 324, 1)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(3)
])