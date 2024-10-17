import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def example_model_simple(input_shape = (180,120,3),output_shape=3):

  input_tensor = tf.keras.layers.Input(shape=input_shape, name='input')
  conv2d = layers.Conv2D(16, 3, padding='same', activation='relu')(input_tensor)
  maxpool = layers.MaxPooling2D()(conv2d)
  conv2d_1 = layers.Conv2D(32, 3, padding='same', activation='relu')(maxpool)
  maxpool_1 = layers.MaxPooling2D()(conv2d_1)
  conv2d_2 = layers.Conv2D(64, 3, padding='same', activation='relu')(maxpool_1)
  maxpool_2 = layers.MaxPooling2D()(conv2d_2)
  flatten = layers.Flatten()(maxpool_2)
  dense = tf.keras.layers.Dense(output_shape, activation='linear')(flatten)

  output = tf.keras.layers.Reshape((output_shape,))(dense)
  model = tf.keras.models.Model([input_tensor], output, name='model')

  # Compile the model with loss function, optimizer, and metrics
  model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

  return model
