import tensorflow as tf

"""
Notes:
performance is good for numerical prediction.
Takes a bit longer to train, bit too many parameters
"""

image_width = 324
image_height = 244



model5 = tf.keras.Sequential([
  tf.keras.Input(shape=(image_width, image_height, 1)),
  tf.keras.layers.AveragePooling2D(10,10),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(3)
])





