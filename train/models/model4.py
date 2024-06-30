import tensorflow as tf

"""
Notes:
performance is good for numerical prediction.
Takes a bit longer to train, bit too many parameters
"""

image_width = 324
image_height = 244



model4 = tf.keras.Sequential([
  tf.keras.Input(shape=(image_width, image_height, 1)),

  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(3)
])