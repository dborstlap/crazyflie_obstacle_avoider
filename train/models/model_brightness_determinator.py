from tensorflow.keras import layers, Sequential

model_brightness = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(244, 324, 1)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(2)
])




