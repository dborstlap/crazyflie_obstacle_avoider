import tensorflow as tf
import numpy as np
import pickle
import os
import sys
import tf2onnx
import tensorflow_model_optimization as tfmot

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural_nets.model1 import model1
from neural_nets.model2 import model2
from data.get_training_data import get_data, data_files


# --------- INPUTS --------------
# which model to use
model = model1

# quantize the model or not
quantize = True

# name to save model to
model_save_name = 'model1_quantized.tflite'

# Load the training data
data_train, labels_train, data_test, labels_test = get_data(data_files)

# training parameters
epochs = 10
batch_size = 50

quant_epochs = 2
quant_batch_size = 50

""""""""""""""""""""""""""""""""""""""
# TRAIN

# Compile the model if not already compiled
model.compile(optimizer='adam', loss='mean_squared_error') #, metrics=['accuracy'])

# Fit the model on the training data
history = model.fit(data_train, labels_train, epochs=epochs, validation_data=(data_test, labels_test)) # batch_size=batch_size, 


""""""""""""""""""""""""""""""""""""""
# QUANTIZE

if quantize:
    # Apply quantization to the layers of the model
    model = tfmot.quantization.keras.quantize_model(model)

    # Compile the quantization-aware model
    model.compile(optimizer='adam', loss='mean_squared_error') #metrics=['accuracy'])

    # Train the quantization-aware model
    model.fit(data_train, labels_train, epochs=quant_epochs, validation_data=(data_test, labels_test)) # batch_size=quant_batch_size, 

""""""""""""""""""""""""""""""""""""""
# SAVE

# Get the directory where the file is located
script_dir = os.path.abspath(os.path.dirname(__file__))
save_dir = os.path.join(script_dir, '..', 'models/trained_models/')

# Convert to TensorFlow Lite model
tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()

# save model
with open(f'{save_dir}' + model_save_name, 'wb') as f:
    f.write(tflite_model)





