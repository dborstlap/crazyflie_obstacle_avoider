import tensorflow as tf
import numpy as np
import os
import sys
import tf2onnx
import tensorflow_model_optimization as tfmot

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.get_data import get_data, data_files
# from neural_nets.model1 import model1
# from neural_nets.model2 import model2
# from neural_nets.model_pulpdronenet_try1 import dronet_model
# from neural_nets.model_pulpdronenet_try2 import DroneNet
from neural_nets.model_brightness_determinator import model_brightness



# --------- INPUTS --------------
# which model to use
model = model_brightness

# quantize the model or not
train = True
quantize = True

# name to save model to
model_save_name = 'model_brightness_quantized_smaller2'

# Load the training data
data_train, labels_train_all, data_test, labels_test_all = get_data(data_files)

labels_train = np.array([label["brightness"] for label in labels_train_all])
labels_test = np.array([label["brightness"] for label in labels_test_all])

# training parameters
epochs = 5
batch_size = 50

quant_epochs = 5
quant_batch_size = 50

""""""""""""""""""""""""""""""""""""""
# TRAIN

# Compile the model if not already compiled
model.compile(optimizer='adam', loss='mae', metrics=['mae', 'mse']) #look into new loss function

# Fit the model on the training data
model.fit(data_train, labels_train, epochs=epochs) #, validation_data=(data_test, labels_test), batch_size=batch_size)

""""""""""""""""""""""""""""""""""""""
# QUANTIZE

if quantize:
    # Apply quantization to the layers of the model
    model = tfmot.quantization.keras.quantize_model(model)

    # Compile the quantization-aware model
    model.compile(optimizer='adam', loss='mean_squared_error') #metrics=['accuracy'])

    # Train the quantization-aware model
    model.fit(data_train, labels_train, epochs=quant_epochs) #, validation_data=(data_test, labels_test)) # batch_size=quant_batch_size, 

""""""""""""""""""""""""""""""""""""""
# SAVE

# Get the directory where the file is located
script_dir = os.path.abspath(os.path.dirname(__file__))
save_dir = os.path.join(script_dir, '..', 'models/trained_models/')

# sav emodel in tensorflow format
# tf.saved_model.save(model, f'{save_dir}' + model_save_name)

# save in onnx format
# tf2onnx.convert.from_keras(model, opset=13, output_path=f'{save_dir}' + model_save_name + '.onnx')

# save in h5 format
# model.save(save_dir+model_save_name+'.h5')

# save in keras format
# model.save(save_dir+model_save_name+'.keras')

# Convert to TensorFlow Lite model
tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()

# save model
with open(f'{save_dir}' + model_save_name + '.tflite', 'wb') as f:
    f.write(tflite_model)




