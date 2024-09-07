import tensorflow as tf
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.get_data import get_data, datasets
from run_model import run_inference_batch

data, labels = get_data(datasets)

script_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(script_dir, 'trained_models/255_input.tflite')
model_path_q = os.path.join(script_dir, 'trained_models/255_input_q.tflite')

predictions = run_inference_batch(model_path, data)
predictions_q = run_inference_batch(model_path_q, data, quant=True)


keras_mean_absolute_error = tf.keras.metrics.MeanAbsoluteError()
keras_mean_absolute_error(predictions, labels)

keras_mean_squared_error = tf.keras.metrics.MeanSquaredError()
keras_mean_squared_error(predictions, labels)

keras_accuracy = tf.keras.metrics.Accuracy()
keras_accuracy(predictions, labels)


print("Raw model MAE: {:.3%}".format(keras_mean_absolute_error.result()))
print("Raw model MSE: {:.3%}".format(keras_mean_squared_error.result()))
print("Raw model accuracy: {:.3%}".format(keras_accuracy.result()))








