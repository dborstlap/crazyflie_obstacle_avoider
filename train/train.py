import sys
import cv2
import csv
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
import json
import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
from models.architectures.model import createModelGateNet
from src.dataset.dataset import Dataset



def trainNetwork(model_name, image_dir, csv_name, input_shape, output_shape, batch_size, epochs, epochs_optimization, save_model, device, aware_quantization):
    # define model
    createModel = createModelGateNet

    log_dir = 'logs/models/'+model_name+'_'+ str(input_shape[1])+'x'+str(input_shape[0])
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    #Define csv file
    csv_file = os.path.join('data/labels', csv_name)

    #initialize the dataset
    data_dir = os.path.join('data/images', image_dir)
    dataset = Dataset(data_dir, csv_file, input_shape, output_shape)

    #Generate the datasets
    train, val = dataset.createDataset(batch_size=batch_size)
    datasetTrain = train
    datasetVal = val
    
    print('Datasets Ready')

    strategy = tf.distribute.OneDeviceStrategy(device=device)
    with strategy.scope():
        # Create the model
        model = createModel(input_shape=input_shape, output_shape=output_shape)
        model.summary()
        # Train the model
        history = model.fit(datasetTrain, epochs=epochs, validation_data=datasetVal, verbose = 1, callbacks = [tensorboard_callback] )
        

    if save_model:
        model.save('models/trained_models/'+model_name+'_'+ str(input_shape[1])+'x'+str(input_shape[0])+'.keras')
        # Convert the TensorFlow model to a TensorFlow Lite model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        # Save the TensorFlow Lite model to a file
        with open('models/trained_models/'+model_name+'_'+ str(input_shape[1])+'x'+str(input_shape[0])+'_base.tflite', 'wb') as f:
            f.write(tflite_model)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('logs/Loss'+ model_name+'_'+ str(input_shape[1])+'x'+str(input_shape[0])+ '.png', format='png')
    baseline_accuracy = history.history['loss'][-1]

    with strategy.scope():
        if aware_quantization:
            model = tfmot.quantization.keras.quantize_model(model)
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
            history = model.fit(datasetTrain, epochs=epochs_optimization, validation_data=datasetVal, verbose = 1)
            quantization_accuracy = history.history['val_loss'][-1]
            model.summary()

        if save_model:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            # Save the TensorFlow Lite model to a file
            with open('models/trained_models/'+model_name+'_'+ str(input_shape[1])+'x'+str(input_shape[0])+'_quant.tflite', 'wb') as f:
                f.write(tflite_model)

    # Visualize Predictions
    predictions = model.predict(datasetVal)
    imagesVal, labelsVal = next(iter(datasetVal))
    return predictions, imagesVal, labelsVal, baseline_accuracy



model_name = 'brightness_net'
image_dir = 'all_data'
csv_name = 'output_labels.csv'
input_shape = (244, 324, 1)
output_shape = 3  ### is 12 in case of dronenet????
batch_size = 20
epochs = 100
epochs_optimization = 100
save_model = True
device = 'GPU:0'
aware_quantization = True

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
accuracies = []

predictions, imagesVal, labelsVal, _ = trainNetwork(model_name, image_dir, csv_name, input_shape, output_shape, batch_size, epochs, epochs_optimization, save_model, device, aware_quantization)



print('done')

