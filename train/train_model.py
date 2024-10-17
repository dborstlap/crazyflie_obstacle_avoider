import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
from tensorflow.keras.callbacks import TensorBoard

sys.path.append(os.getcwd())
from train.utils.utils import save_training_history_plot
# from train.models.architectures.example_model_complex import my_example_model
from models.architectures.example_model_simple import example_model_simple
from data.dataset import Dataset


def trainNetwork(model_to_train, model_name, image_dir, labels_file, input_shape, output_shape, batch_size, epochs, epochs_quant, save_model, device, aware_quantization):

    # logs
    log_dir = 'train/logs/models/'+model_name+'_'+ str(input_shape[1])+'x'+str(input_shape[0])
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # initialize the dataset
    dataset = Dataset(image_dir, labels_file, input_shape, output_shape)

    # Generate the datasets
    train, val = dataset.createDataset(batch_size=batch_size)
    datasetTrain = train
    datasetVal = val
    print('Datasets Ready')

    strategy = tf.distribute.OneDeviceStrategy(device=device)
    with strategy.scope():
        # Create the model
        model = model_to_train(input_shape=input_shape, output_shape=output_shape)
        model.summary()
        # Train the model
        history = model.fit(datasetTrain, epochs=epochs, validation_data=datasetVal, verbose = 1, callbacks = [tensorboard_callback])
    
    if save_model:
        model.save('train/models/trained_models/'+model_name+'_'+ str(input_shape[1])+'x'+str(input_shape[0])+'.keras')
        # Convert the TensorFlow model to a TensorFlow Lite model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        # Save the TensorFlow Lite model to a file
        with open('train/models/trained_models/'+model_name+'_'+ str(input_shape[1])+'x'+str(input_shape[0])+'_base.tflite', 'wb') as f:
            f.write(tflite_model)

    # save a plot showing training loss history, to see if it converged properly
    save_training_history_plot(history, model_name, input_shape)

    with strategy.scope():
        if aware_quantization:
            model = tfmot.quantization.keras.quantize_model(model)
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
            history = model.fit(datasetTrain, epochs=epochs_quant, validation_data=datasetVal, verbose = 1)
            quantization_accuracy = history.history['val_loss'][-1]
            model.summary()

        if save_model:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            # Save the TensorFlow Lite model to a file
            with open('train/models/trained_models/'+model_name+'_'+ str(input_shape[1])+'x'+str(input_shape[0])+'_quant.tflite', 'wb') as f:
                f.write(tflite_model)

    # Visualize Predictions
    predictions = model.predict(datasetVal)
    imagesVal, labelsVal = next(iter(datasetVal))
    return predictions, imagesVal, labelsVal


if __name__ == '__main__':

    # Train the model
    predictions, imagesVal, labelsVal = trainNetwork(
        model_to_train = example_model_simple,
        model_name = 'examplebrightness_net', 
        image_dir = 'data/training_data/images', 
        labels_file = 'data/training_data/labels/example_labels.csv', 
        input_shape = (120,180,1), 
        output_shape = 3, 
        batch_size = 100, 
        epochs = 100, 
        epochs_quant = 50, 
        save_model = True, 
        device = 'GPU:0', 
        aware_quantization = True)

    print('done')

