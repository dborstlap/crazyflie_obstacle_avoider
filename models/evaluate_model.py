# TODO doesnt work yet as intended

import tensorflow as tf
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural_nets.model1 import model1
from neural_nets.model2 import model2
from data.get_training_data import get_data, data_files



def run_inference(interpreter, test_data):
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    predictions = []
    for test_instance in test_data:
        # Test data must be preprocessed to match training
        test_instance = np.expand_dims(test_instance, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_instance)
        
        interpreter.invoke()
        
        prediction = interpreter.get_tensor(output_index)
        predictions.append(prediction[0])

    return np.array(predictions)



if __name__ == '__main__':

    # name of the trained model
    trained_model_file = 'model1_quantized.tflite'

    # Get the directory where the file is located
    script_dir = os.path.abspath(os.path.dirname(__file__))
    modeldir = os.path.join(script_dir, '..', 'models/trained_models', trained_model_file)

    # Load the trained model
    # model = tf.keras.models.load_model(modeldir)

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=modeldir)

    # retrieve test data
    test_data_file = 'data1_augmented.pickle'
    data_dir = os.path.join(script_dir, '..', 'data/datasets', test_data_file)
    data_train, labels_train, data_test, labels_test = get_data(data_files)

    # test the model
    run_inference(interpreter, data_test)


    # Test the model on the test data
    # model.evaluate(data_test, labels_test, verbose=2)
    
    # test_loss = score[0]
    # test_acc = score[1]
    # print('Test loss:', test_loss)
    # print('Test accuracy:', test_acc)
