# work in progress

import tensorflow as tf
import numpy as np
import cv2
import os
import onnxruntime as ort
import matplotlib.pyplot as plt


def run_model_h5(model_path, image):
    model = tf.keras.models.load_model(model_path)
    image_array = image.astype('float32') / 255.0
    # Resize the image
    input_shape = model.input_shape[1:]
    image_array = cv2.resize(image_array, (input_shape[1], input_shape[0]))

    # Expand the dimensions
    image_array = np.expand_dims(image_array, axis=0)

    # Make a prediction
    prediction = model.predict(image_array)

    # Print the output
    return prediction


def run_model_tflite(model_path, image):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output = interpreter.get_tensor(output_details[0]['index'])

    # Print the output
    return output



# def run_onnx_model(model_path, image):
#     """ Run the ONNX model on the preprocessed image data """
#     # Create an ONNX runtime session
#     sess = ort.InferenceSession(model_path)

#     # Get the name of the input node
#     input_name = sess.get_inputs()[0].name

#     # Run the model
#     output = sess.run(None, {input_name: image})

#     return output

def run_onnx_model(model_path, image):
    # Load the ONNX model
    model = tf.keras.models.load_model(model_path)

    # Resize the image to match the input shape of the model
    # input_shape = model.input_shape[1:]
    # image_array = cv2.resize(image_array, (input_shape[1], input_shape[0]))

    # Expand the dimensions of the image array
    image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))

    # Make a prediction using the model
    prediction = model.predict(image_array)

    # Print the prediction
    print(prediction)



if __name__ == '__main__':
    # Specify the path to the image and the model
    model_path = "models/trained_models/model.h5"
    model = tf.keras.models.load_model(model_path)

    model.summary()
    
    image_dir = "data/datasets/cyberzoo_set1"

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)

            # plt.imshow(image)
            # plt.show()

            # Run the model
            output = run_model_h5(model_path, image)
            print(output)




