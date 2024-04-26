
import tensorflow as tf
import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt




def run_model(model_path, image):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    image = image/255.

    print(image.mean())
    # plt.imshow(image_in)
    # plt.show()

    input_details = interpreter.get_input_details()

    #Resize
    image_in = cv2.resize(image,(input_details[0]['shape'][2],input_details[0]['shape'][1]))
    image_in = np.expand_dims(image_in, axis=0)
    image_in = np.expand_dims(image_in, axis=3)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image_in.astype('float32'))

    # Perform inference
    interpreter.invoke()

    # Get the output tensor and post-process if needed
    output_details = interpreter.get_output_details()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(output)

    cv2.imshow('out',image)
    cv2.waitKey(0)
      


if __name__ == '__main__':
    # define model and image folders
    model_dir = "models/trained_models/model_brightness_q_small2.tflite"
    image_dir = "data/datasets/cyberzoo_set1"

    


    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)

            # Read the image from the folder
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            run_model(model_dir, image)







