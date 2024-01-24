
import tensorflow as tf
import cv2
import os 
import numpy as np




model_dir = "models/trained_models/model1_quantized.tflite"
image_dir = "data/datasets/cyberzoo_set2"

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=model_dir)
interpreter.allocate_tensors()



for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)

        # Read the image from the folder
        image_in = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        input_details = interpreter.get_input_details()
        #Resize
        image_in = cv2.resize(image_in,(input_details[0]['shape'][2],input_details[0]['shape'][1]))
        image = np.expand_dims(image_in, axis=0)
        image = np.expand_dims(image, axis=3)
        # Set the input tensor

        # Perform inference
        interpreter.invoke()

        # Get the output tensor and post-process if needed
        output_details = interpreter.get_output_details()
        output = interpreter.get_tensor(output_details[0]['index'])

        print(output)

        # corners = [(output[0,0], output[0,1]),(output[0,2], output[0,3]), (output[0,4], output[0,5]), (output[0,6], output[0,7]) ]
        # for corner in corners:
        #     x, y = corner
        #     cv2.circle(image_in, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.imshow('out',image_in)
        cv2.waitKey(0)
        
        print(output)
      



