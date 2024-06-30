import cv2
import numpy as np
from get_training_data import get_data, datasets



data, labels = get_data(datasets)
images = np.transpose(data, (0, 2, 1, 3))

# Iterate over the data
for image, label in zip(images, labels):

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imshow("image", image)
    print(label)

    # Wait for the user to press the enter key
    cv2.waitKey(0)
    cv2.destroyAllWindows()





