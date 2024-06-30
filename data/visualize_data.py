import cv2

from get_training_data import get_data, datasets

def visualize_data(images, labels):
    # Iterate over the data
    for image, label in zip(images, labels):

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        cv2.imshow("image", image)

        print(label)

        # Wait for the user to press the enter key
        cv2.waitKey(0)

        # Clear the display
        cv2.destroyAllWindows()



if __name__ == "__main__":
    data, labels = get_data(datasets)

    visualize_data(data, labels)

