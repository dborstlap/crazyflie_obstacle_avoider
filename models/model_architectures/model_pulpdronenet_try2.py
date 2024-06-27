import tensorflow as tf
from tensorflow.keras import models, layers

class DroneNet(models.Model):
    def __init__(self):
        # super(DroneNet, self).__init__()
        # Conv 5x5, 1, 32, 200x200, /2
        self.conv1 = tf.keras.layers.Conv2D(32, 5, strides=2, padding='same', activation='relu')
        # Max pooling 2x2, 32, 32, 100x100, /2
        self.pool = tf.keras.layers.MaxPooling2D(2, strides=2)
        # ResBlock1
        self.resBlock1_conv1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='relu')
        self.resBlock1_bn = tf.keras.layers.BatchNormalization()
        self.resBlock1_conv2 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='relu')
        # ResBlock2
        self.resBlock2_conv1 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')
        self.resBlock2_bn = tf.keras.layers.BatchNormalization()
        self.resBlock2_conv2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')
        # ResBlock3
        self.resBlock3_conv1 = tf.keras.layers.Conv2D(128, 3, strides=1, padding='same', activation='relu')
        self.resBlock3_bn = tf.keras.layers.BatchNormalization()
        self.resBlock3_conv2 = tf.keras.layers.Conv2D(128, 3, strides=1, padding='same', activation='relu')
        # Fully connected layer
        self.fc = tf.keras.layers.Dense(2)
        self.sig = tf.keras.layers.Sigmoid()

    def call(self, input_data):
        x = self.conv1(input_data)
        x = self.pool(x)

        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)

        x = tf.layers.flatten(x)
        x = self.fc(x)

        steer = x[:, 0]
        coll = self.sig(x[:, 1])

        return [steer, coll]
