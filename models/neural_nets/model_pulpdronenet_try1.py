import tensorflow as tf
from tensorflow.keras import layers, Model


"""
1. tensorflow recreation of pulp-dronenet
"""

class ResBlock(layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters=out_channels, kernel_size=3, strides=2, padding='same')
        self.conv2 = layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same')
        self.bypass = layers.Conv2D(filters=out_channels, kernel_size=1, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU(max_value=6)

    def call(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x_bypass = self.bypass(identity)
        x = layers.add([x, self.relu(x_bypass)])
        return x

def create_dronet(input_shape=(244, 324, 1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 5, strides=2, padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # x = ResBlock(32, 32)(x)
    x =   layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')
    x =   layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
    x =   layers.Conv2D(filters=32, kernel_size=1, strides=2, padding='same')
    x =   layers.BatchNormalization()
    x =   layers.BatchNormalization()
    x =   layers.ReLU(max_value=6)
    
    # x = ResBlock(32, 64)(x)
    x =   layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')
    x =   layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
    x =   layers.Conv2D(filters=64, kernel_size=1, strides=2, padding='same')
    x =   layers.BatchNormalization()
    x =   layers.BatchNormalization()
    x =   layers.ReLU(max_value=6)

    # x = ResBlock(64, 128)(x)
    x =   layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')
    x =   layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
    x =   layers.Conv2D(filters=128, kernel_size=1, strides=2, padding='same')
    x =   layers.BatchNormalization()
    x =   layers.BatchNormalization()
    x =   layers.ReLU(max_value=6)

    x = layers.Dropout(0.5)(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2)(x)

    # Separate steering and collision outputs
    steer = x[:, 0]
    coll = layers.Activation('sigmoid')(x[:, 1])

    model = Model(inputs=inputs, outputs=steer) #outputs=[steer, coll])
    return model

# Create the model
dronet_model = create_dronet()