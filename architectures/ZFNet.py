import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class ZFNet(Model):
    def __init__(self, num_classes):
        super(ZFNet, self).__init__()

        # Layer 1 (Convolutional + ReLU + Max Pooling)
        self.conv1 = Conv2D(96, (7, 7), strides=(2, 2), padding='same', activation='relu')
        self.maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        # Layer 2 (Convolutional + ReLU + Max Pooling)
        self.conv2 = Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='relu')
        self.maxpool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        # Layer 3 (Convolutional + ReLU)
        self.conv3 = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')

        # Layer 4 (Convolutional + ReLU)
        self.conv4 = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')

        # Layer 5 (Convolutional + ReLU + Max Pooling)
        self.conv5 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        # Fully Connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(4096, activation='relu')
        self.fc2 = Dense(4096, activation='relu')
        self.fc3 = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x