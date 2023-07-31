import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class VGG(tf.keras.Model):
    def __init__(self, version='VGG16', num_classes=1000):
        super(VGG, self).__init__()

        self.version=version
        # Block 1
        self.conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool1 = MaxPooling2D((2, 2), strides=(2, 2))

        # Block 2
        self.conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool2 = MaxPooling2D((2, 2), strides=(2, 2))

        # Block 3
        self.conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same')
        if version == 'VGG19':
            self.conv3_4 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.pool3 = MaxPooling2D((2, 2), strides=(2, 2))

        # Block 4
        self.conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same')
        if version == 'VGG19':
            self.conv4_4 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.pool4 = MaxPooling2D((2, 2), strides=(2, 2))

        # Block 5
        self.conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same')
        if version == 'VGG19':
            self.conv5_4 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.pool5 = MaxPooling2D((2, 2), strides=(2, 2))

        # Classification layers
        self.flatten = Flatten()
        self.fc1 = Dense(4096, activation='relu')
        self.fc2 = Dense(4096, activation='relu')
        self.fc3 = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        if self.version == 'VGG19':  # VGG19
            x = self.conv3_4(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        if self.version == 'VGG19':  # VGG19
            x = self.conv4_4(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        if self.version == 'VGG19':  # VGG19
            x = self.conv5_4(x)
        x = self.pool5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x