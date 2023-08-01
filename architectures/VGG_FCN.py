import tensorflow as tf
from tensorflow.keras import layers

class VGG_FCN(tf.keras.Model):
    def __init__(self, num_classes):
        super(VGG_FCN, self).__init__()

        # Convolutional layers
        self.conv1_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')
        self.conv1_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')

        self.conv2_1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')
        self.conv2_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')

        self.conv3_1 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')
        self.conv3_2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')
        self.conv3_3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')

        self.conv4_1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')
        self.conv4_2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')
        self.conv4_3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')
        self.pool4 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')

        self.conv5_1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')
        self.conv5_2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')
        self.conv5_3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5')

        # Fully convolutional layers
        self.fc6 = tf.keras.layers.Conv2D(4096, (7, 7), activation='relu', padding='valid', name='fc6')
        self.dropout1 = tf.keras.layers.Dropout(0.5)

        self.fc7 = tf.keras.layers.Conv2D(4096, (1, 1), activation='relu', padding='valid', name='fc7')
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.score_fr = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax', padding='same', name='score_fr')

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
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        x = self.fc6(x)
        x = self.dropout1(x)
        x = self.fc7(x)
        x = self.dropout2(x)
        x = self.score_fr(x)
        return x
