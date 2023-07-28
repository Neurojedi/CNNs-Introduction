import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D

class NiNModel(tf.keras.Model):
    def __init__(self, output):
        super(NiNModel, self).__init__()
        self.block1 = self.nin_block(96, 11, 4, padding='valid')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)
        self.block2 = self.nin_block(256, 5, 1)
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)
        self.block3 = self.nin_block(384, 3, 1)
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)
        self.block4 = self.nin_block(512, 3, 1)
        self.pool4 = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(output, activation='softmax')

    def call(self, inputs):
        x = self.block1(inputs)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.pool4(x)
        x = self.dense(x)

        return x

    def nin_block(self, num_filters, kernel_size, strides, padding='same'):
        return tf.keras.Sequential([
        Conv2D(num_filters, kernel_size, strides=strides, padding=padding, activation='relu'),
        Conv2D(num_filters, kernel_size=1, activation='relu'),
        Conv2D(num_filters, kernel_size=1, activation='relu')])
