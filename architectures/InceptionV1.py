import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D

class InceptionModule(layers.Layer):
    def __init__(self, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool):
        super(InceptionModule, self).__init__()

        self.conv1x1 = layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')
        self.conv3x3_reduce = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')
        self.conv3x3 = layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')
        self.conv5x5_reduce = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')
        self.conv5x5 = layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')
        self.maxpool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')
        self.maxpool_proj = layers.Conv2D(filters_pool, (1, 1), padding='same', activation='relu')

    def call(self, inputs):
        conv1x1 = self.conv1x1(inputs)
        conv3x3_reduce = self.conv3x3_reduce(inputs)
        conv3x3 = self.conv3x3(conv3x3_reduce)
        conv5x5_reduce = self.conv5x5_reduce(inputs)
        conv5x5 = self.conv5x5(conv5x5_reduce)
        maxpool = self.maxpool(inputs)
        maxpool_proj = self.maxpool_proj(maxpool)

        output = tf.concat([conv1x1, conv3x3, conv5x5, maxpool_proj], axis=-1)
        return output

class InceptionV1(tf.keras.Model):
    def __init__(self, num_classes=1000, training=False):
        super(InceptionV1, self).__init__()
        self.training=training
        self.num_classes=num_classes
        self.conv1 = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')
        self.maxpool1 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        self.conv2 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')
        self.conv3 = layers.Conv2D(192, (3, 3), padding='same', activation='relu')
        self.maxpool2 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')

        self.inception3a = InceptionModule(64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(128, 128, 192, 32, 96, 64)

        self.maxpool3 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')

        self.inception4a = InceptionModule(192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(256, 160, 320, 32, 128, 128)

        self.maxpool4 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')

        self.inception5a = InceptionModule(256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(384, 192, 384, 48, 128, 128)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.4)
        self.fc = layers.Dense(num_classes, activation='softmax')

        self.auxiliary1 = self.create_auxiliary_classifier()
        self.auxiliary2 = self.create_auxiliary_classifier()

    def create_auxiliary_classifier(self):
      auxiliary_classifier = tf.keras.Sequential([
          layers.AveragePooling2D((5, 5), strides=(3, 3), padding='valid'),
          layers.Conv2D(128, (1, 1), padding='same', activation='relu'),
          layers.Flatten(),
          layers.Dense(1024, activation='relu'),
          layers.Dropout(0.7),
          layers.Dense(self.num_classes, activation='softmax')
        ])

      return auxiliary_classifier

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        # Auxiliary Classifier 1
        if self.training:
          aux1 = self.auxiliary1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        # Auxiliary Classifier 2
        if self.training:
          aux2 = self.auxiliary2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training:
          return x, aux1, aux2
        else:
          return x
