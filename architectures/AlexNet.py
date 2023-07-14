import tensorflow as tf

class AlexNet(tf.keras.Model):
    def __init__(self, num_classes, batch_norm=False):
        super(AlexNet, self).__init__()
        self.batch_norm = batch_norm
        self.conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv3 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv4 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu')
        self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(4096, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(4096, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax')
        if self.batch_norm:
            self.batch_norm1 = tf.keras.layers.BatchNormalization()
            self.batch_norm2 = tf.keras.layers.BatchNormalization()
            self.batch_norm3 = tf.keras.layers.BatchNormalization()
            self.batch_norm4 = tf.keras.layers.BatchNormalization()
            self.batch_norm5 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.batch_norm2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        if self.batch_norm:
            x = self.batch_norm3(x)
        x = self.maxpool2(x)
        x = self.conv4(x)
        if self.batch_norm:
            x = self.batch_norm4(x)
        x = self.conv5(x)
        if self.batch_norm:
            x = self.batch_norm5(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        outputs = self.dense3(x)
        return outputs

class AlexNetLRN(tf.keras.Model):
    def __init__(self, num_classes):
        super(AlexNetLRN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv3 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv4 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu')
        self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(4096, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(4096, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75)  # Apply LRN
        x = self.conv2(x)
        x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75)  # Apply LRN
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        outputs = self.dense3(x)
        return outputs