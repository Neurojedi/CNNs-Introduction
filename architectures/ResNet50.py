import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(ConvBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters[0], kernel_size=(1, 1), strides=(1, 1), padding="valid")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters[1], kernel_size=self.kernel_size, strides=self.strides, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.conv3 = tf.keras.layers.Conv2D(filters=self.filters[2], kernel_size=(1, 1), strides=(1, 1), padding="valid")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.shortcut_conv = tf.keras.layers.Conv2D(filters=self.filters[2], kernel_size=(1, 1), strides=self.strides, padding="valid")
        self.shortcut_bn = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()

    def call(self, inputs):
        shortcut = inputs
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        shortcut = self.shortcut_conv(shortcut)
        shortcut = self.shortcut_bn(shortcut)
        x = tf.keras.layers.add([x, shortcut])
        x = self.relu3(x)
        return x


class IdentityBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters[0], kernel_size=(1, 1), strides=(1, 1), padding="valid")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters[1], kernel_size=self.kernel_size, strides=(1, 1), padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.conv3 = tf.keras.layers.Conv2D(filters=self.filters[2], kernel_size=(1, 1), strides=(1, 1), padding="valid")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()

    def call(self, inputs):
        shortcut = inputs
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.keras.layers.add([x, shortcut])
        x = self.relu3(x)
        return x


class ResNet50(tf.keras.models.Model):
    def __init__(self, input_shape=(224, 224, 3), num_classes=1000):
        super(ResNet50, self).__init__()
        self.model_input_shape = input_shape
        self.num_classes = num_classes
        self.zero_padding = tf.keras.layers.ZeroPadding2D(padding=(3, 3))
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="valid")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.zero_padding2 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
        self.max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        self.conv_block1 = ConvBlock([64, 64, 256], kernel_size=3, strides=(1, 1))
        self.identity_block1_1 = IdentityBlock([64, 64, 256], kernel_size=3)
        self.identity_block1_2 = IdentityBlock([64, 64, 256], kernel_size=3)
        self.conv_block2 = ConvBlock([128, 128, 512], kernel_size=3, strides=(2, 2))
        self.identity_block2_1 = IdentityBlock([128, 128, 512], kernel_size=3)
        self.identity_block2_2 = IdentityBlock([128, 128, 512], kernel_size=3)
        self.identity_block2_3 = IdentityBlock([128, 128, 512], kernel_size=3)
        self.conv_block3 = ConvBlock([256, 256, 1024], kernel_size=3, strides=(2, 2))
        self.identity_block3_1 = IdentityBlock([256, 256, 1024], kernel_size=3)
        self.identity_block3_2 = IdentityBlock([256, 256, 1024], kernel_size=3)
        self.identity_block3_3 = IdentityBlock([256, 256, 1024], kernel_size=3)
        self.identity_block3_4 = IdentityBlock([256, 256, 1024], kernel_size=3)
        self.identity_block3_5 = IdentityBlock([256, 256, 1024], kernel_size=3)
        self.conv_block4 = ConvBlock([512, 512, 2048], kernel_size=3, strides=(2, 2))
        self.identity_block4_1 = IdentityBlock([512, 512, 2048], kernel_size=3)
        self.identity_block4_2 = IdentityBlock([512, 512, 2048], kernel_size=3)
        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(self.num_classes, activation="softmax")

    def call(self, inputs):
        x = self.zero_padding(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.zero_padding2(x)
        x = self.max_pooling(x)
        x = self.conv_block1(x)
        x = self.identity_block1_1(x)
        x = self.identity_block1_2(x)
        x = self.conv_block2(x)
        x = self.identity_block2_1(x)
        x = self.identity_block2_2(x)
        x = self.identity_block2_3(x)
        x = self.conv_block3(x)
        x = self.identity_block3_1(x)
        x = self.identity_block3_2(x)
        x = self.identity_block3_3(x)
        x = self.identity_block3_4(x)
        x = self.identity_block3_5(x)
        x = self.conv_block4(x)
        x = self.identity_block4_1(x)
        x = self.identity_block4_2(x)
        x = self.global_avg_pooling(x)
        x = self.dense(x)
        return x