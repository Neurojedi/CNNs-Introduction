import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, Input, GlobalAveragePooling2D, Dense

def resnet_block(inputs, filters, strides=1, activation='relu'):
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    shortcut = inputs
    if strides != 1 or filters != shortcut.shape[-1]:
        shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    return x

def ResNet34():
    inputs = Input(shape=(224, 224, 3))

    # Initial convolution layer
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Residual blocks
    x = resnet_block(x, filters=64, strides=1)
    x = resnet_block(x, filters=64, strides=1)
    x = resnet_block(x, filters=64, strides=1)

    x = resnet_block(x, filters=128, strides=2)
    x = resnet_block(x, filters=128, strides=1)
    x = resnet_block(x, filters=128, strides=1)
    x = resnet_block(x, filters=128, strides=1)

    x = resnet_block(x, filters=256, strides=2)
    x = resnet_block(x, filters=256, strides=1)
    x = resnet_block(x, filters=256, strides=1)
    x = resnet_block(x, filters=256, strides=1)
    x = resnet_block(x, filters=256, strides=1)
    x = resnet_block(x, filters=256, strides=1)

    x = resnet_block(x, filters=512, strides=2)
    x = resnet_block(x, filters=512, strides=1)
    x = resnet_block(x, filters=512, strides=1)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(units=1000, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model
