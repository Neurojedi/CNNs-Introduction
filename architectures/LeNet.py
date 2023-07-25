import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD

class LeNet5(Model): 
    def __init__(self, num_classes): 
        super(LeNet5, self).__init__()
        self.conv1 = Conv2D(filters=6, kernel_size=(5,5), activation='tanh', padding="same", name="C1")
        self.conv2 = Conv2D(filters=16,kernel_size=(5,5), activation='tanh', padding="valid", name="C3")
        self.avg_pool = AveragePooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.conv3 = Conv2D(filters=120, kernel_size=(5,5), activation='tanh', padding="valid", name="C5")
        self.dense1 = Dense(84, activation='tanh')
        self.dense2 = Dense(num_classes, activation='softmax')
    @tf.function
    def call(self, inputs): 
        stack1 = self.avg_pool(self.conv1(inputs)) 
        stack2 = self.avg_pool(self.conv2(stack1)) 
        flatten = self.flatten(self.conv3(stack2))
        output = self.dense2(self.dense1(flatten)) 
        return output 
