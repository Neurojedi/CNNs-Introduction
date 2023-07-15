import tensorflow as tf
import math
import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler


class WarmUpCallbackEpoch(tf.keras.callbacks.Callback):
    def __init__(self, num_epochs, warmup_epochs):
        super().__init__()
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
    
    def lr_schedule(self, epoch, lr):
        if epoch < self.warmup_epochs:
            lr = lr * (epoch + 1) / self.warmup_epochs
        else:
            lr = 0.5 * lr * (1 + math.cos((epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs) * math.pi))
        return lr

    def warm_up_with_cosine_epoch(self):
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: self.lr_schedule(epoch, lr))
        return lr_scheduler

class WarmUpCallbackBatch(tf.keras.callbacks.Callback):
    def __init__(self, warmup_steps, initial_lr, num_epochs):
        super(WarmUpCallbackBatch, self).__init__()
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.num_epochs = num_epochs
        self.global_step = 0
        self.lrs = []  
        
    def on_train_begin(self, logs=None):         
        self.steps_per_epoch = self.params['steps']
        self.total_steps = self.steps_per_epoch * self.num_epochs # steps_per_epoch * num_epochs = total_steps
        
    def on_train_batch_begin(self, batch, logs=None):
        self.global_step += 1
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        if self.global_step <= self.warmup_steps:
            warmup_lr = self.initial_lr * (self.global_step / self.warmup_steps)
            tf.keras.backend.set_value(self.model.optimizer.lr, warmup_lr)
            self.lrs.append(warmup_lr)
        else:
            epoch = (self.global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps) * 50
            cosine_lr = 0.5 * self.initial_lr * (1 + tf.math.cos(epoch / 50 * np.pi))
            tf.keras.backend.set_value(self.model.optimizer.lr, cosine_lr)
            self.lrs.append(cosine_lr)


class WarmUpCosineDecayRestarts(tf.keras.callbacks.Callback):
    def __init__(self, num_epochs, warmup_steps=0, lr_init=0.001, lr_min=0.0,
                 lr_max=0.1, cycle_length=100, m_mul=1.0):
        super(WarmUpCosineDecayRestarts, self).__init__()
        self.num_epochs=num_epochs
        self.warmup_steps = warmup_steps
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.cycle_length = cycle_length
        self.m_mul = m_mul
        self.steps = 0
        self.lr = 0
        self.restart = False
        self.lrs=[]
        #warmup_steps: the number of steps for the warm-up phase
        #lr_init: the initial learning rate
        #lr_min: the minimum learning rate
        #lr_max: the maximum learning rate
        #cycle_length: the length of a cycle in steps
        #m_mul: the multiplier for the maximum learning rate and cycle length after a restart
    def on_train_begin(self, logs=None):         
        self.steps_per_epoch = self.params['steps']
        self.global_steps = self.steps_per_epoch * self.num_epochs # steps_per_epoch * num_epochs = total_steps
        
    def on_train_batch_begin(self, batch, logs=None):
        self.steps += 1
        if self.steps <= self.warmup_steps:
            self.lr = self.lr_init * (self.steps / self.warmup_steps)
            self.lrs.append(self.lr)
        else:
            if self.steps == self.warmup_steps + 1:
                self.restart = True
            if self.steps == self.global_steps + 1:
                self.steps = 1
                self.restart = True
            if self.restart:
                self.lr_max *= self.m_mul
                self.cycle_length = int(self.cycle_length * self.m_mul)
                self.lr = self.lr_min
                self.restart = False
            else:
                lr_range = self.lr_max - self.lr_min
                position = (self.steps - self.warmup_steps) / (self.global_steps - self.warmup_steps)
                cycle_position = position % (self.cycle_length / (self.global_steps - self.warmup_steps))
                self.lr = self.lr_min + 0.5 * lr_range * (1 + np.cos(np.pi * cycle_position / (self.cycle_length / (self.global_steps - self.warmup_steps))))
                self.lrs.append(self.lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)