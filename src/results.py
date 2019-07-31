import tensorflow.keras
import numpy as np
import sklearn.metrics
import threading
import wandb

class Results(tensorflow.keras.callbacks.Callback):

    def __init__(self):
        self.max_val = -1
        self.max_acc = -1
        
    def on_epoch_end(self, epoch, logs={}):

