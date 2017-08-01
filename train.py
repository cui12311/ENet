import numpy as np
import glob
import os
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint


class Generator(object):
    def __init__(self, dataset, shape, val=False):
        self.dataset = dataset
        self.shape = shape
        self.val = val
        self.build(val)

    def build(self, val):
        pass
