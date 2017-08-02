from __future__ import absolute_import

from keras import backend as K

from model.enet import *


def eval(in_shape=(256,512,3), class_num=5):
    model = ENet(in_shape, class_num)
    model.model.summary()

    pass
