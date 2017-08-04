from __future__ import absolute_import

from skimage import io
from model.enet import *
import numpy as np
from skimage import measure
from skimage import color

def predict(img_path, weight_path, in_shape=(256, 512, 3), class_num=5, is_cityscapes=False):
    model = ENet(in_shape, class_num)
    model.model.load_weights(weight_path)
    # model.model.summary()
    img = io.imread(img_path)
    if is_cityscapes:
        img = measure.block_reduce(img, (4,4,1))
    else:
        img = measure.block_reduce(img, (2,2,1))
    img = np.array([img])

    res = model.model.predict(img, 1, 1)
    return res, img


def vis(flatten, w, h):
    # for vis the difference
    multi = 40.
    img = np.argmax(flatten, axis=-1)
    img = np.reshape(img, (w, h)) * multi
    img = np.array(img, dtype=np.int)
    return img


if __name__ == '__main__':
    #img_path = '../data/cityscapes/img/train/darmstadt/darmstadt_000000_000019_leftImg8bit.png'
    img_path = '../data/self_labeled/img/train/0000200.jpg'
    weight_path = '../checkpoint/after_train_best.h5'
    res, img = predict(img_path, weight_path)
    res = vis(res, 256, 512)
    res = color.gray2rgb(res)
    # res = res.reshape((res.shape[0],res.shape[1],1))
    # res = np.concatenate((res,res,res),axis=-1)
    # res = res * 0.5 + img[0] * 0.5
    # `res = np.array(res, dtype=np.uint8)
    io.imsave('res.jpg', res)