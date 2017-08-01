from __future__ import absolute_import
from skimage import io
import glob
import os
import random
import utils.label_transform


class Dataset(object):
    def __init__(self, data_dir, label_dir, train=True, make_random=True, val_ratio=0.3, is_cityscape=False):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.is_cityscape = is_cityscape
        self.addr = self.build()
        self.label_trans = utils.label_transform.cityscape2mine()
        if make_random:
            random.shuffle(self.addr)
        if train:
            self.train_addr = self.addr[0:int(len(self.addr)*val_ratio)]
            self.val_addr = self.addr[int(len(self.addr)*val_ratio):]
        else:
            self.train_addr = self.addr

    def build(self):
        sep = os.path.sep
        # ---------------   label_dir
        # ***/label/train/*/*labelIds.png
        res = []
        label_addr = glob.glob(self.label_dir + sep + '*' + sep + '*labelIds.png')
        for i in label_addr:
            cityname = i.split(sep)[-2]
            imgname_temp = i.split(sep)[-1].split('_')[:3]
            img_name = imgname_temp[0] + '_' + imgname_temp[1] + '_' + imgname_temp[2] + '_leftImg8bit.png'
            res.append({
                'img_addr':self.data_dir + sep + cityname + sep + img_name,
                'label_addr':i
            })
        return res

    def train_generator(self):
        # for train and eval
        idx = 0
        while 1:
            if idx == len(self.train_addr):
                random.shuffle(self.train_addr)
                idx = 0
            img = io.imread(self.train_addr[idx]['img_addr'])
            label = io.imread(self.train_addr[idx]['label_addr'])
            if self.is_cityscape:
                label = self.label_trans.img_label_trans(label)
            idx += 1
            yield (img, label)

    def val_generator(self):
        # for train only
        idx = 0
        while 1:
            if idx == len(self.val_addr):
                random.shuffle(self.val_addr)
                idx = 0
            img = io.imread(self.val_addr[idx]['img_addr'])
            label = io.imread(self.val_addr[idx]['label_addr'])
            if self.is_cityscape:
                label = self.label_trans.img_label_trans(label)
            idx += 1
            yield (img, label)


if __name__ == '__main__':
    data = Dataset('./data/cityscape/img/train', './data/cityscape/label/train', is_cityscape=True)
    d = data.train_generator()
    print(next(d))