# !/usr/bin/python
# -*- coding:utf-8 -*-
import os
import shutil
from sklearn.externals import joblib
import codecs
import numpy as np
import voc_eval
import tensorflow as tf

Label = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
Label_dict = {}
c = 0
for i in Label:
    Label_dict[i] = c
    Label_dict[c] = i
    c += 1

e = 1e-2


def create_res_txt(resFile, dir):
    Res = joblib.load(resFile)
    names = Res.keys()
    names = sorted(names)
    for cls in Label:
        cls_id = Label_dict[cls]
        # print(cls_id)
        with codecs.open('{}/{}.txt'.format(dir, cls), 'wb', 'utf-8') as f:
            for name in names:
                res = Res[name]
                res = res[res[:, 4] > e]
                res = res[res[:, -1] == cls_id]
                for r in res:
                    s = str(name) + ' ' + str(r[4]) + ' ' + str(r[0]) + ' ' + str(r[1]) + ' ' + str(r[2]) + ' ' + str(
                        r[3]) + '\n'
                    f.write(s)
    pass


def mAP(resFile, dir, annopath, imagesetfile, cachedir, F):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    create_res_txt(resFile, dir)
    AP = np.zeros(20)
    i = 0
    for cls in Label:
        _, _, ap = voc_eval.voc_eval(os.path.join(dir, '{}.txt'), annopath, imagesetfile, cls, cachedir,
                                     use_07_metric=F)
        AP[i] = ap
        i += 1
        # print(i)
    print(AP)
    me = AP.mean() * 100
    print(dir, me)
    print('*********%.1f*********'%me)

    pass


if __name__ == "__main__":
    annopath = '/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCtest2007/VOCdevkit/VOC2007/Annotations/{}.xml'
    imagesetfile = 'imagesetfile.txt'
    cachedir = './'
    F = True
    print(__file__)

    resFile = '/home/zhai/PycharmProjects/Demo35/SSD_mobile/test_2/SSD300_2x.pkl'
    # resFile='/home/zhai/PycharmProjects/Demo35/SSD_4/test/SSD300_2x.pkl'
    dir = 'SSD300_2x'
    mAP(resFile, dir, annopath, imagesetfile, cachedir, F)
    mAP(resFile, dir, annopath, imagesetfile, cachedir, False)
    print('==========================================================')

