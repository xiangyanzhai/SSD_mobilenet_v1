# !/usr/bin/python
# -*- coding:utf-8 -*-

# img_size = 321
# s_min = 0.15
# s_max = 0.9
# num_cls = 20
# num_anchors = [8, 8, 8, 8, 8, 8]
# map_size = [40, 20, 10, 5, 3, 1]
# stride_size = [8, 16, 32, 64, 107, 321]
class Config():
    def __init__(self, is_train, Mean, files, pre_model, lr=1e-3, weight_decay=0.0001,
                 num_cls=20, img_size=300, s_min=0.2, s_max=0.9, num_anchors=[4, 6, 6, 6, 4, 4],
                 stride_size=[8, 16, 32, 64, 100, 300], map_size=[38, 19, 10, 5, 3, 1],
                 batch_size_per_GPU=16, gpus=1,
                 bias_lr_factor=1,
                 crop_iou=0.45,
                 keep_ratio=0.2,
                 jitter_ratio=[0.3, 0.5, 0.7],
                 img_scale_size=[212, 150, 106, 75],
                 ):
        self.is_train = is_train
        self.Mean = Mean
        self.files = files
        self.pre_model = pre_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_cls = num_cls
        self.img_size = img_size
        self.s_min = s_min
        self.s_max = s_max
        self.num_anchors = num_anchors
        self.map_size = map_size
        self.stride_size = stride_size

        self.batch_size_per_GPU = batch_size_per_GPU
        self.gpus = gpus
        self.bias_lr_factor = bias_lr_factor
        self.crop_iou = crop_iou
        self.keep_ratio = keep_ratio
        self.jitter_ratio = jitter_ratio
        self.img_scale_size = img_scale_size
        self.read_img_size = img_size
        print('SSD')
        print('==============================================================')
        print('is_train:\t', self.is_train)
        print('Mean:\t', self.Mean)
        print('files:\t', self.files)
        print('pre_model:\t', self.pre_model)
        print('lr:\t', self.lr)
        print('weight_decay:\t', self.weight_decay)
        print('num_cls:\t', self.num_cls)
        print('img_size:\t', self.img_size)
        print('s_min:\t', self.s_min)
        print('s_max:\t', self.s_max)
        print('num_anchors:\t', num_anchors)
        print('map_size:\t', map_size)
        print('stride_size:\t', stride_size)

        print('batch_size_per_GPU:\t', self.batch_size_per_GPU)
        print('gpus:\t', self.gpus)

        print('bias_lr_factor:\t', self.bias_lr_factor)
        print('==============================================================')
        print('crop_iou:\t', self.crop_iou)
        print('keep_ratio:\t', self.keep_ratio)
        print('jitter_ratio:\t', self.jitter_ratio)
        print('img_scale:\t', self.img_scale_size)
        print('==============================================================')
