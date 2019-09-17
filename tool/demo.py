# !/usr/bin/python
# -*- coding:utf-8 -*-
# b'000000080690'

from pycocotools.coco import COCO
train_file = '/home/zhai/PycharmProjects/Demo35/dataset/coco/instances_valminusminival2014.json'
# train_file='/home/zhai/PycharmProjects/Demo35/dataset/coco/annotations/instances_train2017.json'
cocoGt = COCO(train_file)
i=80690
AnnIds = cocoGt.getAnnIds([i], )
Anns = cocoGt.loadAnns(AnnIds)
print(len(AnnIds),len(Anns))
print(AnnIds)
for ann in Anns:
    print(ann)
if __name__ == "__main__":
    pass