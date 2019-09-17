# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np

num_anchors = None
map_size = None
stride_size = None
box = None
H = []


def get_coord(N):
    t = np.arange(N)
    x, y = np.meshgrid(t, t)
    x = x[..., None]
    y = y[..., None]

    coord = np.concatenate((y, x, y, x), axis=-1)
    coord = coord[:, :, None, :]
    return coord


def generate_anchor_base(base_size, index, num=4):
    ratios = [1, 1.6, 2, 3, 1 / 1.6, 1 / 2, 1 / 3]
    if num==6:
        ratios = [1,  2, 3,  1 / 2, 1 / 3]
    if num==4 or num==3:
        ratios = [1, 2, 1 / 2]
    py = base_size / 2
    px = base_size / 2

    anchors_base = []
    for ratio in ratios:
        h = box[index] * np.sqrt(1 / ratio)
        H.append(h)
        w = box[index] * np.sqrt(ratio)
        anchors_base.append([py - h / 2, px - w / 2, py + h / 2, px + w / 2])
    if num==6:
        t = np.sqrt(box[index] * box[index + 1]) / 2
        H.append(t)
        anchors_base.append([py - t, px - t, py + t, px + t])
    anchors_base = np.array(anchors_base, dtype=np.float32)
    return anchors_base

    pass


def create_anchors():
    Anchors = np.zeros((0, 4))
    for i in range(len(num_anchors)):
        anchors_base = generate_anchor_base(stride_size[i], i, num=num_anchors[i])
        coord = get_coord(map_size[i])
        tanchors = coord * stride_size[i] + anchors_base
        tanchors = tanchors.reshape(-1, 4)
        Anchors = np.concatenate((Anchors, tanchors))
    Anchors = Anchors.astype(np.float32)
    return Anchors


def get_box(s_min, s_max, image_size=300, m=6):
    stride = int((int(s_max * 100) - int(s_min * 100)) / (m - 2))
    bbox = np.zeros(m + 1)
    bbox[0] = int(s_min * 100 / 2) / 100
    bbox[1:] = (s_min + np.arange(m) * stride / 100)
    print(bbox)

    return bbox * image_size

    pass


def get_Anchors(img_size=321, s_min_=0.15, s_max_=0.9, num_anchors_=[8, 8, 8, 8, 8, 8], map_size_=[40, 20, 10, 5, 3, 1],
                stride_size_=[8, 16, 32, 64, 107, 321]):
    global num_anchors, map_size, stride_size, box
    num_anchors = num_anchors_
    map_size = map_size_
    stride_size = stride_size_
    box = get_box(s_min_, s_max_, img_size)
    print(box)
    Anchors = create_anchors()
    return Anchors


def bbox2c_bbox(bboxes):
    y = (bboxes[:, 2:3] + bboxes[:, 0:1]) / 2
    x = (bboxes[:, 3:4] + bboxes[:, 1:2]) / 2
    h = bboxes[:, 2:3] - bboxes[:, 0:1]
    w = bboxes[:, 3:4] - bboxes[:, 1:2]
    return y, x, h, w
def get_cAnchors(img_size=300, s_min_=0.2, s_max_=0.9, num_anchors_=[8, 8, 8, 8, 8, 8], map_size_=[40, 20, 10, 5, 3, 1],
                stride_size_=[8, 16, 32, 64, 107, 321]):
    global num_anchors, map_size, stride_size, box
    num_anchors = num_anchors_
    map_size = map_size_
    stride_size = stride_size_
    box = get_box(s_min_, s_max_, img_size)
    print(box)
    Anchors = create_anchors()
    y, x, h, w = bbox2c_bbox(Anchors)
    for i in H:
       print(i)
    return np.concatenate((y, x, h, w), axis=-1)

if __name__ == "__main__":
    # Anchors = get_Anchors()
    # print(Anchors.shape)
    cAnchors=get_cAnchors()
    print(cAnchors.shape)


    pass
