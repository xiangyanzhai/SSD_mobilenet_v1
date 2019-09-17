# !/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf

iou_thresh = None


def loc2bbox(pre_loc, anchor):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    c_yx = anchor[..., :2] + c_hw / 2
    yx = pre_loc[..., :2] * c_hw + c_yx
    hw = tf.exp(pre_loc[..., 2:4]) * c_hw
    yx1 = yx - hw / 2
    yx2 = yx + hw / 2
    bboxes = tf.concat((yx1, yx2), axis=-1)
    return bboxes


def py_inds(score, inds):
    score[inds] = score[inds] * -1
    inds = score > 0
    score[inds] = 0
    score = score * -1

    return score


def fn_map(x):
    bboxes = x[0]
    score = x[1]

    inds = tf.image.non_max_suppression(bboxes, score, tf.shape(bboxes)[0], iou_threshold=iou_thresh)
    score = tf.py_func(py_inds, [score, inds], tf.float32)

    return score

# def fn_map(x):
#     bboxes = x[0]
#     score = x[1]
#     m = tf.shape(score)
#     inds = tf.image.non_max_suppression(bboxes, score, tf.shape(bboxes)[0], iou_threshold=iou_thresh)
#     inds = inds[..., None]
#     mask = tf.scatter_nd(inds, tf.ones(tf.shape(inds)[0]), m)
#     score = score * (tf.to_float(mask))
#     return score


def predict(bboxes, score, num_cls=20, size=300, iou_thresh_=0.45, c_thresh=1e-2):
    # m*cls*4
    # m*cls
    global iou_thresh
    iou_thresh = iou_thresh_
    score = tf.nn.softmax(score)
    C = 1 - score[:, 0]
    inds = C > c_thresh
    bboxes = tf.boolean_mask(bboxes, inds)
    score = tf.boolean_mask(score, inds)
    bboxes = tf.clip_by_value(bboxes, 0, size)
    mbboxes = tf.expand_dims(bboxes, axis=0)
    mbboxes = tf.tile(mbboxes, [num_cls, 1, 1])
    score = tf.transpose(score)
    score = score[1:]
    # print(mbboxes,score)
    score = tf.map_fn(fn_map, [mbboxes, score], back_prop=False, dtype=tf.float32)

    score = tf.transpose(score)
    cls = tf.argmax(score, axis=1)
    score = tf.reduce_max(score, axis=1)
    cls = tf.to_float(cls)
    score = tf.reshape(score, (-1, 1))
    cls = tf.reshape(cls, (-1, 1))

    pre = tf.concat([bboxes, score, cls], axis=1)
    # pre = tf.boolean_mask(pre, pre[:, -2] > c_thresh)
    _, top_k = tf.nn.top_k(pre[:, -2], tf.shape(pre)[0])
    pre = tf.gather(pre, top_k)
    return pre
