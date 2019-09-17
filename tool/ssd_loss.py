# !/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf


def cal_IOU(pre_bboxes, bboxes):
    hw = pre_bboxes[:, 2:4] - pre_bboxes[:, :2]
    areas1 = tf.reduce_prod(hw, axis=-1)

    hw = bboxes[:, 2:4] - bboxes[:, :2]
    areas2 = tf.reduce_prod(hw, axis=-1)

    yx1 = tf.maximum(pre_bboxes[:, None, :2], bboxes[:, :2])
    yx2 = tf.minimum(pre_bboxes[:, None, 2:4], bboxes[:, 2:4])
    hw = yx2 - yx1
    hw = tf.maximum(hw, 0)
    areas_i = tf.reduce_prod(hw, axis=-1)
    iou = areas_i / (areas1[:, None] + areas2 - areas_i)
    return iou


def bbox2loc(anchor, bbox):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    c_yx = anchor[..., :2] + c_hw / 2
    hw = bbox[..., 2:4] - bbox[..., 0:2]
    yx = bbox[..., :2] + hw / 2
    t_yx = (yx - c_yx) / c_hw
    t_hw = tf.log(hw / c_hw)
    return tf.concat([t_yx, t_hw], axis=1)


def ATC(anchors, bboxes, C, p_thresh=0.5, n_thresh_hi=0.5, n_thresh_lo=0.0, P_max=512):
    IOU = cal_IOU(anchors, bboxes)
    inds_box = tf.argmax(IOU, axis=1, output_type=tf.int32)
    inds = tf.range(tf.shape(anchors)[0])
    inds = tf.concat([tf.reshape(inds, (-1, 1)), tf.reshape(inds_box, (-1, 1))], axis=1)
    iou = tf.gather_nd(IOU, inds)
    indsP1 = iou >= p_thresh
    indsN = (iou >= n_thresh_lo) & (iou < n_thresh_hi)

    t = tf.reduce_max(IOU, axis=0)
    t = tf.equal(IOU, t)
    indsP2 = tf.reduce_any(t, axis=1)

    inds_gt_box = tf.argmax(tf.to_int32(t), axis=1, output_type=tf.int32)
    inds_box = inds_box * tf.to_int32(~indsP2) + inds_gt_box

    indsP = indsP1 | indsP2
    indsN = indsN & (~indsP2)

    indsP = tf.where(indsP)[:, 0]
    indsN = tf.where(indsN)[:, 0]
    indsP = tf.random_shuffle(indsP)[:]
    p_num = tf.shape(indsP)[0]
    C_N = tf.gather(C, indsN)

    n_num = tf.minimum(3 * p_num, tf.shape(indsN)[0])
    p_num = tf.to_int32(tf.to_float(n_num) / 3)
    indsP = indsP[:p_num]
    _, inds_C_N = tf.nn.top_k(-C_N, n_num, )
    indsN = tf.gather(indsN, inds_C_N)

    anchor = tf.gather(anchors, indsP)
    bbox = tf.gather(bboxes, tf.gather(inds_box, indsP))
    labelP = bbox[..., -1] + 1
    loc = bbox2loc(anchor, bbox)
    loc = loc / tf.constant([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)
    labelN = tf.zeros(n_num)
    label = tf.concat((labelP, labelN), axis=0)
    label = tf.to_int32(label)
    inds = tf.concat([indsP, indsN], axis=0)
    return label, inds, indsP, loc


def softmaxloss(score, label):
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score, labels=label))
    return loss


def SmoothL1Loss(net_t, input_T, sigma):
    absx = tf.abs(net_t - input_T)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return tf.reduce_sum(r)


def loss(net, anchors, bboxes):
    pre_cls = tf.nn.softmax(net[..., :-4])
    pre_loc = net[..., -4:]
    label, inds, indsP, loc = ATC(anchors, bboxes, pre_cls[..., 0], )
    pre_cls = tf.gather(net[..., :-4], inds)
    pre_loc = tf.gather(pre_loc, indsP)
    N = tf.shape(indsP)[0]
    loss = softmaxloss(pre_cls, tf.to_int32(label)) + SmoothL1Loss(pre_loc, loc, 1.0)
    # loss = tf.cond(tf.equal(N, 0), lambda: (Zero, Zero), lambda: (loss, tf.to_float(N)))
    return loss,tf.to_float(N)


if __name__ == "__main__":
    pass
