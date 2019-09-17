# !/usr/bin/python
# -*- coding:utf-8 -*-


import tensorflow as  tf


def bbox_flip_left_right(bboxes, w):
    w = tf.to_float(w)
    x1, y1, x2, y2, cls = tf.split(bboxes, 5, axis=1)
    x1, x2 = w - 1. - x2, w - 1. - x1

    return tf.concat([x1, y1, x2, y2, cls], axis=1)


#
# def handle_im(im, bboxes):
#     im = im[None]
#     H = tf.shape(im)[1]
#     W = tf.shape(im)[2]
#     H = tf.to_float(H)
#     W = tf.to_float(W)
#     ma = tf.reduce_max([H, W])
#     mi = tf.reduce_min([H, W])
#     scale = tf.reduce_min([config.read_img_size / ma, config.read_img_size / mi])
#     nh = H * scale
#     nw = W * scale
#     nh = tf.to_int32(nh)
#     nw = tf.to_int32(nw)
#     im = tf.image.resize_images(im, (nh, nw))
#     bboxes = tf.concat([bboxes[..., :4] * scale, bboxes[..., 4:]], axis=-1)
#
#     # im = tf.pad(im, [[0, 0], [0, config.read_img_size - nh], [0, config.read_img_size - nw], [0, 0]], constant_values=127)
#
#     im = im[0]
#     return im, bboxes


def handle_im(im, bboxes):
    im = im[None]
    H = tf.shape(im)[1]
    W = tf.shape(im)[2]
    H = tf.to_float(H)
    W = tf.to_float(W)
    sh = config.read_img_size / H
    sw = config.read_img_size / W
    scale = tf.concat([[sw], [sh], [sw], [sh]], axis=0)

    # ma = tf.reduce_max([H, W])
    # mi = tf.reduce_min([H, W])
    # scale = tf.reduce_min([config.read_img_size / ma, config.read_img_size / mi])

    im = tf.image.resize_images(im, (config.read_img_size, config.read_img_size))
    bboxes = tf.concat([bboxes[..., :4] * scale, bboxes[..., 4:]], axis=-1)

    # im = tf.pad(im, [[0, 0], [0, config.read_img_size - nh], [0, config.read_img_size - nw], [0, 0]], constant_values=127)

    im = im[0]
    return im, bboxes


def pad(im, y, x, t, ):
    a = tf.zeros(shape=(1, y, t, 3)) + config.Mean
    b = tf.zeros(shape=(1, config.read_img_size - t - y, t, 3)) + config.Mean
    im = tf.concat([a, im, b], axis=1)
    a = tf.zeros(shape=(1, config.read_img_size, x, 3)) + config.Mean
    b = tf.zeros(shape=(1, config.read_img_size, config.read_img_size - t - x, 3)) + config.Mean
    im = tf.concat([a, im, b], axis=2)
    return im
    pass
def img_scale(im, bboxes):
    im, bboxes = tf.cond(tf.random_uniform(shape=()) > 0.5, lambda: (
        tf.image.flip_left_right(im), bbox_flip_left_right(bboxes, tf.to_float(tf.shape(im)[1]))), lambda: (im, bboxes))
    scale = tf.constant(config.img_scale_size, dtype=tf.float32)
    index = tf.random_uniform(shape=(), maxval=tf.shape(scale)[0], dtype=tf.int32)
    read_img_size = scale[index]

    im = im[None]
    H = tf.shape(im)[1]
    W = tf.shape(im)[2]
    H = tf.to_float(H)
    W = tf.to_float(W)
    sh = read_img_size / H
    sw = read_img_size / W
    scale = tf.concat([[sw], [sh], [sw], [sh]], axis=0)

    t = tf.to_int32(read_img_size)
    im = tf.image.resize_images(im, (t, t))
    x = tf.random_uniform(shape=(), maxval=config.read_img_size - t, dtype=tf.int32)
    y = tf.random_uniform(shape=(), maxval=config.read_img_size - t, dtype=tf.int32)
    xyxy = tf.concat([[x], [y], [x], [y]], axis=0)
    xyxy = tf.to_float(xyxy)
    bboxes = tf.concat([bboxes[..., :4] * scale + xyxy, bboxes[..., 4:]], axis=-1)

    # im = tf.pad(im,
    #             [[0, 0], [y, config.read_img_size - t - y], [x, config.read_img_size - t - x],
    #              [0, 0]], constant_values=127)
    im=pad(im,y,x,t,)

    im = im[0]

    return im, bboxes


def crop_img_tf(img, bboxes):
    jitter_ratio = tf.constant(config.jitter_ratio, dtype=tf.float32)
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    img, bboxes = tf.cond(tf.random_uniform(shape=()) > 0.5, lambda: (
        tf.image.flip_left_right(img), bbox_flip_left_right(bboxes, tf.to_float(w))), lambda: (img, bboxes))
    ori_img = img
    ori_bboxes = bboxes

    index = tf.random_uniform(shape=(), maxval=len(config.jitter_ratio), dtype=tf.int32)
    a = tf.to_int32(tf.to_float(h) * jitter_ratio[index])
    b = tf.to_int32(tf.to_float(w) * jitter_ratio[index])
    h1 = tf.random_uniform(shape=(), maxval=a, dtype=tf.int32)
    h2 = tf.random_uniform(shape=(), maxval=a - h1, dtype=tf.int32)
    w1 = tf.random_uniform(shape=(), maxval=b, dtype=tf.int32)
    w2 = tf.random_uniform(shape=(), maxval=b - w1, dtype=tf.int32)

    h2 = h - h2
    w2 = w - w2
    img = img[h1:h2, w1:w2]

    w1 = tf.to_float(w1)
    w2 = tf.to_float(w2)
    h1 = tf.to_float(h1)
    h2 = tf.to_float(h2)
    x1 = tf.maximum(w1, bboxes[:, 0:1])
    y1 = tf.maximum(h1, bboxes[:, 1:2])
    x2 = tf.minimum(w2 - 1, bboxes[:, 2:3])
    y2 = tf.minimum(h2 - 1, bboxes[:, 3:4])

    x1 = x1 - w1
    y1 = y1 - h1
    x2 = x2 - w1
    y2 = y2 - h1

    areas1 = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    w = tf.maximum(0., x2 - x1)
    h = tf.maximum(0., y2 - y1)
    areas2 = w * h
    areas2 = tf.reshape(areas2, (-1,))

    inds = (areas2 / areas1) >= config.crop_iou

    bboxes = tf.concat([x1, y1, x2, y2, bboxes[:, 4:5]], axis=1)
    bboxes = tf.boolean_mask(bboxes, inds)

    # a=tf.py_func(py_print,[a],tf.float32)

    flag = tf.equal(tf.shape(bboxes)[0], 0) | (tf.random_uniform(shape=()) < config.keep_ratio)
    img, bboxes = tf.cond(flag, lambda: (ori_img, ori_bboxes), lambda: (img, bboxes))

    img, bboxes = handle_im(img, bboxes)

    return img, bboxes


def distort_color(image, color_ordering=0, fast_mode=False, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def img_distort(img):
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    index = tf.random_uniform(shape=(), maxval=4, dtype=tf.int32)
    img = tf.case({tf.equal(index, 0): lambda: distort_color(img, 0),
                   tf.equal(index, 1): lambda: distort_color(img, 1),
                   tf.equal(index, 2): lambda: distort_color(img, 2),
                   tf.equal(index, 3): lambda: distort_color(img, 3)})
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    return img


def parse(se):
    features = tf.parse_single_example(
        se, features={
            'im': tf.FixedLenFeature([], tf.string),
            'bboxes': tf.FixedLenFeature([], tf.string),
            'Id': tf.FixedLenFeature([], tf.string),
        }
    )
    img = features['im']
    bboxes = features['bboxes']
    Id = features['Id']
    img = tf.image.decode_jpeg(img, channels=3)
    bboxes = tf.decode_raw(bboxes, tf.float32)
    bboxes = tf.reshape(bboxes, (-1, 5))

    # xyxy
    bboxes = tf.gather(bboxes, [1, 0, 3, 2, 4], axis=1)
    img = tf.cond(tf.random_uniform(shape=()) > config.keep_ratio, lambda: img_distort(img), lambda: img)
    img, bboxes = crop_img_tf(img, bboxes)
    img, bboxes = tf.cond(tf.random_uniform(shape=()) > 0.5, lambda: (img, bboxes),
                          lambda: img_scale(img, bboxes))
    num = tf.shape(bboxes)[0]
    bboxes = tf.gather(bboxes, [1, 0, 3, 2, 4], axis=1)
    return img, bboxes, num


def readData(files, config_, batch_size=1, num_epochs=2000, num_threads=16, shuffle_buffer=1024, num_shards=1,
             shard_index=0):
    global config
    config = config_
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.shard(num_shards, shard_index)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(parse, num_parallel_calls=num_threads)

    if shuffle_buffer is not None:
        dataset = dataset.shuffle(shuffle_buffer)
    # dataset = dataset.padded_batch(batch_size, padded_shapes=([None, None, 3], [None, 5], []))
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, None, 3], [None, 5], []),
                                   padding_values=(0., 0., 0))

    iter = dataset.make_initializable_iterator()
    return iter


def draw_gt(im, gt):
    im = im.astype(np.uint8)
    boxes = gt.astype(np.int32)
    print('***', boxes.shape)
    for box in boxes:
        # print(box)
        x1, y1, x2, y2 = box[:4]
        y1, x1, y2, x2 = box[:4]
        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255))
    im = im.astype(np.uint8)
    im = im[..., ::-1]
    print(im.shape)
    cv2.imshow('a', im)
    cv2.waitKey(2000)
    return im


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from yolo.tool.config import Config
    from myDNN.SSD_tf.tool.config import Config
    import numpy as np
    import cv2

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    path = '/home/zhai/PycharmProjects/Demo35/data_set_yxyx/'
    files = [path + 'voc_07.tf', path + 'voc_12.tf']
    Mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    config = Config(None, Mean, files, img_size=300, jitter_ratio=[0.3,0.5,0.7], crop_iou=0.45,
                    img_scale_size=[212,150,106,75])

    Iter = readData(config.files, config, batch_size=8, num_threads=16, shuffle_buffer=None,
                    num_shards=1, shard_index=0)
    im, bboxes, nums = Iter.get_next()
    with tf.Session(config=tf_config) as sess:

        sess.run(Iter.initializer)
        for j in range(2000):
            img, box, num = sess.run([im, bboxes, nums])
            if j > 300:
                for i in range(img.shape[0]):
                    a = img[i]
                    b = box[i]
                    c = num[i]
                    draw_gt(a, b[:c])
                    print(c,a[1,1])
                pass
    pass
