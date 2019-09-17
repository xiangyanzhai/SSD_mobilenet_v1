# !/usr/bin/python
# -*- coding:utf-8 -*-
# !/usr/bin/python
# -*- coding:utf-8 -*-
from sklearn.externals import joblib
from chainercv.evaluations import eval_detection_voc
import numpy as np
GT=joblib.load('newGT.pkl')


def eval(Res,m=4952,F=True):
    pred_bboxes=[]
    pred_labels=[]
    pred_scores=[]
    gt_bboxes=[]
    gt_labels=[]
    names=GT.keys()
    names=sorted(names)
    for name in names[:m]:

        res = Res[name]

        res=res[res[:,4]>1e-10]


        p_bboxes=res[:,:4]

        p_labels=res[:,5]
        p_labels=p_labels.astype(np.int32)
        p_scores=res[:,4]
        pred_bboxes.append(p_bboxes)
        pred_labels.append(p_labels)
        pred_scores.append(p_scores)


        gt=GT[name]
        gt=gt[gt[:,-1]==0]

        g_bboxes = gt[:,:4]
        g_labels=gt[:,4]-1
        g_labels=g_labels.astype(np.int32)

        gt_bboxes.append(g_bboxes)
        gt_labels.append(g_labels)
    result=eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,use_07_metric=F)

    print(result)
from sklearn.externals import joblib
if __name__ == "__main__":
    m=4952
    # lr=0.001,img_max=1000,img_min=600
    res=joblib.load('FPN_101.pkl')
    eval(res,m,F=True)
    eval(res,m,F=False)
    print('===============================')
    # res = joblib.load('Faster_vgg16_m_2.pkl')
    # eval(res, m, F=True)
    # eval(res, m, F=False)
    # print('===============================')
    #last
    # m=4952
    # # lr=0.001,img_max=1000,img_min=600
    # res=joblib.load('Faster_vgg16_2.pkl')
    # eval(res,m,F=True)
    # eval(res,m,F=False)
    # print('===============================')
    # # lr=0.001,img_max=1000,img_min=600
    # res = joblib.load('Faster_101_3.pkl')
    # eval(res, m, F=True)
    # eval(res, m, F=False)
    # print('===============================')
    # # lr=0.00125,img_max=1333,img_min=800,
    # # ATC include the anchor boxes that are outside the image for training,
    # # PTC n_sample=256
    # res = joblib.load('Faster_101_FPN_4_2.pkl')
    # eval(res, m, F=True)
    # eval(res, m, F=False)

    pass
