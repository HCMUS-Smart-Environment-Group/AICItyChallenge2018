#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.resnet_v1 import resnetv1

# CLASSES = ('__background__', 'car')

CLASSES = ('__background__',  # always index 0
                     'dog', 'person', 'cat', 
                     'tv', 'car', 'meatballs', 
                     'marinara sauce', 'tomato soup', 'chicken noodle soup',
                     'french onion soup', 'chicken breast', 'ribs', 
                     'pulled pork', 'hamburger', 'cavity')


NETS = {
    'res101': ('res101_faster_rcnn_iter_10000.ckpt',),}

DATASETS= {
    'car_track1': ('car_track1_train',),
    'tiny_car_track1':('tiny_car_track1_train',),
    'horizontal_car_track1': ('horizontal_car_track1_train',)}

VIDEO_DIR = 'data/Track1'
OUTPUT_DIR = 'output_bbox'

def store_bbox(im, class_name, dets, infor, idx, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        info.append(np.append(dets[i], idx))

def extract_bbox(sess, net, im, idx, info):
    scores, boxes = im_detect(sess, net, im)
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
            cls_scores[:, np.newaxis])).astype(np.float32)

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        store_bbox(im, cls, dets, info, idx, thresh=CONF_THRESH)

if __name__ == '__main__':

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    demonet = 'res101'
    dataset = 'car_track1'
    tfmodel = os.path.join(
        'output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
            'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError

    net.create_architecture("TEST", len(CLASSES),
        tag='default', anchor_scales=[2, 4, 8, 16])

    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    video_names = os.listdir(VIDEO_DIR)

    for video_name in video_names:
        if (video_name[:4] in ['Loc1', 'Loc2', 'Loc4']):

            video_path = os.path.join(VIDEO_DIR, video_name)
            video = cv2.VideoCapture(video_path)

            info = []
            idx = 1
            while video.isOpened():
                isReadable, frame = video.read()

                extract_bbox(sess, net, frame, idx, info)
                idx += 1

            np.save(os.path.join(OUTPUT_DIR, video_name[:-4]), np.array(info))
